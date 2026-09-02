"""Anchored coordinate edits over sparse Jacobian-lens decompositions."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import torch

from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    _SWAP_WARN_COSINE,
    DEFAULT_K,
    JSpaceDecomposition,
    get_sparse_decomposition,
)

#: Above this condition number the column-normalized edit basis is treated as ill-conditioned and
#: the patch warns. It sits at the inverse square-root of float32 epsilon so a benign, well-scaled
#: basis (condition number of order one) never trips the warning.
_BASIS_CONDITION_WARNING = 1.0 / math.sqrt(torch.finfo(torch.float32).eps)


@dataclass(frozen=True)
class CoordinatePatch:
    """Result of an anchored edit to sparse J-space coordinates.

    All tensors are detached. Vector outputs use float32 on the dictionary device; the two support
    tensors are ``torch.long`` on CPU. This is a report only: it retains no model, lens, tokenizer,
    hook, or full dictionary.

    Attributes:
        support_before: Active support from the sparse decomposition, in decomposition order.
        support_after: The coordinate frame the edit acts on: ``support_before`` in the same order,
            with an absent target appended. Both coordinate tensors align with this frame.
        coordinates_before: Original coordinates in ``support_after``, including an appended zero
            when the target was absent.
        coordinates_after: Coordinates after applying ``mode`` and ``alpha``.
        source_slot: Position of the source atom within ``support_after``.
        target_slot: Position of the target atom within ``support_after``.
        target_was_appended: Whether the target was absent and appended to ``support_after``.
        overwritten_target_coordinate: The discarded old target coordinate when a ``substitute``
            overwrites an already-active target; ``None`` when the target was absent or ``mode`` is
            ``"swap"`` (a swap relocates a coordinate and discards nothing).
        reconstruction_before: Sparse reconstruction from ``support_before``.
        reconstruction_after: Reconstruction from ``support_after`` and ``coordinates_after``.
        residual: Anchored residual ``x - reconstruction_before``. This is not necessarily the
            decomposition's orthogonal ``non_j_space_component``.
        delta: ``reconstruction_after - reconstruction_before``.
        patched: ``x + delta``; an unchanged clone of the float32 input when ``alpha`` is zero.
        nonedited_coordinate_max_delta: Maximum absolute change across every coordinate that is
            neither source nor target. A postcondition witness: it must be numerically zero.
        residual_max_delta: Maximum absolute difference between ``patched - reconstruction_after``
            and ``residual``. A postcondition witness: the anchored residual must be preserved.
        source_target_cosine: Signed cosine between the source and target atoms.
        basis_rank: Numerical rank of the column-normalized edit basis at float32 precision.
        basis_condition_number: Condition number of that basis, or infinity when rank deficient.
    """

    support_before: torch.Tensor
    support_after: torch.Tensor
    coordinates_before: torch.Tensor
    coordinates_after: torch.Tensor
    source_slot: int
    target_slot: int
    target_was_appended: bool
    overwritten_target_coordinate: Optional[float]
    reconstruction_before: torch.Tensor
    reconstruction_after: torch.Tensor
    residual: torch.Tensor
    delta: torch.Tensor
    patched: torch.Tensor
    nonedited_coordinate_max_delta: float
    residual_max_delta: float
    source_target_cosine: float
    basis_rank: int
    basis_condition_number: float


def _validate_inputs(
    x: torch.Tensor, dictionary: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if dictionary.ndim != 2:
        raise ValueError(
            f"dictionary must be 2-D [num_atoms, d_model], got shape {tuple(dictionary.shape)}"
        )
    num_atoms, d_model = dictionary.shape
    if x.ndim != 1 or x.shape[0] != d_model:
        raise ValueError(f"x must be 1-D of length d_model={d_model}, got shape {tuple(x.shape)}")
    if num_atoms == 0:
        raise ValueError("dictionary must contain at least one atom")
    if torch.is_complex(x) or torch.is_complex(dictionary):
        raise ValueError("x and dictionary must be real-valued")
    if x.device != dictionary.device:
        raise ValueError("x and dictionary must be on the same device")
    target = x.float()
    if not bool(torch.isfinite(target).all()):
        raise ValueError("x contains non-finite entries")
    return target, dictionary


def _validate_atom_rows(
    dictionary: torch.Tensor, indices: torch.Tensor, *, context: str
) -> torch.Tensor:
    atoms = dictionary[indices].float()
    if not bool(torch.isfinite(atoms).all()):
        raise ValueError(f"{context} contains a non-finite dictionary atom")
    if bool((torch.linalg.vector_norm(atoms, dim=1) == 0).any()):
        raise ValueError(f"{context} contains a zero-norm dictionary atom")
    return atoms


def _validate_index(name: str, value: int, num_atoms: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer atom index")
    index = int(value)
    if not 0 <= index < num_atoms:
        raise ValueError(f"{name}={index} out of range for dictionary with {num_atoms} atoms")
    return index


def _close(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape:
        return False
    magnitude = torch.stack([actual.abs().max(), expected.abs().max()]).max()
    tolerance = 32 * torch.finfo(torch.float32).eps
    return bool(
        torch.allclose(actual, expected, rtol=tolerance, atol=tolerance * float(magnitude.item()))
    )


def _validate_decomposition(
    decomposition: JSpaceDecomposition,
    target: torch.Tensor,
    dictionary: torch.Tensor,
    target_idx: int,
) -> None:
    num_atoms, d_model = dictionary.shape
    support = decomposition.support
    selected = decomposition.selected_support
    if support.ndim != 1 or selected.ndim != 1:
        raise ValueError("precomputed decomposition supports must be 1-D")
    if support.dtype != torch.long or selected.dtype != torch.long:
        raise ValueError("precomputed decomposition supports must have dtype torch.long")
    support_cpu = support.detach().cpu()
    selected_cpu = selected.detach().cpu()
    if (
        support_cpu.unique().numel() != support_cpu.numel()
        or selected_cpu.unique().numel() != selected_cpu.numel()
    ):
        raise ValueError("precomputed decomposition supports must not contain duplicates")
    if bool(((support_cpu < 0) | (support_cpu >= num_atoms)).any()) or bool(
        ((selected_cpu < 0) | (selected_cpu >= num_atoms)).any()
    ):
        raise ValueError("precomputed decomposition support is outside the dictionary")
    if not set(support_cpu.tolist()).issubset(set(selected_cpu.tolist())):
        raise ValueError("precomputed active support must be a subset of selected_support")
    referenced = torch.unique(
        torch.cat([support_cpu, selected_cpu, torch.tensor([target_idx], dtype=torch.long)])
    )
    _validate_atom_rows(dictionary, referenced, context="precomputed decomposition")
    coordinates = decomposition.coordinates.float().to(dictionary.device)
    if coordinates.ndim != 1 or coordinates.numel() != support_cpu.numel():
        raise ValueError("precomputed decomposition coordinates must align with active support")
    if not bool(torch.isfinite(coordinates).all()) or bool((coordinates <= 0).any()):
        raise ValueError("precomputed decomposition active coordinates must be finite and positive")
    vectors = (
        decomposition.reconstruction,
        decomposition.j_space_component,
        decomposition.non_j_space_component,
    )
    if any(vector.ndim != 1 or vector.shape[0] != d_model for vector in vectors):
        raise ValueError("precomputed decomposition vector outputs must have shape [d_model]")
    local_vectors = [vector.float().to(dictionary.device) for vector in vectors]
    if not all(bool(torch.isfinite(vector).all()) for vector in local_vectors):
        raise ValueError("precomputed decomposition contains non-finite vector outputs")
    support_atoms = dictionary[support_cpu].float()
    expected_reconstruction = support_atoms.T @ coordinates
    if not _close(local_vectors[0], expected_reconstruction):
        raise ValueError("precomputed decomposition reconstruction is incompatible with dictionary")
    if not _close(local_vectors[1] + local_vectors[2], target):
        raise ValueError("precomputed decomposition is incompatible with activation")
    if selected_cpu.numel():
        selected_atoms = dictionary[selected_cpu].float().T
        if target.device.type == "mps":
            work_atoms = selected_atoms.cpu()
            work_target = target.cpu()
            expected_component = (work_atoms @ (torch.linalg.pinv(work_atoms) @ work_target)).to(
                target.device
            )
        else:
            expected_component = selected_atoms @ (torch.linalg.pinv(selected_atoms) @ target)
    else:
        expected_component = torch.zeros_like(target)
    if not _close(local_vectors[1], expected_component):
        raise ValueError("precomputed decomposition selected span is incompatible with dictionary")


def _basis_diagnostics(basis: torch.Tensor) -> tuple[int, float]:
    """Rank and condition number of the column-normalized edit basis, warning when ill-conditioned.

    Both warnings are non-fatal: the anchored edit performs no inverse of ``basis``, so a
    rank-deficient or poorly conditioned basis only means coordinate attribution is non-unique.
    """
    normalized = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True)
    diagnostic_basis = normalized.detach().cpu().double()
    rank_rtol = torch.finfo(torch.float32).eps * max(diagnostic_basis.shape)
    rank = int(torch.linalg.matrix_rank(diagnostic_basis, rtol=rank_rtol).item())
    if rank < diagnostic_basis.shape[1]:
        condition = math.inf
        warnings.warn(
            "coordinate-patch basis is rank deficient (condition number=inf); "
            "coordinate attribution may be non-unique",
            UserWarning,
            stacklevel=3,
        )
    else:
        singular_values = torch.linalg.svdvals(diagnostic_basis)
        condition = float((singular_values.max() / singular_values.min()).item())
        if condition >= _BASIS_CONDITION_WARNING:
            warnings.warn(
                f"coordinate-patch basis is poorly conditioned (condition number={condition:.6g}); "
                "coordinate attribution may be non-unique",
                UserWarning,
                stacklevel=3,
            )
    return rank, condition


def _warn_if_near_parallel(pair_units: torch.Tensor) -> float:
    """Warn (never raise) when the source and target atoms are near-parallel; return their cosine.

    A swap or substitution between near-parallel atoms is approximately a no-op, so the user should
    hear it. Unlike ``JacobianLens.swap_hooks`` (which inverts a two-atom basis and therefore raises
    at the parallel extreme), the anchored patch performs no inverse and stays non-fatal.
    """
    cosine = float((pair_units[0] @ pair_units[1]).item())
    abs_cosine = abs(cosine)
    if abs_cosine >= _SWAP_WARN_COSINE:
        warnings.warn(
            f"coordinate-patch source and target atoms are near-parallel "
            f"(abs cosine={abs_cosine:.6f}); the edit is approximately a no-op",
            UserWarning,
            stacklevel=3,
        )
    return cosine


def solve_coordinate_patch(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    source_idx: int,
    target_idx: int,
    *,
    decomposition: Optional[JSpaceDecomposition] = None,
    k: int = DEFAULT_K,
    mode: str = "substitute",
    alpha: float = 1.0,
    algorithm: str = "nonnegative_orthogonal_matching_pursuit",
) -> CoordinatePatch:
    """Edit one sparse J-space coordinate while preserving its recovered frame.

    ``substitute`` zeros the source and replaces the target coordinate with the source value;
    ``swap`` exchanges the two values. An absent target is appended at zero. The edit is blended
    by ``alpha`` and reconstructed over the original residual ``x - reconstruction``.

    Args:
        x: Activation vector with shape ``[d_model]``.
        dictionary: Atom matrix with shape ``[num_atoms, d_model]``; rows are atoms.
        source_idx: Atom index that must occur in the decomposition's active support.
        target_idx: Distinct atom index to receive or exchange the source coordinate.
        decomposition: Optional decomposition of ``x`` under ``dictionary``. Reuse validates only
            the selected and edited rows, avoiding another full-dictionary scan.
        k: Sparse-solver upper bound when no decomposition is supplied.
        mode: ``"substitute"`` to overwrite the target, or ``"swap"`` to exchange coordinates.
        alpha: Finite interpolation strength. Zero returns an unchanged float32 clone.
        algorithm: Sparse coefficient-update rule when no decomposition is supplied.

    Returns:
        A :class:`CoordinatePatch` with the anchored edit and numerical diagnostics.

    Raises:
        ValueError: If tensors, indices, edit arguments, atoms, or a supplied decomposition are
            invalid or incompatible.
        RuntimeError: If a fresh default NNLS decomposition fails its KKT certification, or an
            edit postcondition is violated.

    Warns:
        UserWarning: If the source/target pair is near-parallel, or the normalized edit basis is
            rank deficient or poorly conditioned. Both are non-fatal.
    """
    target, dictionary = _validate_inputs(x, dictionary)
    source_idx = _validate_index("source_idx", source_idx, dictionary.shape[0])
    target_idx = _validate_index("target_idx", target_idx, dictionary.shape[0])
    if source_idx == target_idx:
        raise ValueError("source_idx and target_idx must be distinct")
    if mode not in ("substitute", "swap"):
        raise ValueError(f"mode must be 'substitute' or 'swap', got {mode!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be a finite real number")

    if decomposition is None:
        decomposition = get_sparse_decomposition(target, dictionary, k, algorithm=algorithm)
    elif not isinstance(decomposition, JSpaceDecomposition):
        raise ValueError("decomposition must be a JSpaceDecomposition")
    else:
        _validate_decomposition(decomposition, target, dictionary, target_idx)

    support_before = decomposition.support.detach().cpu().clone()
    support_ids = support_before.tolist()
    if source_idx not in support_ids:
        raise ValueError(f"source_idx={source_idx} is not in the decomposition's active support")
    target_was_active = target_idx in support_ids
    if target_was_active:
        support_after = support_before.clone()
        coordinates_before = decomposition.coordinates.float().to(dictionary.device).clone()
    else:
        support_after = torch.cat([support_before, torch.tensor([target_idx], dtype=torch.long)])
        coordinates_before = torch.cat(
            [decomposition.coordinates.float().to(dictionary.device).clone(), target.new_zeros(1)]
        )

    frame = support_after.tolist()
    source_slot = frame.index(source_idx)
    target_slot = frame.index(target_idx)
    edit_coordinates = coordinates_before.clone()
    source_coordinate = coordinates_before[source_slot].clone()
    target_coordinate = coordinates_before[target_slot].clone()
    if mode == "swap":
        edit_coordinates[source_slot] = target_coordinate
        overwritten_target_coordinate: Optional[float] = None
    else:
        edit_coordinates[source_slot] = 0.0
        overwritten_target_coordinate = (
            float(target_coordinate.item()) if target_was_active else None
        )
    edit_coordinates[target_slot] = source_coordinate
    coordinates_after = coordinates_before + float(alpha) * (edit_coordinates - coordinates_before)

    pair = _validate_atom_rows(
        dictionary,
        torch.tensor([source_idx, target_idx], dtype=torch.long),
        context="coordinate patch",
    )
    pair_units = pair / torch.linalg.vector_norm(pair, dim=1, keepdim=True)
    cosine = _warn_if_near_parallel(pair_units)
    basis = dictionary[support_after].float().T
    rank, condition = _basis_diagnostics(basis)

    reconstruction_before = decomposition.reconstruction.float().to(dictionary.device).clone()
    residual = target - reconstruction_before
    if float(alpha) == 0.0:
        reconstruction_after = reconstruction_before.clone()
        delta = torch.zeros_like(target)
        patched = target.clone()
    else:
        reconstruction_after = basis @ coordinates_after
        delta = reconstruction_after - reconstruction_before
        patched = target + delta

    nonedited_max, residual_max = _postcondition_witnesses(
        coordinates_before,
        coordinates_after,
        source_slot,
        target_slot,
        patched,
        reconstruction_after,
        residual,
    )
    return CoordinatePatch(
        support_before=support_before,
        support_after=support_after,
        coordinates_before=coordinates_before.detach(),
        coordinates_after=coordinates_after.detach(),
        source_slot=source_slot,
        target_slot=target_slot,
        target_was_appended=not target_was_active,
        overwritten_target_coordinate=overwritten_target_coordinate,
        reconstruction_before=reconstruction_before.detach(),
        reconstruction_after=reconstruction_after.detach(),
        residual=residual.detach(),
        delta=delta.detach(),
        patched=patched.detach(),
        nonedited_coordinate_max_delta=nonedited_max,
        residual_max_delta=residual_max,
        source_target_cosine=cosine,
        basis_rank=rank,
        basis_condition_number=condition,
    )


def _postcondition_witnesses(
    coordinates_before: torch.Tensor,
    coordinates_after: torch.Tensor,
    source_slot: int,
    target_slot: int,
    patched: torch.Tensor,
    reconstruction_after: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[float, float]:
    """Measure — and enforce — that the edit only moved the source/target coordinates.

    Diagnostic reductions run in float64 on CPU (the decomposition module's convention). Returns
    ``(nonedited_coordinate_max_delta, residual_max_delta)``; raises if either exceeds a float32
    tolerance, since the anchored edit must leave every other coordinate and the residual intact.
    """
    before = coordinates_before.detach().cpu().double()
    after = coordinates_after.detach().cpu().double()
    keep = torch.ones(before.numel(), dtype=torch.bool)
    keep[source_slot] = False
    keep[target_slot] = False
    nonedited_max = (
        float((after[keep] - before[keep]).abs().max().item()) if bool(keep.any()) else 0.0
    )

    residual64 = residual.detach().cpu().double()
    recovered_residual = (
        patched.detach().cpu().double() - reconstruction_after.detach().cpu().double()
    )
    residual_max = float((recovered_residual - residual64).abs().max().item())

    magnitude = max(1.0, float(residual64.abs().max().item()), float(before.abs().max().item()))
    tolerance = 256 * float(torch.finfo(torch.float32).eps) * magnitude
    if nonedited_max > tolerance:
        raise RuntimeError(
            f"coordinate patch changed a non-edited coordinate (max delta={nonedited_max:.3g})"
        )
    if residual_max > tolerance:
        raise RuntimeError(
            f"coordinate patch did not preserve the anchored residual (max delta={residual_max:.3g})"
        )
    return nonedited_max, residual_max
