"""Anchored coordinate edits over sparse Jacobian-lens decompositions."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, MutableMapping, Optional, Sequence, Tuple

import torch

from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    _SWAP_WARN_COSINE,
    DEFAULT_K,
    JSpaceDecomposition,
    _linalg_on_cpu_if_mps,
    _validate_nnls_kkt,
    get_sparse_decomposition,
)

#: Warn above 1/sqrt(float32 eps): a well-scaled basis (condition number ~1) never trips it.
_BASIS_CONDITION_WARNING = 1.0 / math.sqrt(torch.finfo(torch.float32).eps)


@dataclass(frozen=True)
class CoordinatePatch:
    """Detached result of an anchored J-space edit; vectors are float32 on the dictionary device."""

    support_before: torch.Tensor  # long on CPU, in decomposition order
    support_after: torch.Tensor  # support_before + an appended absent target; coordinates' frame
    coordinates_before: torch.Tensor
    coordinates_after: torch.Tensor
    source_slot: int
    target_slot: int
    target_was_appended: bool
    target_was_selected: bool  # in selected_support, even if pursuit then zeroed it
    overwritten_target_coordinate: Optional[float]  # None for swap/absent target; alpha-independent
    reconstruction_before: torch.Tensor
    reconstruction_after: torch.Tensor
    residual: torch.Tensor  # x - reconstruction_before, not non_j_space_component
    delta: torch.Tensor
    patched: torch.Tensor
    nonedited_coordinate_max_delta: float
    residual_max_delta: float
    source_target_cosine: float
    basis_rank: int  # of the column-normalized edit basis
    basis_condition_number: float  # inf when rank deficient
    coordinates_after_nonnegative: bool


def _validate_inputs(x: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
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
    return target


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


def _close(actual: torch.Tensor, expected: torch.Tensor, condition_number: float = 1.0) -> bool:
    """Float32 allclose with tolerance scaled by ``condition_number`` (clamped to >= 1), since a
    pinv recompute through an ill-conditioned span amplifies float32 error by that factor."""
    if actual.shape != expected.shape:
        return False
    magnitude = torch.stack([actual.abs().max(), expected.abs().max()]).max()
    tolerance = 32 * torch.finfo(torch.float32).eps * max(condition_number, 1.0)
    return bool(
        torch.allclose(actual, expected, rtol=tolerance, atol=tolerance * float(magnitude.item()))
    )


def _condition_number(atoms: torch.Tensor) -> float:
    """Condition number of a column matrix, on CPU when the device lacks an ``svdvals`` kernel."""
    singular_values = _linalg_on_cpu_if_mps(torch.linalg.svdvals, atoms)
    smallest = float(singular_values.min().item())
    if smallest <= 0.0:
        return math.inf
    return float(singular_values.max().item()) / smallest


def _validate_decomposition(
    decomposition: JSpaceDecomposition,
    target: torch.Tensor,
    dictionary: torch.Tensor,
    target_idx: int,
) -> None:
    num_atoms, d_model = dictionary.shape
    supports = (decomposition.support, decomposition.selected_support)
    if any(indices.ndim != 1 for indices in supports):
        raise ValueError("precomputed decomposition supports must be 1-D")
    if any(indices.dtype != torch.long for indices in supports):
        raise ValueError("precomputed decomposition supports must have dtype torch.long")
    support_cpu, selected_cpu = (indices.detach().cpu() for indices in supports)
    if any(indices.unique().numel() != indices.numel() for indices in (support_cpu, selected_cpu)):
        raise ValueError("precomputed decomposition supports must not contain duplicates")
    if any(
        bool(((indices < 0) | (indices >= num_atoms)).any())
        for indices in (support_cpu, selected_cpu)
    ):
        raise ValueError("precomputed decomposition support is outside the dictionary")
    if not set(support_cpu.tolist()).issubset(set(selected_cpu.tolist())):
        raise ValueError("precomputed active support must be a subset of selected_support")
    # Only referenced rows are checked, so a reused decomposition skips the full-dictionary scan.
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
    # The recomputes below inherit float32 error growing with the selected span's conditioning.
    if selected_cpu.numel():
        selected_atoms = dictionary[selected_cpu].float().T
        span_condition = _condition_number(selected_atoms)
    else:
        selected_atoms = None
        span_condition = 1.0
    expected_reconstruction = support_atoms.T @ coordinates
    if not _close(local_vectors[0], expected_reconstruction, span_condition):
        raise ValueError("precomputed decomposition reconstruction is incompatible with dictionary")
    # The reconstruction check passes any self-consistently rescaled pair, so certify the NNLS KKT
    # conditions against ``target`` too, with the solver's result-dtype (float32) tolerances.
    active_atoms = support_atoms.T
    tiny = torch.finfo(target.dtype).tiny
    relative_tolerance = math.sqrt(torch.finfo(target.dtype).eps)
    atom_norms = active_atoms.norm(dim=0).clamp_min(tiny)
    scale = (target.norm() + (active_atoms.abs() @ coordinates.abs()).norm()).clamp_min(tiny)
    dual_tol = relative_tolerance * atom_norms * scale
    coefficient_tol = relative_tolerance * scale / atom_norms
    try:
        _validate_nnls_kkt(active_atoms, target, coordinates, dual_tol, coefficient_tol)
    except RuntimeError as error:
        raise ValueError(
            f"precomputed decomposition coordinates do not fit the activation ({error})"
        ) from error
    if not _close(local_vectors[1] + local_vectors[2], target, span_condition):
        raise ValueError("precomputed decomposition is incompatible with activation")
    if selected_atoms is not None:
        expected_component = _linalg_on_cpu_if_mps(
            lambda atoms, vector: atoms @ (torch.linalg.pinv(atoms) @ vector),
            selected_atoms,
            target,
        )
    else:
        expected_component = torch.zeros_like(target)
    if not _close(local_vectors[1], expected_component, span_condition):
        raise ValueError("precomputed decomposition selected span is incompatible with dictionary")


def _basis_diagnostics(basis: torch.Tensor) -> tuple[int, float]:
    """Rank and condition number of the column-normalized edit basis, warning when ill-conditioned.
    Non-fatal: the edit inverts nothing, so a bad basis only makes attribution non-unique."""
    normalized = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True)
    diagnostic_basis = normalized.detach().cpu().double()
    # rtol scales by the column count, not max(shape) = d_model: eps * d_model exceeds the smallest
    # singular value of a full-rank basis once d_model >= ~2896.
    rank_rtol = torch.finfo(torch.float32).eps * diagnostic_basis.shape[1]
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
        condition = _condition_number(diagnostic_basis)
        if condition >= _BASIS_CONDITION_WARNING:
            warnings.warn(
                f"coordinate-patch basis is poorly conditioned (condition number={condition:.6g}); "
                "coordinate attribution may be non-unique",
                UserWarning,
                stacklevel=3,
            )
    return rank, condition


def _warn_if_near_parallel(pair_units: torch.Tensor) -> float:
    """Warn on near-parallel source/target atoms and return their signed cosine.
    Never raises: unlike ``swap_hooks``, the anchored patch inverts no two-atom basis."""
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


def _warn_if_alpha_extrapolates(coordinates_after: torch.Tensor, alpha: float) -> bool:
    """Return whether ``coordinates_after >= 0``, warning (non-fatally) when an ``alpha`` outside
    ``[0, 1]`` extrapolates past the nonnegative pursuit frame."""
    nonnegative = bool((coordinates_after >= 0).all().item())
    if not nonnegative:
        warnings.warn(
            f"coordinate-patch alpha={alpha:g} extrapolates coordinates_after outside the "
            "nonnegative pursuit frame (a coordinate is negative); treat the patch as an "
            "extrapolation, not a fitted decomposition",
            UserWarning,
            stacklevel=3,
        )
    return nonnegative


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
    """Edit sparse J-space coordinates of ``x``, preserving the residual ``x - reconstruction``.
    ``substitute`` moves the source value onto the target and zeros the source; ``swap`` swaps."""
    target = _validate_inputs(x, dictionary)
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
    target_was_selected = target_was_active or (
        target_idx in decomposition.selected_support.detach().cpu().tolist()
    )
    support_after = support_before.clone()
    coordinates_before = decomposition.coordinates.float().to(dictionary.device).clone()
    if not target_was_active:
        support_after = torch.cat([support_after, torch.tensor([target_idx], dtype=torch.long)])
        coordinates_before = torch.cat([coordinates_before, target.new_zeros(1)])

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
    coordinates_after_nonnegative = _warn_if_alpha_extrapolates(coordinates_after, float(alpha))

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
    # Moving by the coordinate change alone keeps delta exactly zero at alpha=0 and proportional to
    # alpha (no recompute floor), and leaves the residual ``x - reconstruction_before`` exact.
    delta = basis @ (coordinates_after - coordinates_before)
    reconstruction_after = reconstruction_before + delta
    patched = target + delta

    nonedited_max, residual_max = _postcondition_witnesses(
        coordinates_before,
        coordinates_after,
        source_slot,
        target_slot,
        patched,
        basis,
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
        target_was_selected=target_was_selected,
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
        coordinates_after_nonnegative=coordinates_after_nonnegative,
    )


def solve_coordinate_patch_positions(
    activations: torch.Tensor,
    dictionary: torch.Tensor,
    position_labels: Sequence[int],
    source_idx: int,
    target_idx: int,
    *,
    layer: int,
    decomposition_cache: Optional[MutableMapping[Tuple[int, int, int], JSpaceDecomposition]] = None,
    k: int = DEFAULT_K,
    mode: str = "substitute",
    alpha: float = 1.0,
    algorithm: str = "nonnegative_orthogonal_matching_pursuit",
) -> Tuple[torch.Tensor, Dict[Tuple[int, int, int], CoordinatePatch]]:
    """Apply :func:`solve_coordinate_patch` to every ``(batch, column)`` of ``activations``.
    Patches and ``decomposition_cache`` key on ``(layer, batch_idx, position_labels[column])``."""
    if activations.ndim != 3:
        raise ValueError(
            "activations must be 3-D [batch, num_positions, d_model], got shape "
            f"{tuple(activations.shape)}"
        )
    batch_size, num_positions, _ = activations.shape
    if len(position_labels) != num_positions:
        raise ValueError(
            f"position_labels has {len(position_labels)} entries, expected {num_positions} "
            "to match activations.shape[1]"
        )
    patched = activations.new_empty(activations.shape)
    patches: Dict[Tuple[int, int, int], CoordinatePatch] = {}
    for batch_idx in range(batch_size):
        for column, position in enumerate(position_labels):
            key = (layer, batch_idx, position)
            x = activations[batch_idx, column]
            if decomposition_cache is not None and key in decomposition_cache:
                decomposition = decomposition_cache[key]
            else:
                decomposition = get_sparse_decomposition(x, dictionary, k, algorithm=algorithm)
                if decomposition_cache is not None:
                    decomposition_cache[key] = decomposition
            patch = solve_coordinate_patch(
                x,
                dictionary,
                source_idx,
                target_idx,
                decomposition=decomposition,
                k=k,
                mode=mode,
                alpha=alpha,
                algorithm=algorithm,
            )
            # Straight-through: ``patch.patched`` is detached, but ``x + patch.delta`` equals it for
            # float32 ``x`` and keeps ``x``'s graph (an identity gradient at the patch).
            patched[batch_idx, column] = x + patch.delta
            patches[key] = patch
    return patched, patches


def _postcondition_witnesses(
    coordinates_before: torch.Tensor,
    coordinates_after: torch.Tensor,
    source_slot: int,
    target_slot: int,
    patched: torch.Tensor,
    basis: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[float, float]:
    """Return and enforce ``(nonedited_coordinate_max_delta, residual_max_delta)`` in float64.
    Recomputes the reconstruction from ``basis``, not ``delta``, so a misalignment cannot cancel."""
    before = coordinates_before.detach().cpu().double()
    after = coordinates_after.detach().cpu().double()
    keep = torch.ones(before.numel(), dtype=torch.bool)
    keep[source_slot] = False
    keep[target_slot] = False
    nonedited_max = (
        float((after[keep] - before[keep]).abs().max().item()) if bool(keep.any()) else 0.0
    )

    residual64 = residual.detach().cpu().double()
    independent_reconstruction = (basis @ coordinates_after).detach().cpu().double()
    recovered_residual = patched.detach().cpu().double() - independent_reconstruction
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
