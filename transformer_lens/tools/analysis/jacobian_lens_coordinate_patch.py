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
    _linalg_on_cpu_if_mps,
    _validate_nnls_kkt,
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
        target_was_appended: Whether the target was absent from the active support and appended
            to ``support_after``. ``target_was_selected`` disambiguates why: pursuit may never
            have considered the target at all, or may have selected it and then assigned it a
            zero coordinate.
        target_was_selected: Whether the target atom was selected by pursuit
            (``decomposition.selected_support``), whether or not it ended up numerically active.
            ``target_was_appended and not target_was_selected`` means pursuit never considered the
            target; ``target_was_appended and target_was_selected`` means pursuit selected it but
            assigned it a zero coordinate. Always ``True`` when ``target_was_appended`` is
            ``False``, since the active support is a subset of the selected support.
        overwritten_target_coordinate: The old target coordinate that a ``substitute`` discards
            when it overwrites an already-active target; ``None`` when the target was absent or
            ``mode`` is ``"swap"`` (a swap relocates a coordinate and discards nothing).
            Alpha-independent by design, like ``target_was_appended`` and the support fields: it
            reports the coordinate the requested edit targets for replacement, not the blended
            outcome, so it is non-``None`` even at ``alpha=0`` where nothing is actually discarded.
        reconstruction_before: Sparse reconstruction from ``support_before``.
        reconstruction_after: ``reconstruction_before + delta``, the anchored reconstruction after
            the edit.
        residual: Anchored residual ``x - reconstruction_before``. This is not necessarily the
            decomposition's orthogonal ``non_j_space_component``.
        delta: ``basis @ (coordinates_after - coordinates_before)``; equivalently
            ``reconstruction_after - reconstruction_before``. Exactly zero and proportional to
            ``alpha``, so it carries no recompute floor.
        patched: ``x + delta``; equal to the float32 input when ``alpha`` is zero, where ``delta``
            is then exactly zero.
        nonedited_coordinate_max_delta: Maximum absolute change across every coordinate that is
            neither source nor target. A postcondition witness: it must be numerically zero.
        residual_max_delta: Maximum absolute difference between ``residual`` and
            ``patched - dictionary[support_after].T @ coordinates_after``, the reconstruction
            recomputed independently from the dictionary rows. A postcondition witness: the
            anchored residual must be preserved.
        source_target_cosine: Signed cosine between the source and target atoms.
        basis_rank: Numerical rank of the column-normalized edit basis at float32 precision.
        basis_condition_number: Condition number of that basis, or infinity when rank deficient.
        coordinates_after_nonnegative: Whether every entry of ``coordinates_after`` is ``>= 0``.
            ``alpha`` in ``[0, 1]`` interpolates within the nonnegative pursuit frame and always
            leaves this ``True``; an ``alpha`` outside that range (e.g. ``2.0`` or ``-1.0``)
            extrapolates and can drive a coordinate negative, in which case this is ``False`` and
            a warning is raised.
    """

    support_before: torch.Tensor
    support_after: torch.Tensor
    coordinates_before: torch.Tensor
    coordinates_after: torch.Tensor
    source_slot: int
    target_slot: int
    target_was_appended: bool
    target_was_selected: bool
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
    coordinates_after_nonnegative: bool


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


def _close(actual: torch.Tensor, expected: torch.Tensor, condition_number: float = 1.0) -> bool:
    # The base tolerance certifies a well-conditioned comparison. When ``expected`` is recomputed
    # through an ill-conditioned span (a pinv projection), the float32 recompute error grows with
    # the span's condition number, so scale the tolerance by it: a fixed 32-eps floor otherwise
    # rejects an exactly-valid hand-built decomposition whose activation lies near the span's
    # null direction. ``condition_number`` clamps to one so well-conditioned checks stay strict.
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
    # Measure the selected span's conditioning once and relax the compatibility checks by it: the
    # pinv projection recompute below (and the dictionary reconstruction) inherit float32 error
    # that grows with this condition number, so a fixed tolerance would reject a genuinely valid
    # decomposition on an ill-conditioned span (see ``_close``).
    if selected_cpu.numel():
        selected_atoms = dictionary[selected_cpu].float().T
        span_condition = _condition_number(selected_atoms)
    else:
        selected_atoms = None
        span_condition = 1.0
    expected_reconstruction = support_atoms.T @ coordinates
    if not _close(local_vectors[0], expected_reconstruction, span_condition):
        raise ValueError("precomputed decomposition reconstruction is incompatible with dictionary")
    # Couple the coordinates to ``target`` (the activation), not just to the dictionary: the
    # reconstruction/dictionary check above passes for any self-consistently scaled pair, so a
    # wrongly-scaled decomposition would slip through and silently anchor the edit to a bogus
    # residual. Reuse the solver's NNLS KKT stationarity test -- the active coordinates are all
    # strictly positive (validated above), so stationarity ``A^T (target - A c) == 0`` is the
    # exact condition for ``c`` being the nonnegative fit of ``target`` over the active support.
    # Tolerances mirror the solver's result-dtype certification (sqrt(eps) of the float32
    # coordinates) so genuine float32-rounded decompositions are not falsely rejected.
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

    Both warnings are non-fatal: the anchored edit performs no inverse of ``basis``, so a
    rank-deficient or poorly conditioned basis only means coordinate attribution is non-unique.
    """
    normalized = basis / torch.linalg.vector_norm(basis, dim=0, keepdim=True)
    diagnostic_basis = normalized.detach().cpu().double()
    # Scale the rank tolerance by the column count (the number of edit atoms), not ``max(shape)``.
    # ``basis`` is ``[d_model, num_atoms]`` with ``d_model >> num_atoms``, so ``max(shape)`` is the
    # d_model row count; for ``d_model >= ~2896`` its ``eps * d_model`` tolerance exceeds the
    # smallest singular value of a genuinely full-rank basis, mislabelling it rank deficient
    # (condition number=inf) at model scale. Column count is the dimension that bounds the rank.
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


def _warn_if_alpha_extrapolates(coordinates_after: torch.Tensor, alpha: float) -> bool:
    """Warn (never raise) when ``alpha`` pushes ``coordinates_after`` outside ``c >= 0``.

    ``alpha`` in ``[0, 1]`` interpolates within the nonnegative pursuit frame, but a caller passing
    ``alpha`` outside that range (e.g. ``2.0`` or ``-1.0``) extrapolates past it and can leave a
    coordinate negative -- meaningful only as an explicit extrapolation, not a fitted
    decomposition. Non-fatal, matching ``_warn_if_near_parallel`` and ``_basis_diagnostics``: the
    edit still completes and the caller can inspect the returned flag.
    """
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
        UserWarning: If the source/target pair is near-parallel, the normalized edit basis is
            rank deficient or poorly conditioned, or ``alpha`` extrapolates ``coordinates_after``
            outside the nonnegative pursuit frame. All three are non-fatal.
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
    target_was_selected = target_was_active or (
        target_idx in decomposition.selected_support.detach().cpu().tolist()
    )
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
    # Move the reconstruction by the coordinate change alone. This is exactly zero at alpha=0
    # (no special case) and strictly proportional to alpha, so ``patched`` has no recompute floor
    # and is continuous through the origin. ``reconstruction_after`` stays anchored to the stored
    # ``reconstruction_before`` so the residual ``x - reconstruction_before`` is preserved exactly.
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


def _postcondition_witnesses(
    coordinates_before: torch.Tensor,
    coordinates_after: torch.Tensor,
    source_slot: int,
    target_slot: int,
    patched: torch.Tensor,
    basis: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[float, float]:
    """Measure — and enforce — that the edit only moved the source/target coordinates.

    Diagnostic reductions run in float64 on CPU (the decomposition module's convention). Returns
    ``(nonedited_coordinate_max_delta, residual_max_delta)``; raises if either exceeds a float32
    tolerance, since the anchored edit must leave every other coordinate and the residual intact.

    The residual witness recomputes the reconstruction independently from the dictionary rows
    (``basis @ coordinates_after``) rather than reusing the ``delta`` that built ``patched``, so a
    basis/coordinate misalignment surfaces as ``patched - basis @ coordinates_after`` drifting from
    ``residual`` instead of cancelling as an algebraic identity of the construction.
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
