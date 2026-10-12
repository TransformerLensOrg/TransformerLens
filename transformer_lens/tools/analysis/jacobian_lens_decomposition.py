"""Greedy nonnegative J-space decomposition (Gurnee et al., 2026) by NNLS-backed OMP or gradient
pursuit (Blumensath & Davies, 2008), plus the occupancy estimator."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

#: The paper "typically" uses "no more than 25" atoms.
DEFAULT_K = 25

#: sqrt(float64 eps) for NNLS rank and KKT decisions; unlike a threshold proportional to
#: ``d_model``, this is invariant to appending zero rows to the same problem.
_NNLS_RELATIVE_TOLERANCE = math.sqrt(torch.finfo(torch.float64).eps)
_NNLS_RANK_RTOL = _NNLS_RELATIVE_TOLERANCE

#: Activity threshold on ``c_i * ||v_i|| / ||x||`` (scale-invariant across atom norms). Equal to
#: the NNLS zeroing scale so the active support is exactly the strictly-positive NNLS coordinates.
_ACTIVE_RELATIVE_TOLERANCE = _NNLS_RELATIVE_TOLERANCE

_GRADIENT_BACKTRACK_STEPS = 20

#: Early stop on the best normalized correlation over ``||x||``, set at the float32 noise floor:
#: a full-rank target stops instead of selecting noise atoms, keeping the support scale-invariant.
_CORRELATION_RELATIVE_TOLERANCE = math.sqrt(torch.finfo(torch.float32).eps)

#: Near-parallel warning threshold shared by ``swap_hooks`` and the anchored coordinate patch.
_SWAP_WARN_COSINE = 0.99
#: ``swap_hooks`` raises above this: inverting a near-collinear two-atom basis is hopeless.
_SWAP_ERROR_COSINE = 0.999


def _linalg_on_cpu_if_mps(op: Callable[..., torch.Tensor], *tensors: torch.Tensor) -> torch.Tensor:
    """Run ``op`` on CPU copies when inputs are on MPS (no ``pinv``/``svdvals`` kernel there);
    all inputs must share one device."""
    device = tensors[0].device
    if device.type == "mps":
        return op(*(tensor.cpu() for tensor in tensors)).to(device)
    return op(*tensors)


def _project_onto_column_span(atoms: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """``atoms @ pinv(atoms) @ vector`` with the default pinv rtol, routed to CPU on MPS."""
    return _linalg_on_cpu_if_mps(lambda a, v: a @ (torch.linalg.pinv(a) @ v), atoms, vector)


def _validate_vector_and_dictionary(x: torch.Tensor, dictionary: torch.Tensor) -> Tuple[int, int]:
    """Shared ``dictionary``/``x`` shape checks; returns ``(num_atoms, d_model)``."""
    if dictionary.ndim != 2:
        raise ValueError(
            f"dictionary must be 2-D [num_atoms, d_model], got shape {tuple(dictionary.shape)}"
        )
    num_atoms, d_model = dictionary.shape
    if x.ndim != 1 or x.shape[0] != d_model:
        raise ValueError(f"x must be 1-D of length d_model={d_model}, got shape {tuple(x.shape)}")
    return num_atoms, d_model


def _diagnose_intervention_pair(
    unit_vectors: torch.Tensor, *, description: str, stacklevel: int
) -> float:
    """Signed cosine of two unit vectors, under the shared near-parallel warn/raise policy."""
    cosine = float((unit_vectors[0] @ unit_vectors[1]).item())
    abs_cosine = abs(cosine)
    if not math.isfinite(abs_cosine) or abs_cosine >= _SWAP_ERROR_COSINE:
        raise ValueError(
            f"{description} are numerically near-parallel "
            f"(abs cosine={abs_cosine:.6f}); choose better-separated concepts"
        )
    if abs_cosine >= _SWAP_WARN_COSINE:
        warnings.warn(
            f"{description} are poorly conditioned "
            f"(abs cosine={abs_cosine:.6f}); the intervention may be amplified",
            UserWarning,
            stacklevel=stacklevel,
        )
    return cosine


@dataclass
class JSpaceDecomposition:
    """Result of :func:`get_sparse_decomposition`."""

    support: torch.Tensor  # active subset of selected_support (coordinate materially > 0)
    coordinates: torch.Tensor  # nonnegative, aligned with support
    selected_support: torch.Tensor  # every greedily selected atom, active or not
    reconstruction: torch.Tensor
    j_space_component: torch.Tensor  # projection onto span(selected_support), not the active span
    non_j_space_component: torch.Tensor


def _nnls_tolerances(
    active_atoms: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    relative_tolerance: float = _NNLS_RELATIVE_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column ``(dual_tol, coefficient_tol)``; the scale includes ``|| |A| |c| ||`` so a
    nearly cancelling fit gets enough absolute slack without depending on the row count."""
    tiny = torch.finfo(target.dtype).tiny
    atom_norms = active_atoms.norm(dim=0).clamp_min(tiny)
    scale = (target.norm() + (active_atoms.abs() @ coefficients.abs()).norm()).clamp_min(tiny)
    dual_tol = relative_tolerance * atom_norms * scale
    coefficient_tol = relative_tolerance * scale / atom_norms
    return dual_tol, coefficient_tol


def _validate_nnls_kkt(
    active_atoms: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    dual_tol: torch.Tensor,
    coefficient_tol: torch.Tensor,
) -> None:
    """Raise :class:`RuntimeError` unless ``coefficients`` satisfy the NNLS KKT conditions."""
    dual = active_atoms.T @ (target - active_atoms @ coefficients)
    if not bool(torch.isfinite(coefficients).all()) or not bool(torch.isfinite(dual).all()):
        raise RuntimeError("NNLS result contains non-finite values")
    positive = coefficients > coefficient_tol
    if bool((coefficients < -coefficient_tol).any()):
        raise RuntimeError("NNLS result violates primal feasibility (c >= 0)")
    if bool((dual[~positive] > dual_tol[~positive]).any()):
        raise RuntimeError("NNLS result violates dual feasibility on zeroed coordinates")
    if bool((dual[positive].abs() > dual_tol[positive]).any()):
        raise RuntimeError("NNLS result violates stationarity on positive coordinates")
    complementarity = coefficients.abs() * dual.abs()
    complementarity_tol = (coefficients.abs() + coefficient_tol) * dual_tol
    if bool((complementarity > complementarity_tol).any()):
        raise RuntimeError("NNLS result violates complementarity (c_i * w_i = 0)")


def _nonnegative_least_squares(active_atoms: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lawson-Hanson ``argmin_{c >= 0} ||active_atoms @ c - target||``, solved in float64 and
    KKT-certified in both float64 and the target's dtype; raises RuntimeError if it cannot be."""
    if active_atoms.ndim != 2 or target.ndim != 1:
        raise ValueError("active_atoms must be 2-D and target must be 1-D")
    if active_atoms.shape[0] != target.shape[0]:
        raise ValueError("active_atoms and target have incompatible dimensions")
    num_active = active_atoms.shape[1]
    if num_active == 0:
        return target.new_zeros(0)

    result_dtype = target.dtype
    result_device = target.device
    work_device = torch.device("cpu") if target.device.type == "mps" else target.device
    work_atoms = active_atoms.to(device=work_device).to(dtype=torch.float64)
    work_target = target.to(device=work_device).to(dtype=torch.float64)
    if not bool(torch.isfinite(work_atoms).all()) or not bool(torch.isfinite(work_target).all()):
        raise ValueError("active_atoms and target must contain only finite values")

    coefficients = work_target.new_zeros(num_active)
    passive = work_target.new_zeros(num_active, dtype=torch.bool)  # free coordinates (> 0)
    max_corrections = 3 * num_active
    # Cycle safeguard only: a correction can release several atoms, each of which may re-enter.
    max_admissions = num_active * (max_corrections + 1)
    corrections = 0
    admissions = 0
    while not bool(passive.all()):
        dual_tol, coefficient_tol = _nnls_tolerances(work_atoms, work_target, coefficients)
        dual = work_atoms.T @ (work_target - work_atoms @ coefficients)
        dual = dual.masked_fill(passive, float("-inf"))
        # KKT dual feasibility: no zeroed atom can still reduce the residual.
        if bool((dual <= dual_tol).all()):
            break
        admissions += 1
        if admissions > max_admissions:
            raise RuntimeError("NNLS exceeded the active-set admission safeguard")
        passive[int(torch.argmax(dual))] = True
        # Each correction drops >= 1 passive atom (progress guard below), so this terminates.
        while True:
            columns = passive.nonzero(as_tuple=True)[0]
            before = int(columns.numel())
            solution = torch.linalg.pinv(work_atoms[:, columns], rtol=_NNLS_RANK_RTOL) @ work_target
            candidate = work_target.new_zeros(num_active)
            candidate[columns] = solution
            negative = passive & (candidate < -coefficient_tol)
            if not bool(negative.any()):
                passive = candidate > coefficient_tol
                coefficients = torch.where(passive, candidate, torch.zeros_like(candidate))
                break
            corrections += 1
            if corrections > max_corrections:
                raise RuntimeError("NNLS exceeded the Lawson-Hanson feasibility budget")
            # A zero-current atom heading to <= 0 forces a zero-length step; it is still released
            # below, so the passive set shrinks.
            denominator = coefficients - candidate
            ratios = torch.where(
                negative & (denominator > 0),
                coefficients / denominator,
                work_target.new_full((num_active,), float("inf")),
            )
            step = float(ratios.min())
            if not math.isfinite(step) or not (
                -_NNLS_RELATIVE_TOLERANCE <= step <= 1.0 + _NNLS_RELATIVE_TOLERANCE
            ):
                raise RuntimeError(f"NNLS line-search step {step} outside [0, 1]")
            step = min(max(step, 0.0), 1.0)
            updated = coefficients + step * (candidate - coefficients)
            passive = updated > coefficient_tol
            coefficients = torch.where(passive, updated, torch.zeros_like(updated))
            if int(passive.sum()) >= before:
                raise RuntimeError("NNLS feasibility correction made no progress")

    dual_tol, coefficient_tol = _nnls_tolerances(work_atoms, work_target, coefficients)
    _validate_nnls_kkt(work_atoms, work_target, coefficients, dual_tol, coefficient_tol)
    result = coefficients.to(device=result_device, dtype=result_dtype)
    result_atoms = active_atoms.to(device=result_device, dtype=result_dtype)
    result_target = target.to(device=result_device, dtype=result_dtype)
    result_dual_tol, result_coefficient_tol = _nnls_tolerances(
        result_atoms,
        result_target,
        result,
        relative_tolerance=math.sqrt(torch.finfo(result_dtype).eps),
    )
    _validate_nnls_kkt(result_atoms, result_target, result, result_dual_tol, result_coefficient_tol)
    return result


def _gradient_pursuit_step(
    active_atoms: torch.Tensor, target: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """One exact-line-search projected-gradient step over column atoms (Blumensath & Davies,
    2008); projection can make it ascend, so the step is halved until it does not."""
    residual = target - active_atoms @ coefficients
    # Callers add the new atom at 0, so this is the incoming point: a fallback that never ascends.
    feasible = coefficients.clamp_min(0.0)
    direction = active_atoms.T @ residual
    projected = active_atoms @ direction
    denominator = float((projected @ projected).detach())
    if denominator <= 0.0:
        return feasible
    step = float((projected @ residual).detach()) / denominator
    current_residual_squared = float((residual @ residual).detach())
    for _ in range(_GRADIENT_BACKTRACK_STEPS + 1):
        candidate = (coefficients + step * direction).clamp_min(0.0)
        candidate_residual = target - active_atoms @ candidate
        candidate_residual_squared = float((candidate_residual @ candidate_residual).detach())
        if candidate_residual_squared <= current_residual_squared:
            return candidate
        step *= 0.5
    return feasible


def get_sparse_decomposition(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    k: int = DEFAULT_K,
    *,
    algorithm: str = "nonnegative_orthogonal_matching_pursuit",
) -> JSpaceDecomposition:
    """Greedily decompose ``x`` over ``dictionary`` rows into at most ``k`` nonnegative atoms,
    stopping early once no unselected atom is positively correlated with the residual."""
    if algorithm not in ("nonnegative_orthogonal_matching_pursuit", "gradient_pursuit"):
        raise ValueError(
            "algorithm must be 'nonnegative_orthogonal_matching_pursuit' or "
            f"'gradient_pursuit', got {algorithm!r}"
        )
    num_atoms, d_model = _validate_vector_and_dictionary(x, dictionary)
    if not 1 <= k <= num_atoms:
        raise ValueError(f"k must be between 1 and num_atoms={num_atoms}, got {k}")
    if torch.is_complex(x) or torch.is_complex(dictionary):
        raise ValueError("x and dictionary must be real-valued")

    target = x.float()
    atoms = dictionary.float()
    if not bool(torch.isfinite(target).all()):
        raise ValueError("x contains non-finite entries")
    if not bool(torch.isfinite(atoms).all()):
        raise ValueError("dictionary contains non-finite entries")
    atom_norms = torch.linalg.vector_norm(atoms, dim=1)
    if not bool(torch.isfinite(atom_norms).all()) or bool((atom_norms == 0).any()):
        raise ValueError("dictionary contains a non-finite or zero-norm atom")

    x_norm = float(torch.linalg.vector_norm(target).detach())
    correlation_tol = _CORRELATION_RELATIVE_TOLERANCE * x_norm

    residual = target.clone()
    selected: List[int] = []
    coordinates = target.new_zeros(0)
    for _ in range(k):
        # Normalize so high-norm atoms are not preferred for their scale alone.
        correlation = (atoms @ residual) / atom_norms
        for chosen in selected:
            correlation[chosen] = float("-inf")
        candidate = int(torch.argmax(correlation).item())
        # Under nonnegativity a non-positive correlation cannot reduce the residual.
        if float(correlation[candidate].detach()) <= correlation_tol:
            break
        selected.append(candidate)

        selected_atoms = atoms[selected].T
        if algorithm == "nonnegative_orthogonal_matching_pursuit":
            # Joint re-solve: sequential per-atom updates are wrong once selected atoms correlate.
            coordinates = _nonnegative_least_squares(selected_atoms, target)
        else:
            # The paper's update; cheaper, but not jointly optimal on correlated atoms.
            coordinates = _gradient_pursuit_step(
                selected_atoms, target, torch.cat([coordinates, coordinates.new_zeros(1)])
            )
        residual = target - selected_atoms @ coordinates

    # Supports stay on CPU for token decoding; vector outputs stay on the computation device.
    selected_support = torch.tensor(selected, dtype=torch.long)
    selected_atoms = atoms[selected_support].T

    contribution = coordinates * torch.linalg.vector_norm(selected_atoms, dim=0)
    active = contribution > _ACTIVE_RELATIVE_TOLERANCE * x_norm
    support = selected_support[active.cpu()]
    active_coordinates = coordinates[active]
    active_atoms = selected_atoms[:, active]

    reconstruction = active_atoms @ active_coordinates
    # Paper appendix: project onto the whole selected span, which can exceed the active support,
    # with the same default-rtol pinv construction as swap_hooks.
    if selected:
        j_space_component = _project_onto_column_span(selected_atoms, target)
    else:
        j_space_component = target.new_zeros(d_model)
    non_j_space_component = target - j_space_component
    return JSpaceDecomposition(
        support=support,
        coordinates=active_coordinates,
        selected_support=selected_support,
        reconstruction=reconstruction,
        j_space_component=j_space_component,
        non_j_space_component=non_j_space_component,
    )


@dataclass
class JSpaceOccupancy:
    """Result of :func:`estimate_occupancy`."""

    occupancy: int
    marginal_captured_variance: torch.Tensor  # per-step gain, [max_atoms]
    control_captured_variance: torch.Tensor  # per-step gain averaged over random controls
    support: torch.Tensor


def _greedy_captured_variance_gains(
    atoms: torch.Tensor, atom_norms: torch.Tensor, target: torch.Tensor, max_atoms: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select exactly ``max_atoms`` atoms with a span-projection residual (unlike the NNLS one in
    :func:`get_sparse_decomposition`); return per-step captured-variance gains and the support."""
    total_variance = float(target @ target)
    support: List[int] = []
    residual = target.clone()
    captured_variance_gains: List[float] = []
    previous_captured_variance = 0.0
    for _ in range(max_atoms):
        correlation = (atoms @ residual) / atom_norms
        for chosen in support:
            correlation[chosen] = float("-inf")
        support.append(int(torch.argmax(correlation).item()))
        active_atoms = atoms[support].T
        projection = active_atoms @ (torch.linalg.pinv(active_atoms) @ target)
        captured_variance = float((projection @ projection) / total_variance)
        captured_variance_gains.append(captured_variance - previous_captured_variance)
        previous_captured_variance = captured_variance
        residual = target - projection
    return torch.tensor(captured_variance_gains), torch.tensor(support, dtype=torch.long)


def estimate_occupancy(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    max_atoms: int = DEFAULT_K,
    num_control_dictionaries: int = 32,
    seed: int = 0,
) -> JSpaceOccupancy:
    """Estimate how many atoms are meaningfully active in ``x``: the step where the real cumulative
    captured variance most exceeds the mean over seeded random unit-norm control dictionaries."""
    num_atoms, d_model = _validate_vector_and_dictionary(x, dictionary)
    if not 1 <= max_atoms <= num_atoms:
        raise ValueError(f"max_atoms must be between 1 and num_atoms={num_atoms}, got {max_atoms}")
    if num_control_dictionaries < 1:
        raise ValueError(
            f"num_control_dictionaries must be at least 1, got {num_control_dictionaries}"
        )
    if torch.is_complex(x) or torch.is_complex(dictionary):
        raise ValueError("x and dictionary must be real-valued")

    target = x.float()
    atoms = dictionary.float()
    if not bool(torch.isfinite(target).all()):
        raise ValueError("x contains non-finite entries")
    target_squared_norm = target @ target
    if not bool(torch.isfinite(target_squared_norm)):
        raise ValueError("x must have finite norm")
    if float(target_squared_norm) <= 0.0:
        raise ValueError("x must have non-zero norm")
    if not bool(torch.isfinite(atoms).all()):
        raise ValueError("dictionary contains non-finite entries")
    atom_norms = (atoms * atoms).sum(dim=1).sqrt()
    if not bool(torch.isfinite(atom_norms).all()) or bool((atom_norms == 0).any()):
        raise ValueError("dictionary contains a non-finite or zero-norm atom")

    real_captured_variance, support = _greedy_captured_variance_gains(
        atoms, atom_norms, target, max_atoms
    )

    generator = torch.Generator(device=atoms.device).manual_seed(seed)
    # Captured variance is a projection (scale-free), so unit-norm control atoms suffice.
    control_atom_norms = torch.ones(num_atoms, device=atoms.device)
    control_variance_runs: List[torch.Tensor] = []
    for _ in range(num_control_dictionaries):
        random_atoms = torch.randn(
            num_atoms, d_model, generator=generator, device=atoms.device, dtype=atoms.dtype
        )
        random_atoms = random_atoms / (random_atoms * random_atoms).sum(dim=1, keepdim=True).sqrt()
        control_run_variance, _ = _greedy_captured_variance_gains(
            random_atoms, control_atom_norms, target, max_atoms
        )
        control_variance_runs.append(control_run_variance)
    control_captured_variance = torch.stack(control_variance_runs).mean(dim=0)

    separation = real_captured_variance.cumsum(dim=0) - control_captured_variance.cumsum(dim=0)
    occupancy = int(torch.argmax(separation).item()) + 1
    return JSpaceOccupancy(
        occupancy=occupancy,
        marginal_captured_variance=real_captured_variance,
        control_captured_variance=control_captured_variance,
        support=support,
    )


@dataclass
class JSpaceVarianceProfile:
    """Per-layer J-space variance profile from :meth:`JacobianLens.fraction_of_variance`."""

    layers: List[int]
    median: Dict[int, float]  # median over positions of ||j_space_component||^2 / ||activation||^2
    pooled: Dict[int, float]  # sum(||j_space_component||^2) / sum(||activation||^2) over corpus
    per_position: Dict[int, torch.Tensor]
