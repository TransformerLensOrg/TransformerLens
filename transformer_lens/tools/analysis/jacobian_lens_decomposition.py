"""J-space sparse decomposition for the Jacobian lens.

Decomposes an activation (or a steering / sparse-autoencoder direction) into a sparse
nonnegative combination of J-lens vectors, following Gurnee et al. (2026), "Verbalizable
Representations Form a Global Workspace in Language Models" (Transformer Circuits Thread).

The decomposition is greedy: at each step the atom most correlated with the current
residual (using unit-normalised atoms, so high-norm atoms are not preferred for their scale
alone) is added to the active set, and the active-set coefficients are updated under a
nonnegativity constraint. Two coefficient-update rules are provided via ``algorithm``:

- ``"nonnegative_orthogonal_matching_pursuit"`` (default) -- the active-set coefficients are
    re-solved with a float64 active-set nonnegative least-squares fit at each step, then checked
    against the KKT conditions.
- ``"gradient_pursuit"`` -- the directional update of Blumensath & Davies (2008): a single
  optimal-step-size gradient step on the active coefficients, projected onto the
  nonnegative orthant.

Both use the same greedy selection rule and both are nonnegative; they differ in the
coefficient update and therefore can choose different atoms at later steps. At vocabulary
scale, correlation over all atoms costs ``O(num_atoms * d_model)`` per step, while the exact
re-solve adds float64 linear algebra on the small selected set. The exact re-solve is the
default because it returns optimal coefficients on that set; ``"gradient_pursuit"`` avoids
that solve and is provided for faithfulness to the paper and for larger active sets.

Two outputs are returned because they need not coincide: the J-space *component* is the
orthogonal projection of the target onto the span of the selected atoms (the paper's
appendix definition, and the residual its interventions use), while the *coordinates* are
the nonnegative pursuit coefficients. They differ whenever a coefficient is driven to zero
by the nonnegativity constraint.

This module is model-free: it operates on a raw dictionary tensor, so it can be used and
tested without loading a model.

References:
    - Gurnee et al. (2026), "Verbalizable Representations Form a Global Workspace in
      Language Models," Transformer Circuits Thread -- the J-space decomposition.
    - Pati, Rezaiifar & Krishnaprasad (1993), "Orthogonal Matching Pursuit: Recursive
      Function Approximation with Applications to Wavelet Decomposition," 27th Asilomar
      Conference on Signals, Systems and Computers -- the greedy atom selection shared
      by both algorithms.
    - Blumensath & Davies (2008), "Gradient Pursuits," IEEE Transactions on Signal
      Processing 56(6):2370-2382 -- the ``gradient_pursuit`` coefficient update.
    - Lawson & Hanson (1974), "Solving Least Squares Problems," Prentice-Hall -- the
      active-set nonnegative least-squares re-solve used by the default algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch

#: The paper varies the sparsity level but "typically" chooses "no more than 25".
DEFAULT_K = 25

#: Active problems are normally small (``DEFAULT_K`` is 25), so solve them in float64 and use
#: a conservative square-root-epsilon threshold for rank and KKT decisions. Unlike a threshold
#: proportional to ``d_model``, this is invariant to appending zero rows to the same problem.
_NNLS_RELATIVE_TOLERANCE = math.sqrt(torch.finfo(torch.float64).eps)
_NNLS_RANK_RTOL = _NNLS_RELATIVE_TOLERANCE


@dataclass
class JSpaceDecomposition:
    """Result of a sparse J-space decomposition.

    Attributes:
        support: Indices of the selected dictionary atoms (token ids when the
            dictionary is the vocabulary of J-lens vectors).
        coordinates: Nonnegative pursuit coefficients aligned with ``support`` (the
            "local J-space coordinates").
        reconstruction: The nonnegative combination ``sum(coordinates * atoms)``.
        j_space_component: The orthogonal projection of the target onto the span of the
            selected atoms (the paper's "J-space component"). Differs from
            ``reconstruction`` whenever a coordinate is clamped to zero.
        non_j_space_component: The residual ``target - j_space_component`` (the
            "non-J-space component"), orthogonal to the selected span.
    """

    support: torch.Tensor
    coordinates: torch.Tensor
    reconstruction: torch.Tensor
    j_space_component: torch.Tensor
    non_j_space_component: torch.Tensor


def _nnls_tolerances(
    active_atoms: torch.Tensor, target: torch.Tensor, coefficients: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale- and dtype-aware tolerances for the NNLS solver and its KKT check.

    The dual score ``w_j = A[:, j]^T r`` (``r = target - A c``) scales as an atom norm times
    the residual-computation scale. That scale includes ``|| |A| |c| ||`` so a nearly
    cancelling fit receives enough absolute tolerance without making the relative threshold
    depend on the number of rows. A numerically-zero coefficient has the corresponding inverse
    atom-norm scale. Returns ``(dual_tol, coefficient_tol)``, each aligned with the columns of
    ``active_atoms``.
    """
    tiny = torch.finfo(target.dtype).tiny
    atom_norms = active_atoms.norm(dim=0).clamp_min(tiny)
    scale = (target.norm() + (active_atoms.abs() @ coefficients.abs()).norm()).clamp_min(tiny)
    dual_tol = _NNLS_RELATIVE_TOLERANCE * atom_norms * scale
    coefficient_tol = _NNLS_RELATIVE_TOLERANCE * scale / atom_norms
    return dual_tol, coefficient_tol


def _validate_nnls_kkt(
    active_atoms: torch.Tensor,
    target: torch.Tensor,
    coefficients: torch.Tensor,
    dual_tol: torch.Tensor,
    coefficient_tol: torch.Tensor,
) -> None:
    """Raise unless ``coefficients`` satisfy the NNLS KKT conditions within tolerance.

    This is a postcondition check using the solver's numerical tolerances: primal
    feasibility (``c >= 0``), dual feasibility on zeroed coordinates (``w_j <= 0``),
    stationarity on positive coordinates (``w_j = 0``), and complementarity
    (``c_i w_i = 0``).
    """
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
    """Minimize ``||active_atoms @ coefficients - target||`` over ``coefficients >= 0``.

    Uses the Lawson-Hanson active-set method. A zeroed atom enters the passive (free) set when
    it is most positively correlated with the residual; a passive atom leaves when its trial
    coefficient is negative. Released atoms may later re-enter.

    The active problem is solved in float64 with an explicit pseudoinverse rank threshold.
    Coefficients are checked against the KKT conditions before and after they are cast back to
    the target's dtype and device. Feasibility corrections use the ``3 * num_active`` budget;
    a separate admission limit guards against numerical cycling. An exhausted limit, invalid
    line-search step, stalled correction, or failed KKT check raises :class:`RuntimeError`.
    Returns coefficients aligned with the columns of ``active_atoms``.
    """
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
    work_atoms = active_atoms.to(device=work_device, dtype=torch.float64)
    work_target = target.to(device=work_device, dtype=torch.float64)
    if not bool(torch.isfinite(work_atoms).all()) or not bool(torch.isfinite(work_target).all()):
        raise ValueError("active_atoms and target must contain only finite values")

    coefficients = work_target.new_zeros(num_active)
    passive = work_target.new_zeros(num_active, dtype=torch.bool)  # free coordinates (> 0)
    max_corrections = 3 * num_active
    # A correction can release several variables, each of which may later be re-admitted.
    # This is only a cycle safeguard; the mathematical convergence budget is the correction count.
    max_admissions = num_active * (max_corrections + 1)
    corrections = 0
    admissions = 0
    # Main loop: free the zeroed atom most correlated with the residual until no zeroed atom
    # has a materially positive correlation (KKT dual feasibility), or every atom is free.
    while not bool(passive.all()):
        dual_tol, coefficient_tol = _nnls_tolerances(work_atoms, work_target, coefficients)
        dual = work_atoms.T @ (work_target - work_atoms @ coefficients)
        dual = dual.masked_fill(passive, float("-inf"))
        if bool((dual <= dual_tol).all()):
            break
        admissions += 1
        if admissions > max_admissions:
            raise RuntimeError("NNLS exceeded the active-set admission safeguard")
        passive[int(torch.argmax(dual))] = True
        # Feasibility loop: solve the unconstrained fit on the passive set; while it has a
        # materially negative coordinate, step toward it until a passive coefficient reaches
        # zero, then release the vanished coordinates. Each correction drops >= 1 passive atom
        # (enforced by the progress guard), so the loop is bounded.
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
            # Step toward the unconstrained fit, limited by the first coordinate to reach zero.
            # Only materially-negative coordinates with a strictly positive denominator define
            # the step; a zero-current/zero-candidate coordinate (e.g. a freshly admitted atom
            # heading straight to <= 0) has a zero ratio, forcing a zero-length step -- it makes
            # no move but is released by the cleanup below, so the passive set still shrinks.
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
    result_tiny = torch.finfo(result_dtype).tiny
    result_relative_tolerance = math.sqrt(torch.finfo(result_dtype).eps)
    result_atom_norms = result_atoms.norm(dim=0).clamp_min(result_tiny)
    result_scale = (result_target.norm() + (result_atoms.abs() @ result.abs()).norm()).clamp_min(
        result_tiny
    )
    result_dual_tol = result_relative_tolerance * result_atom_norms * result_scale
    result_coefficient_tol = result_relative_tolerance * result_scale / result_atom_norms
    _validate_nnls_kkt(result_atoms, result_target, result, result_dual_tol, result_coefficient_tol)
    return result


def _gradient_pursuit_step(
    active_atoms: torch.Tensor, target: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """One optimal-step-size projected-gradient update (Blumensath & Davies, 2008).

    ``active_atoms`` is ``[d_model, num_active]`` (the selected atoms as columns) and
    ``coefficients`` the current ``[num_active]`` coefficients (the newly added atom starts at
    zero). Moves the coefficients along the steepest-descent direction ``active_atoms^T
    residual`` with the exact line-search step, then projects onto the nonnegative orthant.
    """
    residual = target - active_atoms @ coefficients
    direction = active_atoms.T @ residual  # [num_active]; gradient up to sign
    projected = active_atoms @ direction  # [d_model]
    denominator = float((projected @ projected).detach())
    if denominator <= 0.0:
        return coefficients.clamp_min(0.0)
    step = float((projected @ residual).detach()) / denominator
    return (coefficients + step * direction).clamp_min(0.0)


def get_sparse_decomposition(
    x: torch.Tensor,
    dictionary: torch.Tensor,
    k: int = DEFAULT_K,
    *,
    algorithm: str = "nonnegative_orthogonal_matching_pursuit",
) -> JSpaceDecomposition:
    """Greedily decompose ``x`` into a ``k``-sparse nonnegative combination of atoms.

    Args:
        x: Target vector, shape ``[d_model]``.
        dictionary: Atom matrix, shape ``[num_atoms, d_model]`` (rows are atoms).
        k: Number of atoms to select.
        algorithm: Coefficient-update rule.
            ``"nonnegative_orthogonal_matching_pursuit"`` (default) re-solves the active-set
            coefficients exactly as a nonnegative least-squares fit;
            ``"gradient_pursuit"`` takes a single projected-gradient step per atom. See the
            module docstring for the trade-off (they use the same selection rule, while the
            exact re-solve is optimal on each selected set).

    Returns:
        A :class:`JSpaceDecomposition`.

    Raises:
        ValueError: On an unknown ``algorithm``, complex or non-finite inputs, a non-2-D
            dictionary, a target whose length does not match ``d_model``, ``k`` outside
            ``[1, num_atoms]``, or a dictionary with non-finite or zero-norm atoms.
        RuntimeError: If ``algorithm="nonnegative_orthogonal_matching_pursuit"`` and the
            nonnegative least-squares solve cannot be certified against its KKT conditions
            within its numerical tolerance.
    """
    if algorithm not in ("nonnegative_orthogonal_matching_pursuit", "gradient_pursuit"):
        raise ValueError(
            "algorithm must be 'nonnegative_orthogonal_matching_pursuit' or "
            f"'gradient_pursuit', got {algorithm!r}"
        )
    if dictionary.ndim != 2:
        raise ValueError(
            f"dictionary must be 2-D [num_atoms, d_model], got shape {tuple(dictionary.shape)}"
        )
    num_atoms, d_model = dictionary.shape
    if x.ndim != 1 or x.shape[0] != d_model:
        raise ValueError(f"x must be 1-D of length d_model={d_model}, got shape {tuple(x.shape)}")
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

    residual = target.clone()
    support: List[int] = []
    coordinates = target.new_zeros(0)
    for _ in range(k):
        # Select the atom most correlated with the current residual, using unit-norm atoms
        # so high-norm atoms are not preferred for their scale alone.
        correlation = (atoms @ residual) / atom_norms
        for chosen in support:
            correlation[chosen] = float("-inf")
        support.append(int(torch.argmax(correlation).item()))

        active_atoms = atoms[support].T  # [d_model, len(support)]
        if algorithm == "nonnegative_orthogonal_matching_pursuit":
            # Re-solve the coefficients jointly over the active set as a nonnegative
            # least-squares fit. Sequential per-atom updates are wrong once selected atoms
            # are correlated.
            coordinates = _nonnegative_least_squares(active_atoms, target)
        else:
            # Carry the coefficients forward, initialising the new atom at zero, and take a
            # single projected-gradient step over the active set.
            coordinates = _gradient_pursuit_step(
                active_atoms, target, torch.cat([coordinates, coordinates.new_zeros(1)])
            )
        residual = target - active_atoms @ coordinates

    support_tensor = torch.tensor(support, dtype=torch.long)
    active_atoms = atoms[support_tensor].T  # [d_model, len(support)]
    reconstruction = active_atoms @ coordinates
    # The J-space component is the orthogonal projection of the target onto the span of the
    # selected atoms (paper appendix), computed with the same pseudoinverse construction as
    # swap_hooks. It differs from the nonnegative reconstruction whenever a coordinate was
    # clamped to zero.
    j_space_component = active_atoms @ (torch.linalg.pinv(active_atoms) @ target)
    non_j_space_component = target - j_space_component
    return JSpaceDecomposition(
        support=support_tensor,
        coordinates=coordinates,
        reconstruction=reconstruction,
        j_space_component=j_space_component,
        non_j_space_component=non_j_space_component,
    )
