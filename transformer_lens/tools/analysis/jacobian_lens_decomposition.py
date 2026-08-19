"""J-space sparse decomposition for the Jacobian lens.

Decomposes an activation (or a steering / sparse-autoencoder direction) into a sparse
nonnegative combination of J-lens vectors, following Gurnee et al. (2026), "Verbalizable
Representations Form a Global Workspace in Language Models" (Transformer Circuits Thread).

The decomposition is greedy: at each of at most ``k`` steps the atom most correlated with the
current residual (using unit-normalised atoms, so high-norm atoms are not preferred for their
scale alone) is added to the *selected* set, and the selected-set coefficients are updated
under a nonnegativity constraint. ``k`` is an upper bound, not a target: selection stops early
once no unselected atom has a materially positive residual correlation (under nonnegativity a
negatively-correlated atom cannot reduce the residual), so fewer than ``k`` atoms may be
selected. Two coefficient-update rules are provided via ``algorithm``:

- ``"nonnegative_orthogonal_matching_pursuit"`` (default) -- the active-set coefficients are
    re-solved with a float64 active-set nonnegative least-squares fit at each step, then checked
    against the KKT conditions.
- ``"gradient_pursuit"`` -- the directional update of Blumensath & Davies (2008): a single
  optimal-step-size gradient step on the active coefficients, projected onto the
  nonnegative orthant.

Both use the same greedy selection rule and both are nonnegative; they differ in the
coefficient update and therefore can choose different atoms at later steps. At vocabulary
scale, correlation over all atoms costs ``O(num_atoms * d_model)`` per step, while the exact
re-solve adds float64 linear algebra on the small selected set. The NNLS update is the default
because it optimizes all selected coefficients jointly. ``"gradient_pursuit"`` skips that solve
and matches the update used in the paper.

Two supports are exposed because the paper uses two inconsistent operationalizations. The
``support`` is the numerically *active* set -- the selected atoms whose nonnegative
coordinate materially contributes -- and ``coordinates`` are aligned with it; this is the
paper's main-text sparse nonnegative reconstruction. The ``selected_support`` is every
greedily selected atom, including any whose coordinate was driven to zero by the
nonnegativity constraint; it defines the span for the paper's appendix projection. Hence
``len(support) <= len(selected_support) <= k``.

Two vector outputs correspondingly need not coincide: the ``reconstruction`` is the
nonnegative combination over the active ``support``, while the J-space *component* is the
orthogonal projection of the target onto the span of ``selected_support`` (the paper's
appendix definition, and the residual its interventions use). For the exact NNLS re-solve
the reconstruction equals the projection onto the *active* support (KKT stationarity), so the
two vectors differ exactly when a selected atom has a zero coordinate.

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
from typing import Dict, List, Tuple

import torch

#: The paper varies the sparsity level but "typically" chooses "no more than 25".
DEFAULT_K = 25

#: Active problems are normally small (``DEFAULT_K`` is 25), so solve them in float64 and use
#: a conservative square-root-epsilon threshold for rank and KKT decisions. Unlike a threshold
#: proportional to ``d_model``, this is invariant to appending zero rows to the same problem.
_NNLS_RELATIVE_TOLERANCE = math.sqrt(torch.finfo(torch.float64).eps)
_NNLS_RANK_RTOL = _NNLS_RELATIVE_TOLERANCE

#: A selected atom counts as numerically *active* when its contribution ``c_i * ||v_i||`` is a
#: materially nonzero fraction of the target scale ``||x||``. Using the contribution (not the
#: raw coefficient) keeps the test scale-invariant across J-lens vectors of different native
#: norms. This is a numerical-activity threshold, not an interpretability one. It matches the
#: default NNLS coefficient-zeroing scale (:data:`_NNLS_RELATIVE_TOLERANCE`), so the active
#: support equals the set of strictly-positive NNLS coordinates and the reconstruction over the
#: active support is the projection onto it.
_ACTIVE_RELATIVE_TOLERANCE = _NNLS_RELATIVE_TOLERANCE

_GRADIENT_BACKTRACK_STEPS = 20

#: Early-stopping correlation threshold. Once the best unselected normalized correlation drops
#: to this fraction of ``||x||``, no atom can materially reduce the residual and selection
#: stops. It sits at the float32 residual noise floor (the residual is computed in float32),
#: so a full-rank target stops instead of selecting noise atoms -- this is what makes the
#: selected support scale-invariant. It is deliberately coarser than the activity threshold:
#: selection is a float32 correlation decision, activity a check against the float64 solve.
_CORRELATION_RELATIVE_TOLERANCE = math.sqrt(torch.finfo(torch.float32).eps)


@dataclass
class JSpaceDecomposition:
    """Result of a sparse J-space decomposition.

    Attributes:
        support: Indices of the numerically *active* selected atoms -- those whose
            nonnegative coordinate materially contributes (token ids when the dictionary is
            the vocabulary of J-lens vectors). A subset of ``selected_support``.
        coordinates: Nonnegative pursuit coefficients aligned with ``support`` (the
            "local J-space coordinates"); every entry is materially nonzero.
        selected_support: Indices of every greedily selected atom, including any whose
            coordinate was driven to zero by the nonnegativity constraint. Defines the span
            for ``j_space_component``. Satisfies
            ``support.numel() <= selected_support.numel() <= k``.
        reconstruction: The nonnegative combination ``sum(coordinates * active atoms)`` over
            ``support``.
        j_space_component: The orthogonal projection of the target onto the span of
            ``selected_support`` (the paper's "J-space component"). For the exact NNLS
            re-solve it equals ``reconstruction`` unless a selected atom has a zero
            coordinate, in which case the projection uses a larger span.
        non_j_space_component: The residual ``target - j_space_component`` (the
            "non-J-space component"), orthogonal to the selected span.
    """

    support: torch.Tensor
    coordinates: torch.Tensor
    selected_support: torch.Tensor
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
    zero, so the incoming point is feasible and its residual is the previous residual). Moves
    the coefficients along the steepest-descent direction ``active_atoms^T residual`` with the
    exact line-search step, then projects onto the nonnegative orthant. The exact line search
    happens before that projection, so the projected step can in principle increase the
    residual. In that case the step is halved until the projected update no longer increases
    the residual, with the feasible incoming point retained if the bounded search finds none.
    """
    residual = target - active_atoms @ coefficients
    feasible = coefficients.clamp_min(0.0)
    direction = active_atoms.T @ residual  # [num_active]; gradient up to sign
    projected = active_atoms @ direction  # [d_model]
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
    """Greedily decompose ``x`` into a ``k``-sparse nonnegative combination of atoms.

    Args:
        x: Target vector, shape ``[d_model]``.
        dictionary: Atom matrix, shape ``[num_atoms, d_model]`` (rows are atoms).
        k: Upper bound on the number of atoms to select. Selection stops early once no
            unselected atom is materially positively correlated with the residual, so fewer
            than ``k`` atoms may be selected (and fewer still may be numerically active).
        algorithm: Coefficient-update rule.
            ``"nonnegative_orthogonal_matching_pursuit"`` (default) re-solves the selected-set
            coefficients exactly as a nonnegative least-squares fit;
            ``"gradient_pursuit"`` takes a single projected-gradient step per atom. See the
            module docstring for the trade-off (they use the same selection rule, while the
            exact re-solve is optimal on each selected set).

    Returns:
        A :class:`JSpaceDecomposition`. Its ``support`` holds only the numerically active
        atoms and ``selected_support`` every selected atom, with
        ``support.numel() <= selected_support.numel() <= k``.

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

    x_norm = float(torch.linalg.vector_norm(target).detach())
    correlation_tol = _CORRELATION_RELATIVE_TOLERANCE * x_norm

    residual = target.clone()
    selected: List[int] = []
    coordinates = target.new_zeros(0)
    for _ in range(k):
        # Select the unselected atom most correlated with the current residual, using
        # unit-norm atoms so high-norm atoms are not preferred for their scale alone.
        correlation = (atoms @ residual) / atom_norms
        for chosen in selected:
            correlation[chosen] = float("-inf")
        candidate = int(torch.argmax(correlation).item())
        # Stop early once no unselected atom is materially positively correlated: under the
        # nonnegativity constraint a non-positive correlation cannot reduce the residual, so
        # ``k`` is an upper bound on the number of selected atoms, not a target.
        if float(correlation[candidate].detach()) <= correlation_tol:
            break
        selected.append(candidate)

        selected_atoms = atoms[selected].T  # [d_model, len(selected)]
        if algorithm == "nonnegative_orthogonal_matching_pursuit":
            # Re-solve the coefficients jointly over the selected set as a nonnegative
            # least-squares fit. Sequential per-atom updates are wrong once selected atoms
            # are correlated.
            coordinates = _nonnegative_least_squares(selected_atoms, target)
        else:
            # Carry the coefficients forward, initialising the new atom at zero, and take a
            # single projected-gradient step over the selected set.
            coordinates = _gradient_pursuit_step(
                selected_atoms, target, torch.cat([coordinates, coordinates.new_zeros(1)])
            )
        residual = target - selected_atoms @ coordinates

    # ``selected_support`` stays on CPU for token decoding; the vector-valued outputs stay on
    # the computation device.
    selected_support = torch.tensor(selected, dtype=torch.long)
    selected_atoms = atoms[selected_support].T  # [d_model, num_selected] on atoms.device

    # Public active support: selected atoms whose contribution ``c_i * ||v_i||`` is a
    # materially nonzero fraction of ``||x||``. For the NNLS re-solve this is exactly the
    # strictly-positive coordinate set (its coefficient-zeroing uses the same scale); for
    # gradient pursuit it prunes atoms left at (near-)zero by the final projected step.
    contribution = coordinates * torch.linalg.vector_norm(selected_atoms, dim=0)
    active = contribution > _ACTIVE_RELATIVE_TOLERANCE * x_norm  # on the computation device
    support = selected_support[active.cpu()]  # CPU, aligned with the token-decoding tensors
    active_coordinates = coordinates[active]
    active_atoms = selected_atoms[:, active]  # [d_model, num_active] on atoms.device

    # Reconstruction is the nonnegative combination over the active support (empty -> zeros).
    reconstruction = active_atoms @ active_coordinates
    # The J-space component is the orthogonal projection of the target onto the span of every
    # selected atom (paper appendix), computed with the same pseudoinverse construction as
    # swap_hooks. That span can be larger than the active support when a selected coordinate is
    # zero, so the projection differs from the nonnegative reconstruction in that case.
    if selected:
        j_space_component = selected_atoms @ (torch.linalg.pinv(selected_atoms) @ target)
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
    """Result of a J-space occupancy estimate.

    Attributes:
        occupancy: Estimated number of meaningfully-active atoms -- the step of maximum
            separation between the real and random-control cumulative captured variance.
        marginal_captured_variance: Per-step captured-variance gain of the real greedy selection,
            shape ``[max_atoms]``.
        control_captured_variance: Per-step captured-variance gain averaged over the random
            control dictionaries, shape ``[max_atoms]``.
        support: Greedily selected atom indices, shape ``[max_atoms]`` (token ids when the
            dictionary is the vocabulary of J-lens vectors).
    """

    occupancy: int
    marginal_captured_variance: torch.Tensor
    control_captured_variance: torch.Tensor
    support: torch.Tensor


def _greedy_captured_variance_gains(
    atoms: torch.Tensor, atom_norms: torch.Tensor, target: torch.Tensor, max_atoms: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Greedily select exactly ``max_atoms`` atoms and return captured-variance gains.

    At each step, add the unused atom with the greatest signed, norm-normalized correlation with
    the current residual. Project ``target`` orthogonally onto the full selected span using a
    pseudoinverse, then set the next residual to ``target - projection``. The captured variance is
    ``||Pi_S target||^2 / ||target||^2``; the returned values are its per-step increments.

    This shares the per-step correlation rule with :func:`get_sparse_decomposition`, but not its
    residual recurrence: sparse decomposition uses a nonnegative coefficient-fit residual and may
    stop early, while this recurrence does not stop early, so the selected supports can differ.
    """
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
    """Estimate how many dictionary atoms are meaningfully active in ``x``.

    Runs the projection-residual recurrence described in
    :func:`_greedy_captured_variance_gains` for exactly ``max_atoms`` steps and compares the real
    per-step captured-variance curve against the same recurrence on ``num_control_dictionaries``
    random unit-norm dictionaries of the same size. This shares sparse decomposition's per-step
    correlation rule, but uses an unconstrained span-projection residual rather than a nonnegative
    coefficient-fit residual, so their supports need not match. The occupancy is the step of
    maximum separation between the real and (averaged) control *cumulative* captured variance --
    the point past which further atoms add no more than random directions would. Deterministic
    given ``seed`` and needs no threshold. (Captured variance is a projection, hence scale-free,
    so the random control atoms are simply unit-norm.)

    Args:
        x: Target vector, shape ``[d_model]``.
        dictionary: Atom matrix, shape ``[num_atoms, d_model]`` (rows are atoms).
        max_atoms: Number of atoms to select in the real and control recurrences.
        num_control_dictionaries: Number of random control dictionaries to average over.
        seed: Seed for the random control dictionaries (reproducibility).

    Returns:
        An :class:`JSpaceOccupancy`.

    Raises:
        ValueError: On complex inputs, a non-2-D dictionary, a target whose length does not match
            ``d_model``, ``max_atoms`` outside ``[1, num_atoms]``,
            ``num_control_dictionaries < 1``, a target with non-finite entries or a non-finite or
            zero norm, or a dictionary with non-finite or zero-norm atoms.
    """
    if dictionary.ndim != 2:
        raise ValueError(
            f"dictionary must be 2-D [num_atoms, d_model], got shape {tuple(dictionary.shape)}"
        )
    num_atoms, d_model = dictionary.shape
    if x.ndim != 1 or x.shape[0] != d_model:
        raise ValueError(f"x must be 1-D of length d_model={d_model}, got shape {tuple(x.shape)}")
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
    """Per-layer J-space variance profile over a prompt corpus.

    Produced by :meth:`JacobianLens.fraction_of_variance`.

    Attributes:
        layers: The source layers profiled, in order.
        median: Per-layer median over positions of the J-space variance fraction
            ``||j_space_component||^2 / ||activation||^2``.
        pooled: Per-layer pooled ratio ``sum(||j_space_component||^2) / sum(||activation||^2)``
            across the corpus (the paper's "fraction of total variance").
        per_position: Per-layer 1-D tensor of the raw per-position variance fractions.
    """

    layers: List[int]
    median: Dict[int, float]
    pooled: Dict[int, float]
    per_position: Dict[int, torch.Tensor]
