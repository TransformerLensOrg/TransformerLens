"""J-space sparse decomposition for the Jacobian lens.

Decomposes an activation (or a steering / sparse-autoencoder direction) into a sparse
nonnegative combination of J-lens vectors, following Gurnee et al. (2026), "Verbalizable
Representations Form a Global Workspace in Language Models" (Transformer Circuits Thread).

The decomposition is greedy: at each step the atom most correlated with the current
residual (using unit-normalised atoms, so high-norm atoms are not preferred for their scale
alone) is added to the active set, and the active-set coefficients are updated under a
nonnegativity constraint. Two coefficient-update rules are provided via ``algorithm``:

- ``"nonnegative_orthogonal_matching_pursuit"`` (default) -- the active-set coefficients are
  re-solved *exactly* as a nonnegative least-squares fit at each step.
- ``"gradient_pursuit"`` -- the directional update of Blumensath & Davies (2008): a single
  optimal-step-size gradient step on the active coefficients, projected onto the
  nonnegative orthant.

Both share the same greedy selection and both are nonnegative; they differ only in the
coefficient update. That difference is negligible in this setting: for a vocabulary-sized
dictionary of ``num_atoms`` atoms the per-step cost is dominated by the correlation over all
atoms (``O(num_atoms * d_model)``), so the exact re-solve (``O(k^2 * d_model)`` at sparsity
``k``) adds a small fraction while returning the *optimal* coefficients on the selected
support. ``"gradient_pursuit"`` only pays off when the active set is large (its update avoids
the exact solve); it is provided for faithfulness to the paper and for that regime. Hence
the exact re-solve is the default.

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

from dataclasses import dataclass
from typing import List

import torch

#: The paper varies the sparsity level but "typically" chooses "no more than 25".
DEFAULT_K = 25


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


def _nonnegative_least_squares(active_atoms: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Minimize ``||active_atoms @ coefficients - target||`` over ``coefficients >= 0``.

    A small active-set solver: solve the unconstrained least squares, and while any
    coefficient is negative, drop the most-negative atom and re-solve. Exact for the tiny
    active sets used here (at most ``k`` atoms). Returns coefficients aligned with the
    columns of ``active_atoms``.
    """
    num_active = active_atoms.shape[1]
    active_indices = list(range(num_active))
    coefficients = target.new_zeros(num_active)
    while active_indices:
        solution = torch.linalg.lstsq(
            active_atoms[:, active_indices], target.unsqueeze(1)
        ).solution.squeeze(1)
        if bool((solution >= -1e-9).all()):
            coefficients = target.new_zeros(num_active)
            coefficients[active_indices] = solution.clamp_min(0.0)
            return coefficients
        active_indices.pop(int(torch.argmin(solution).item()))
    return coefficients


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
    denominator = float(projected @ projected)
    if denominator <= 0.0:
        return coefficients.clamp_min(0.0)
    step = float(projected @ residual) / denominator
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
            module docstring for the trade-off (they agree on the selection and, for
            well-conditioned supports, on the coefficients; the exact re-solve is optimal on
            the support).

    Returns:
        A :class:`JSpaceDecomposition`.

    Raises:
        ValueError: On an unknown ``algorithm``, a non-2-D dictionary, a target whose length
            does not match ``d_model``, ``k`` outside ``[1, num_atoms]``, or a dictionary
            with non-finite or zero-norm atoms.
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

    target = x.float()
    atoms = dictionary.float()
    if not bool(torch.isfinite(atoms).all()):
        raise ValueError("dictionary contains non-finite entries")
    atom_norms = (atoms * atoms).sum(dim=1).sqrt()
    if bool((atom_norms == 0).any()):
        raise ValueError("dictionary contains a zero-norm atom")

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
