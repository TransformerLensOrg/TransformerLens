"""Unit tests for Jacobian-lens J-space sparse decomposition.

Model-free: they exercise the nonnegative sparse-decomposition solver directly on raw
dictionary tensors, so no model is loaded.
"""

import itertools

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
    _nonnegative_least_squares,
    get_sparse_decomposition,
)

ALGORITHMS = ["nonnegative_orthogonal_matching_pursuit", "gradient_pursuit"]


# --------------------------------------------------------------------------- #
# Invariants that hold for both coefficient-update rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_recovers_planted_combination_on_orthonormal_dictionary(algorithm):
    """On an orthonormal dictionary both algorithms recover the exact planted support and
    its nonnegative coefficients (a single gradient step is exact when atoms are
    orthonormal, so gradient_pursuit and nonnegative OMP agree)."""
    d_model = 8
    dictionary = torch.eye(d_model)
    x = 3.0 * dictionary[2] + 1.5 * dictionary[5]

    result = get_sparse_decomposition(x, dictionary, k=2, algorithm=algorithm)

    assert isinstance(result, JSpaceDecomposition)
    assert set(result.support.tolist()) == {2, 5}
    coefficient = dict(zip(result.support.tolist(), result.coordinates.tolist()))
    assert coefficient[2] == pytest.approx(3.0, abs=1e-5)
    assert coefficient[5] == pytest.approx(1.5, abs=1e-5)
    assert torch.allclose(result.reconstruction, x, atol=1e-5)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_j_space_component_is_orthogonal_projection_onto_selected_span(algorithm):
    """The J-space component is the orthogonal projection of the target onto the span of the
    selected atoms (paper appendix) -- independent of the coefficient-update rule -- so the
    non-J-space component is orthogonal to every selected atom and the two sum to target."""
    atoms = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],  # correlated with atom 0
            [0.0, 0.0, 1.0],
        ]
    )
    x = torch.tensor([3.8, 2.4, 7.0])  # xy part in span{atom0, atom1}; z part out of it

    result = get_sparse_decomposition(x, atoms, k=2, algorithm=algorithm)

    assert set(result.support.tolist()) == {1, 2}
    assert torch.allclose(result.j_space_component + result.non_j_space_component, x, atol=1e-5)
    for i in result.support.tolist():
        assert torch.dot(result.non_j_space_component, atoms[i]).abs().item() < 1e-5
    selected_atoms = atoms[result.support].T
    projection = selected_atoms @ torch.linalg.pinv(selected_atoms) @ x
    assert torch.allclose(result.j_space_component, projection, atol=1e-5)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_coordinates_are_nonnegative(algorithm):
    torch.manual_seed(0)
    dictionary = torch.randn(15, 5)
    x = torch.randn(5)
    result = get_sparse_decomposition(x, dictionary, k=4, algorithm=algorithm)
    assert (result.coordinates >= 0).all()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_is_deterministic(algorithm):
    """Identical inputs give identical outputs (the solver has no stochastic component)."""
    torch.manual_seed(0)
    dictionary = torch.randn(20, 6)
    x = torch.randn(6)
    first = get_sparse_decomposition(x, dictionary, k=4, algorithm=algorithm)
    second = get_sparse_decomposition(x, dictionary, k=4, algorithm=algorithm)
    assert torch.equal(first.support, second.support)
    assert torch.allclose(first.coordinates, second.coordinates)


# --------------------------------------------------------------------------- #
# nonnegative OMP exactness on the active set
# --------------------------------------------------------------------------- #
def test_exact_resolve_coordinates_match_joint_least_squares_on_correlated_dictionary():
    """With correlated selected atoms the nonnegative OMP coefficients are the joint least-squares
    solution over the active set, not sequential residual subtraction. x lies exactly in
    span{atom0, atom1}, so the correct coefficients are the planted (2, 3)."""
    dictionary = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],  # cos = 0.6 with atom 0
            [0.0, 0.0, 1.0, 0.0],  # orthogonal to x -> never selected
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    x = 2.0 * dictionary[0] + 3.0 * dictionary[1]

    result = get_sparse_decomposition(
        x, dictionary, k=2, algorithm="nonnegative_orthogonal_matching_pursuit"
    )

    assert set(result.support.tolist()) == {0, 1}
    coefficient = dict(zip(result.support.tolist(), result.coordinates.tolist()))
    assert coefficient[0] == pytest.approx(2.0, abs=1e-4)
    assert coefficient[1] == pytest.approx(3.0, abs=1e-4)
    assert torch.allclose(result.reconstruction, x, atol=1e-4)


def test_exact_resolve_returns_true_nonnegative_least_squares_when_unconstrained_is_negative():
    """When the unconstrained fit assigns a negative coefficient, nonnegative OMP returns the true
    nonnegative solution (drop that atom and re-solve), not a naive clamp of the negative
    coefficient to zero. atoms span R^2 and the unconstrained fit of x is (3, -1); the
    nonnegative optimum is (2, 0). Because the two atoms still span R^2, the orthogonal
    projection recovers x and differs from the nonnegative reconstruction -- the
    'return both' point."""
    atoms = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    x = torch.tensor([2.0, -1.0])

    result = get_sparse_decomposition(
        x, atoms, k=2, algorithm="nonnegative_orthogonal_matching_pursuit"
    )

    assert (result.coordinates >= 0).all()
    coefficient = dict(zip(result.support.tolist(), result.coordinates.tolist()))
    assert coefficient.get(0, 0.0) == pytest.approx(2.0, abs=1e-4)  # not the clamped 3.0
    assert coefficient.get(1, 0.0) == pytest.approx(0.0, abs=1e-4)
    assert torch.allclose(result.reconstruction, torch.tensor([2.0, 0.0]), atol=1e-4)
    assert torch.allclose(result.j_space_component, x, atol=1e-5)
    assert not torch.allclose(result.reconstruction, result.j_space_component, atol=1e-3)


# --------------------------------------------------------------------------- #
# gradient_pursuit behaviour
# --------------------------------------------------------------------------- #
def test_gradient_pursuit_agrees_with_exact_resolve_on_orthonormal_dictionary():
    """A single optimal-step gradient update is exact for orthonormal atoms, so
    gradient_pursuit reproduces the nonnegative OMP solution there."""
    orthonormal = torch.linalg.qr(torch.randn(6, 6)).Q  # square -> orthonormal atoms
    x = 2.0 * orthonormal[0] + 1.5 * orthonormal[3]

    gradient_result = get_sparse_decomposition(x, orthonormal, k=2, algorithm="gradient_pursuit")
    exact_result = get_sparse_decomposition(
        x, orthonormal, k=2, algorithm="nonnegative_orthogonal_matching_pursuit"
    )

    assert set(gradient_result.support.tolist()) == set(exact_result.support.tolist()) == {0, 3}
    assert torch.allclose(gradient_result.reconstruction, exact_result.reconstruction, atol=1e-5)
    assert torch.allclose(gradient_result.reconstruction, x, atol=1e-5)


def test_gradient_pursuit_does_not_increase_residual_on_correlated_dictionary():
    """gradient_pursuit is approximate on correlated atoms, but its fit is still a valid
    descent: the reconstruction is no worse than using no atoms, and coefficients stay
    nonnegative."""
    torch.manual_seed(2)
    dictionary = torch.randn(12, 5)
    x = torch.randn(5)

    result = get_sparse_decomposition(x, dictionary, k=4, algorithm="gradient_pursuit")

    assert (result.coordinates >= 0).all()
    assert (x - result.reconstruction).norm().item() <= x.norm().item() + 1e-6


# --------------------------------------------------------------------------- #
# Brute-force oracle (no external reference implementation exists)
# --------------------------------------------------------------------------- #
def _brute_force_best_support(x, dictionary, k):
    """Exhaustive best-``k``-subset nonnegative fit. O(C(num_atoms, k)); tiny sizes only."""
    best_support = None
    best_residual = None
    for combination in itertools.combinations(range(dictionary.shape[0]), k):
        active_atoms = dictionary[list(combination)].T.float()
        coefficients = _nonnegative_least_squares(active_atoms, x.float())
        residual = (x.float() - active_atoms @ coefficients).norm().item()
        if best_residual is None or residual < best_residual - 1e-9:
            best_residual = residual
            best_support = combination
    return best_support, best_residual


def test_exact_resolve_never_beats_the_brute_force_optimum():
    """Greedy nonnegative OMP cannot achieve a smaller residual than the exhaustive best-k-subset
    nonnegative fit (greedy is at best optimal, never better)."""
    torch.manual_seed(3)
    dictionary = torch.randn(9, 4)
    x = torch.randn(4)
    k = 2

    result = get_sparse_decomposition(
        x, dictionary, k=k, algorithm="nonnegative_orthogonal_matching_pursuit"
    )
    greedy_residual = (x - result.reconstruction).norm().item()
    _, optimal_residual = _brute_force_best_support(x, dictionary, k)

    assert greedy_residual >= optimal_residual - 1e-5


def test_exact_resolve_matches_brute_force_optimum_on_low_coherence_dictionary():
    """On a low-coherence (orthonormal) dictionary greedy selection is optimal, so nonnegative OMP
    finds the same support as the exhaustive search -- here the planted one."""
    orthonormal = torch.linalg.qr(torch.randn(8, 8)).Q  # 8 orthonormal atoms in R^8
    x = 2.0 * orthonormal[1] + 3.0 * orthonormal[4] + 1.0 * orthonormal[6]
    k = 3

    result = get_sparse_decomposition(
        x, orthonormal, k=k, algorithm="nonnegative_orthogonal_matching_pursuit"
    )
    best_support, _ = _brute_force_best_support(x, orthonormal, k)

    assert set(result.support.tolist()) == set(best_support) == {1, 4, 6}


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(4), torch.eye(4), k=1, algorithm="omp")


def test_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(4), torch.eye(4), k=0)


def test_rejects_k_larger_than_dictionary():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(3), torch.eye(3), k=5)


def test_rejects_target_dictionary_dimension_mismatch():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(5), torch.eye(4), k=1)


def test_rejects_zero_norm_atom():
    dictionary = torch.eye(4)
    dictionary[1] = 0.0
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(4), dictionary, k=2)


def test_rejects_non_finite_atom():
    dictionary = torch.eye(4)
    dictionary[2, 0] = float("nan")
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(4), dictionary, k=2)


def test_rejects_non_2d_dictionary():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(4), torch.ones(4), k=1)


def test_rejects_non_1d_target():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(2, 4), torch.eye(4), k=1)
