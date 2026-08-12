"""Unit tests for Jacobian-lens J-space sparse decomposition.

Model-free: they exercise the nonnegative sparse-decomposition solver directly on raw
dictionary tensors, so no model is loaded.
"""

import itertools

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
    _nnls_tolerances,
    _nonnegative_least_squares,
    _validate_nnls_kkt,
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


def _reference_nnls(active_atoms: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Independent brute-force exact nonnegative least squares, by support enumeration.

    At the optimum the strictly-positive coordinates equal the *unconstrained* least-squares
    fit on their support (KKT stationarity) and are feasible, so the optimum is the
    minimum-residual feasible unconstrained fit over all support subsets. Deliberately shares
    no code with ``_nonnegative_least_squares`` so it can independently check it (the suite's
    other oracle, ``_brute_force_best_support``, *reuses* that function and so cannot).
    """
    active_atoms = active_atoms.double()
    target = target.double()
    num_atoms = active_atoms.shape[1]
    best = target.new_zeros(num_atoms)
    best_residual = float(target.norm())
    for size in range(1, num_atoms + 1):
        for support in itertools.combinations(range(num_atoms), size):
            columns = list(support)
            solution = torch.linalg.lstsq(
                active_atoms[:, columns], target.unsqueeze(1)
            ).solution.squeeze(1)
            tolerance = 1e-10 * max(1.0, float(solution.abs().max()))
            if bool((solution >= -tolerance).all()):
                candidate = target.new_zeros(num_atoms)
                candidate[columns] = solution.clamp_min(0.0)
                residual = float((active_atoms @ candidate - target).norm())
                if residual < best_residual - 1e-12:
                    best_residual, best = residual, candidate
    return best


def _assert_nnls_kkt(
    active_atoms: torch.Tensor, target: torch.Tensor, coefficients: torch.Tensor
) -> None:
    """Independent KKT certificate for a nonnegative-least-squares solution.

    Reimplements the optimality conditions -- primal feasibility, dual feasibility on zeroed
    coordinates, stationarity on positive coordinates (which together give complementarity) --
    without calling the solver's own validator. The tolerance is scale-aware but deliberately
    looser than the production threshold, so this certifies correctness rather than restating
    the solver's internal check. Robust for rank-deficient and ill-conditioned systems because
    the least-squares residual is orthogonal to the passive column span regardless of
    conditioning.
    """
    input_epsilon = torch.finfo(active_atoms.dtype).eps
    active_atoms = active_atoms.double()
    target = target.double()
    coefficients = coefficients.double()
    residual = target - active_atoms @ coefficients
    atom_norms = active_atoms.norm(dim=0).clamp_min(torch.finfo(torch.float64).tiny)
    dual = active_atoms.T @ residual
    scale = target.norm() + (active_atoms.abs() @ coefficients.abs()).norm()
    # Combine a strict residual-cosine criterion with an independent forward-error floor for
    # coefficients cast back to the caller's dtype. This does not reuse production tolerances.
    residual_tol = 2e-6 * atom_norms * residual.norm()
    cast_tol = 8.0 * input_epsilon * atom_norms * scale
    dual_tol = torch.maximum(residual_tol, cast_tol)
    positive = coefficients > 0
    assert bool((coefficients >= 0).all()), "primal feasibility (c >= 0)"
    assert bool((dual[~positive] <= dual_tol[~positive]).all()), "dual feasibility (zeroed)"
    assert bool((dual[positive].abs() <= dual_tol[positive]).all()), "stationarity (positive)"
    complementarity = coefficients.abs() * dual.abs()
    assert bool(
        (complementarity <= coefficients.abs() * dual_tol).all()
    ), "complementarity (c_i w_i = 0)"


def test_exact_resolve_recovers_true_optimum_when_an_atom_must_re_enter():
    """The active-set NNLS re-solve must return the true nonnegative-least-squares optimum,
    which requires letting a released atom re-enter the active set. On this tall, full-rank
    system (atoms as columns, no zero atom) a one-pass "drop the most negative coefficient,
    never re-add" rule strands atoms and returns a badly suboptimal fit; the exact optimum
    uses atoms 1 and 2."""
    active_atoms = torch.tensor(
        [[2.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -2.0, 2.0], [-2.0, 1.0, 1.0]]
    )
    target = torch.tensor([-1.0, -2.0, 0.0, 2.0])

    coefficients = _nonnegative_least_squares(active_atoms, target)

    assert (coefficients >= 0).all()
    assert torch.allclose(coefficients, torch.tensor([0.0, 0.95, 0.55]), atol=1e-4)
    residual = (active_atoms @ coefficients - target).norm().item()
    assert residual == pytest.approx(1.923538, abs=1e-4)
    assert torch.allclose(coefficients.double(), _reference_nnls(active_atoms, target), atol=1e-4)
    _assert_nnls_kkt(active_atoms, target, coefficients)


@pytest.mark.parametrize("seed", range(60))
def test_exact_resolve_matches_independent_brute_force_nnls(seed):
    """``_nonnegative_least_squares`` equals an independent brute-force NNLS oracle across many
    shapes (tall, square and underdetermined). This is the exactness check the existing suite
    lacked, since its support oracle reuses the function under test."""
    torch.manual_seed(seed)
    num_active = int(torch.randint(2, 7, (1,)))
    d_model = int(torch.randint(2, 9, (1,)))
    active_atoms = torch.randn(d_model, num_active)
    target = torch.randn(d_model)

    coefficients = _nonnegative_least_squares(active_atoms, target)

    assert (coefficients >= 0).all()
    got = (active_atoms.double() @ coefficients.double() - target.double()).norm().item()
    reference = (
        (active_atoms.double() @ _reference_nnls(active_atoms, target) - target.double())
        .norm()
        .item()
    )
    # Two-sided: the solver must match the exhaustive optimum, not merely not exceed it. A
    # residual materially below the "exhaustive" optimum is evidence the oracle is wrong, not
    # that the solver is better.
    assert abs(got - reference) <= 1e-4
    _assert_nnls_kkt(active_atoms, target, coefficients)


# --------------------------------------------------------------------------- #
# NNLS optimality certificates on degenerate and rescaled systems
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(6, 3), (4, 4), (3, 5)])  # tall, square, underdetermined
def test_nnls_satisfies_kkt_on_varied_shapes(shape):
    d_model, num_active = shape
    torch.manual_seed(100 * d_model + num_active)
    active_atoms = torch.randn(d_model, num_active)
    target = torch.randn(d_model)
    coefficients = _nonnegative_least_squares(active_atoms, target)
    _assert_nnls_kkt(active_atoms, target, coefficients)


@pytest.mark.parametrize("seed", range(10))
def test_nnls_kkt_and_objective_on_duplicate_columns(seed):
    """Rank-deficient (duplicate columns): the enumeration oracle can pick a negative
    minimum-norm representative for a non-unique support, so certify via KKT plus a one-sided
    objective bound (the solver is no worse than the oracle's feasible optimum)."""
    torch.manual_seed(3000 + seed)
    shared = torch.randn(4)
    active_atoms = torch.stack([shared, shared, torch.randn(4), torch.randn(4)], dim=1)
    target = torch.randn(4)
    coefficients = _nonnegative_least_squares(active_atoms, target)
    _assert_nnls_kkt(active_atoms, target, coefficients)
    got = (active_atoms.double() @ coefficients.double() - target.double()).norm().item()
    reference = (
        (active_atoms.double() @ _reference_nnls(active_atoms, target) - target.double())
        .norm()
        .item()
    )
    assert got <= reference + 1e-4


@pytest.mark.parametrize("seed", range(10))
def test_nnls_kkt_on_nearly_collinear_columns(seed):
    """Ill-conditioned near-collinear columns force large coefficients and catastrophic
    cancellation in ``target - A c``; the scale-aware tolerance must still certify KKT rather
    than raise on the resulting (correct) large-coefficient fit."""
    torch.manual_seed(4000 + seed)
    base = torch.randn(5)
    active_atoms = torch.stack([base, base + 1e-4 * torch.randn(5), torch.randn(5)], dim=1)
    target = torch.randn(5)
    coefficients = _nonnegative_least_squares(active_atoms, target)
    _assert_nnls_kkt(active_atoms, target, coefficients)


def test_nnls_returns_zero_on_zero_target():
    coefficients = _nonnegative_least_squares(torch.randn(4, 3), torch.zeros(4))
    assert torch.allclose(coefficients, torch.zeros(3), atol=1e-6)


def test_nnls_accepts_empty_active_set():
    coefficients = _nonnegative_least_squares(torch.empty(4, 0), torch.randn(4))
    assert coefficients.shape == (0,)


@pytest.mark.parametrize("input_name", ["active_atoms", "target"])
def test_nnls_rejects_non_finite_inputs(input_name):
    active_atoms = torch.eye(3)
    target = torch.ones(3)
    if input_name == "active_atoms":
        active_atoms[0, 0] = float("nan")
    else:
        target[0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        _nonnegative_least_squares(active_atoms, target)


def test_nnls_preserves_result_dtype_and_device():
    active_atoms = torch.eye(3, dtype=torch.float32)
    target = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    coefficients = _nonnegative_least_squares(active_atoms, target)
    assert coefficients.dtype == target.dtype
    assert coefficients.device == target.device


@pytest.mark.parametrize("scale", [1e-3, 1e-2, 1.0, 1e2, 1e3])
def test_nnls_coefficients_are_invariant_under_joint_rescaling(scale):
    """Jointly rescaling ``A`` and ``x`` leaves ``argmin over c >= 0`` unchanged; the
    scale-aware tolerance keeps the solver numerically consistent across magnitudes."""
    torch.manual_seed(7)
    active_atoms = torch.randn(6, 4)
    target = torch.randn(6)
    base = _nonnegative_least_squares(active_atoms, target)
    scaled = _nonnegative_least_squares(active_atoms * scale, target * scale)
    assert torch.allclose(scaled, base, atol=1e-3, rtol=1e-3)


def test_nnls_is_invariant_to_appending_zero_rows():
    """Zero equations do not change an NNLS problem or its numerical rank policy."""
    torch.manual_seed(17)
    active_atoms = torch.randn(16, 8)
    target = torch.randn(16)
    base = _nonnegative_least_squares(active_atoms, target)
    padded = _nonnegative_least_squares(
        torch.cat([active_atoms, torch.zeros(752, 8)]),
        torch.cat([target, torch.zeros(752)]),
    )
    assert torch.allclose(padded, base, atol=1e-5, rtol=1e-5)


def test_nnls_realistic_width_and_active_set_match_float64_reference():
    """At GPT-2 residual width and the default k, fp32 inputs match a float64 solve."""
    torch.manual_seed(117)
    active_atoms = torch.randn(768, 25)
    target = torch.randn(768)
    coefficients = _nonnegative_least_squares(active_atoms, target)
    reference = _nonnegative_least_squares(active_atoms.double(), target.double())

    got_residual = (active_atoms.double() @ coefficients.double() - target.double()).norm()
    reference_residual = (active_atoms.double() @ reference - target.double()).norm()
    assert torch.allclose(got_residual, reference_residual, atol=1e-7, rtol=1e-7)
    assert torch.allclose(coefficients.double(), reference, atol=2e-6, rtol=2e-6)
    _assert_nnls_kkt(active_atoms, target, coefficients)


def test_nnls_returns_exact_zeros_outside_numerical_passive_set():
    torch.manual_seed(23)
    active_atoms = torch.randn(64, 25)
    target = torch.randn(64)
    coefficients = _nonnegative_least_squares(active_atoms, target)
    _, coefficient_tol = _nnls_tolerances(
        active_atoms.double(), target.double(), coefficients.double()
    )
    assert bool(((coefficients == 0) | (coefficients.double() > coefficient_tol)).all())


def test_nnls_fails_closed_when_it_cannot_converge(monkeypatch):
    """A safeguard exhaustion must raise, never return an unverified vector. Forcing every
    admitted atom to be judged numerically zero (huge coefficient tolerance) stops the passive
    set from ever stabilising, and an unreachable dual threshold stops the KKT break from ever
    firing, so the bounded outer loop must fail closed for any input."""
    import transformer_lens.tools.analysis.jacobian_lens_decomposition as module

    def unsatisfiable(active_atoms, target, coefficients):
        num_active = active_atoms.shape[1]
        return (
            target.new_full((num_active,), -1e30),  # dual break unreachable
            target.new_full((num_active,), 1e30),  # every coefficient judged "zero"
        )

    monkeypatch.setattr(module, "_nnls_tolerances", unsatisfiable)
    with pytest.raises(RuntimeError):
        module._nonnegative_least_squares(torch.randn(5, 3), torch.randn(5))


def test_validate_nnls_kkt_rejects_each_violation():
    """Accept the optimum and reject primal, dual, stationarity and complementarity errors."""
    active_atoms = torch.eye(2)
    target = torch.tensor([1.0, 1.0])
    dual_tol, coefficient_tol = _nnls_tolerances(active_atoms, target, torch.zeros(2))

    # The true optimum passes.
    _validate_nnls_kkt(active_atoms, target, torch.tensor([1.0, 1.0]), dual_tol, coefficient_tol)

    # Primal: a negative coordinate.
    with pytest.raises(RuntimeError, match="primal"):
        _validate_nnls_kkt(
            active_atoms, target, torch.tensor([-1.0, 1.0]), dual_tol, coefficient_tol
        )
    # Dual feasibility: an all-zero fit leaves both zeroed atoms with large positive correlation.
    with pytest.raises(RuntimeError, match="dual feasibility"):
        _validate_nnls_kkt(
            active_atoms, target, torch.tensor([0.0, 0.0]), dual_tol, coefficient_tol
        )
    # Stationarity: positive coordinates whose residual correlation is far from zero.
    with pytest.raises(RuntimeError, match="stationarity"):
        _validate_nnls_kkt(
            active_atoms, target, torch.tensor([0.5, 0.5]), dual_tol, coefficient_tol
        )
    # Complementarity: classify a small positive coefficient as zero and give it a large
    # negative dual. Primal and one-sided dual feasibility pass, but c_i * w_i does not.
    with pytest.raises(RuntimeError, match="complementarity"):
        _validate_nnls_kkt(
            active_atoms,
            torch.tensor([-10.0, -10.0]),
            torch.tensor([0.5, 0.5]),
            torch.full((2,), 1.0),
            torch.full((2,), 1.0),
        )


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


def test_model_free_decomposition_preserves_autograd_contract():
    """The model-free primitive must not silently detach caller-owned tensors."""
    dictionary = torch.eye(3, requires_grad=True)
    x = torch.tensor([3.0, 2.0, 1.0], requires_grad=True)
    result = get_sparse_decomposition(x, dictionary, k=2, algorithm="gradient_pursuit")
    assert result.reconstruction.requires_grad


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


def test_rejects_non_finite_atom_norm():
    dictionary = torch.eye(4)
    dictionary[0, :2] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="non-finite or zero-norm"):
        get_sparse_decomposition(torch.ones(4), dictionary, k=2)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_rejects_non_finite_target(algorithm):
    target = torch.ones(4)
    target[1] = float("inf")
    with pytest.raises(ValueError, match="x contains non-finite"):
        get_sparse_decomposition(target, torch.eye(4), k=2, algorithm=algorithm)


@pytest.mark.parametrize("complex_input", ["x", "dictionary"])
def test_rejects_complex_inputs(complex_input):
    target = torch.ones(4)
    dictionary = torch.eye(4)
    if complex_input == "x":
        target = target.to(torch.complex64)
    else:
        dictionary = dictionary.to(torch.complex64)
    with pytest.raises(ValueError, match="real-valued"):
        get_sparse_decomposition(target, dictionary, k=2)


def test_rejects_non_2d_dictionary():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(4), torch.ones(4), k=1)


def test_rejects_non_1d_target():
    with pytest.raises(ValueError):
        get_sparse_decomposition(torch.ones(2, 4), torch.eye(4), k=1)
