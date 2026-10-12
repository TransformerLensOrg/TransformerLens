"""Model-free unit tests for J-space sparse decomposition and occupancy."""

import itertools
import warnings

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
    JSpaceOccupancy,
    _gradient_pursuit_step,
    _nnls_tolerances,
    _nonnegative_least_squares,
    _validate_nnls_kkt,
    estimate_occupancy,
    get_sparse_decomposition,
)

ALGORITHMS = ["nonnegative_orthogonal_matching_pursuit", "gradient_pursuit"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_recovers_planted_combination_on_orthonormal_dictionary(algorithm):
    """Both algorithms recover the planted support and coefficients on orthonormal atoms."""
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
    """``j_space_component`` projects onto the selected span for either update rule."""
    atoms = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],  # correlated with atom 0
            [0.0, 0.0, 1.0],
        ]
    )
    x = torch.tensor([3.8, 2.4, 7.0])  # xy part in span{atom0, atom1}; z part out of it

    result = get_sparse_decomposition(x, atoms, k=2, algorithm=algorithm)

    assert set(result.selected_support.tolist()) == {1, 2}
    assert torch.allclose(result.j_space_component + result.non_j_space_component, x, atol=1e-5)
    for i in result.selected_support.tolist():
        assert torch.dot(result.non_j_space_component, atoms[i]).abs().item() < 1e-5
    selected_atoms = atoms[result.selected_support].T
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
    torch.manual_seed(0)
    dictionary = torch.randn(20, 6)
    x = torch.randn(6)
    first = get_sparse_decomposition(x, dictionary, k=4, algorithm=algorithm)
    second = get_sparse_decomposition(x, dictionary, k=4, algorithm=algorithm)
    assert torch.equal(first.support, second.support)
    assert torch.allclose(first.coordinates, second.coordinates)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_support_is_the_active_subset_of_selected_support(algorithm):
    """``support`` is the strictly-positive subset of ``selected_support``, within ``k``."""
    torch.manual_seed(2)
    dictionary = torch.randn(40, 6)
    x = torch.randn(6)
    k = 10

    result = get_sparse_decomposition(x, dictionary, k=k, algorithm=algorithm)

    assert result.support.numel() <= result.selected_support.numel() <= k
    assert set(result.support.tolist()).issubset(set(result.selected_support.tolist()))
    assert result.coordinates.numel() == result.support.numel()
    assert bool((result.coordinates > 0).all())


def test_overcomplete_correlated_dictionary_has_no_dead_support_slots():
    """Selection stops early and ``support`` is never padded to ``k`` with zero coordinates."""
    torch.manual_seed(0)
    bases = torch.randn(10, 32)
    dictionary = torch.randn(400, 10).abs() @ bases + 0.05 * torch.randn(400, 32)
    x = torch.randn(32)

    result = get_sparse_decomposition(x, dictionary, k=25)

    assert result.selected_support.numel() < 25
    assert result.support.numel() == int((result.coordinates > 0).sum())
    assert bool((result.coordinates > 0).all())


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_zero_target_returns_empty_decomposition(algorithm):
    dictionary = torch.randn(6, 4)

    result = get_sparse_decomposition(torch.zeros(4), dictionary, k=3, algorithm=algorithm)

    assert result.support.numel() == 0
    assert result.selected_support.numel() == 0
    assert result.coordinates.numel() == 0
    assert torch.allclose(result.reconstruction, torch.zeros(4))
    assert torch.allclose(result.j_space_component, torch.zeros(4))
    assert torch.allclose(result.non_j_space_component, torch.zeros(4))


def test_target_orthogonal_to_all_atoms_gives_empty_active_support():
    atoms = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    x = torch.tensor([0.0, 0.0, 5.0])

    result = get_sparse_decomposition(x, atoms, k=2)

    assert result.support.numel() == 0
    assert torch.allclose(result.reconstruction, torch.zeros(3))
    assert torch.allclose(result.non_j_space_component, x)


@pytest.mark.parametrize("scale", [0.25, 3.0, 50.0])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_support_is_invariant_under_positive_rescaling_of_target(scale, algorithm):
    """Both thresholds scale with ``||x||``, so supports are fixed and coordinates scale."""
    torch.manual_seed(5)
    dictionary = torch.randn(30, 6)
    x = torch.randn(6)

    base = get_sparse_decomposition(x, dictionary, k=10, algorithm=algorithm)
    scaled = get_sparse_decomposition(scale * x, dictionary, k=10, algorithm=algorithm)

    assert scaled.support.tolist() == base.support.tolist()
    assert scaled.selected_support.tolist() == base.selected_support.tolist()
    assert torch.allclose(scaled.coordinates, scale * base.coordinates, rtol=1e-4, atol=1e-4)


def test_active_reconstruction_is_a_projection_while_selected_projection_can_differ():
    """NNLS reconstruction projects onto the active span; ``j_space_component`` onto the larger
    selected span (seed 7 leaves its first selected atom inactive)."""
    torch.manual_seed(7)
    dictionary = torch.randn(6, 3)
    x = torch.randn(3)

    result = get_sparse_decomposition(x, dictionary, k=6)

    assert result.selected_support.numel() > result.support.numel()
    active_atoms = dictionary[result.support].T
    selected_atoms = dictionary[result.selected_support].T
    projection_onto_active = active_atoms @ torch.linalg.pinv(active_atoms) @ x
    projection_onto_selected = selected_atoms @ torch.linalg.pinv(selected_atoms) @ x
    assert torch.allclose(result.reconstruction, projection_onto_active, atol=1e-4)
    assert torch.allclose(result.j_space_component, projection_onto_selected, atol=1e-4)
    assert not torch.allclose(result.reconstruction, result.j_space_component, atol=1e-3)


def test_exact_resolve_coordinates_match_joint_least_squares_on_correlated_dictionary():
    """Correlated atoms get the joint least-squares coefficients, not sequential subtraction."""
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


def test_selection_stops_when_no_unselected_atom_is_positively_correlated():
    """A negatively correlated atom is not selected even when ``k`` allows it; the unconstrained
    fit of x is (3, -1), so the nonnegative coefficient must be 2.0."""
    atoms = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    x = torch.tensor([2.0, -1.0])

    result = get_sparse_decomposition(
        x, atoms, k=2, algorithm="nonnegative_orthogonal_matching_pursuit"
    )

    assert result.selected_support.tolist() == [0]
    assert result.support.tolist() == [0]
    assert result.coordinates.tolist() == pytest.approx([2.0], abs=1e-4)  # not the clamped 3.0
    assert torch.allclose(result.reconstruction, torch.tensor([2.0, 0.0]), atol=1e-4)


def _reference_nnls(active_atoms: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Brute-force NNLS by support enumeration; shares no code with the solver under test."""
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
    """Independent NNLS KKT certificate, looser than the solver's own validator."""
    input_epsilon = torch.finfo(active_atoms.dtype).eps
    active_atoms = active_atoms.double()
    target = target.double()
    coefficients = coefficients.double()
    residual = target - active_atoms @ coefficients
    atom_norms = active_atoms.norm(dim=0).clamp_min(torch.finfo(torch.float64).tiny)
    dual = active_atoms.T @ residual
    scale = target.norm() + (active_atoms.abs() @ coefficients.abs()).norm()
    # Residual-cosine criterion plus a forward-error floor for the cast back to the input dtype.
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
    """The optimum here needs a released atom to re-enter; a drop-only rule strands atoms."""
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
    """The solver matches the brute-force NNLS oracle on tall, square and wide systems."""
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
    # Two-sided: a residual below the exhaustive optimum would mean the oracle is wrong.
    assert abs(got - reference) <= 1e-4
    _assert_nnls_kkt(active_atoms, target, coefficients)


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
    """Duplicate columns make the oracle's support non-unique, so only a one-sided bound holds."""
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
    """Near-collinear columns cancel in ``target - A c``; the scaled tolerance must still pass."""
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
    torch.manual_seed(7)
    active_atoms = torch.randn(6, 4)
    target = torch.randn(6)
    base = _nonnegative_least_squares(active_atoms, target)
    scaled = _nonnegative_least_squares(active_atoms * scale, target * scale)
    assert torch.allclose(scaled, base, atol=1e-3, rtol=1e-3)


def test_nnls_is_invariant_to_appending_zero_rows():
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
    """Unsatisfiable tolerances must exhaust the safeguards and raise, never return."""
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
    active_atoms = torch.eye(2)
    target = torch.tensor([1.0, 1.0])
    dual_tol, coefficient_tol = _nnls_tolerances(active_atoms, target, torch.zeros(2))

    _validate_nnls_kkt(active_atoms, target, torch.tensor([1.0, 1.0]), dual_tol, coefficient_tol)

    with pytest.raises(RuntimeError, match="primal"):
        _validate_nnls_kkt(
            active_atoms, target, torch.tensor([-1.0, 1.0]), dual_tol, coefficient_tol
        )
    with pytest.raises(RuntimeError, match="dual feasibility"):
        _validate_nnls_kkt(
            active_atoms, target, torch.tensor([0.0, 0.0]), dual_tol, coefficient_tol
        )
    with pytest.raises(RuntimeError, match="stationarity"):
        _validate_nnls_kkt(
            active_atoms, target, torch.tensor([0.5, 0.5]), dual_tol, coefficient_tol
        )
    # Positive coefficients judged zero with a large negative dual pass every check but c_i w_i.
    with pytest.raises(RuntimeError, match="complementarity"):
        _validate_nnls_kkt(
            active_atoms,
            torch.tensor([-10.0, -10.0]),
            torch.tensor([0.5, 0.5]),
            torch.full((2,), 1.0),
            torch.full((2,), 1.0),
        )


def test_gradient_pursuit_agrees_with_exact_resolve_on_orthonormal_dictionary():
    """One optimal-step gradient update is exact for orthonormal atoms."""
    orthonormal = torch.linalg.qr(torch.randn(6, 6)).Q
    x = 2.0 * orthonormal[0] + 1.5 * orthonormal[3]

    gradient_result = get_sparse_decomposition(x, orthonormal, k=2, algorithm="gradient_pursuit")
    exact_result = get_sparse_decomposition(
        x, orthonormal, k=2, algorithm="nonnegative_orthogonal_matching_pursuit"
    )

    assert set(gradient_result.support.tolist()) == set(exact_result.support.tolist()) == {0, 3}
    assert torch.allclose(gradient_result.reconstruction, exact_result.reconstruction, atol=1e-5)
    assert torch.allclose(gradient_result.reconstruction, x, atol=1e-5)


def test_gradient_pursuit_does_not_increase_residual_on_correlated_dictionary():
    torch.manual_seed(2)
    dictionary = torch.randn(12, 5)
    x = torch.randn(5)

    result = get_sparse_decomposition(x, dictionary, k=4, algorithm="gradient_pursuit")

    assert (result.coordinates >= 0).all()
    assert (x - result.reconstruction).norm().item() <= x.norm().item() + 1e-6


def test_model_free_decomposition_preserves_autograd_contract():
    dictionary = torch.eye(3, requires_grad=True)
    x = torch.tensor([3.0, 2.0, 1.0], requires_grad=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Converting a tensor with requires_grad=True to a scalar"
        )
        result = get_sparse_decomposition(x, dictionary, k=2, algorithm="gradient_pursuit")
    assert result.reconstruction.requires_grad
    result.reconstruction.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    assert dictionary.grad is not None and bool(torch.isfinite(dictionary.grad).all())


def test_gradient_pursuit_support_is_only_final_active_atoms():
    torch.manual_seed(3)
    dictionary = torch.randn(20, 6)
    x = torch.randn(6)

    result = get_sparse_decomposition(x, dictionary, k=5, algorithm="gradient_pursuit")

    assert bool((result.coordinates > 0).all())
    assert result.coordinates.numel() == result.support.numel()
    assert set(result.support.tolist()).issubset(set(result.selected_support.tolist()))


def test_gradient_pursuit_step_never_increases_the_residual():
    torch.manual_seed(1)
    dictionary = torch.randn(12, 5)
    x = torch.randn(5)

    previous = x.norm().item()
    for k in range(1, 6):
        result = get_sparse_decomposition(x, dictionary, k=k, algorithm="gradient_pursuit")
        current = (x - result.reconstruction).norm().item()
        assert current <= previous + 1e-5
        previous = current


def test_gradient_pursuit_backtracks_when_projection_makes_the_full_step_ascend():
    active_atoms = torch.tensor([[-2.0, -2.0], [-1.0, 1.0]])
    target = torch.tensor([-2.0, -2.0])
    coefficients = torch.tensor([1.0, 0.0])
    initial_residual = (target - active_atoms @ coefficients).norm()

    updated = _gradient_pursuit_step(active_atoms, target, coefficients)

    updated_residual = (target - active_atoms @ updated).norm()
    assert updated_residual < initial_residual
    assert not torch.equal(updated, coefficients)


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
    """On orthonormal atoms greedy selection is optimal and finds the planted support."""
    orthonormal = torch.linalg.qr(torch.randn(8, 8)).Q
    x = 2.0 * orthonormal[1] + 3.0 * orthonormal[4] + 1.0 * orthonormal[6]
    k = 3

    result = get_sparse_decomposition(
        x, orthonormal, k=k, algorithm="nonnegative_orthogonal_matching_pursuit"
    )
    best_support, _ = _brute_force_best_support(x, orthonormal, k)

    assert set(result.support.tolist()) == set(best_support) == {1, 4, 6}


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


@pytest.mark.parametrize("planted_atom_count", [1, 4, 7])
def test_occupancy_recovers_planted_sparsity_on_orthonormal_dictionary(planted_atom_count):
    """A k-sparse planted target on orthonormal atoms has occupancy exactly k."""
    torch.manual_seed(0)
    d_model = 24
    dictionary = torch.linalg.qr(torch.randn(d_model, d_model)).Q
    target = torch.zeros(d_model)
    for index in range(planted_atom_count):
        target = target + (3.0 - 0.3 * index) * dictionary[2 * index + 1]

    max_atoms = min(2 * planted_atom_count + 4, d_model)
    result = estimate_occupancy(target, dictionary, max_atoms=max_atoms)

    assert isinstance(result, JSpaceOccupancy)
    assert result.occupancy == planted_atom_count
    assert result.support.numel() == max_atoms


def test_occupancy_increases_with_planted_sparsity():
    torch.manual_seed(1)
    d_model = 24
    dictionary = torch.linalg.qr(torch.randn(d_model, d_model)).Q

    def planted(count):
        target = torch.zeros(d_model)
        for index in range(count):
            target = target + (3.0 - 0.2 * index) * dictionary[2 * index + 1]
        return target

    assert (
        estimate_occupancy(planted(2), dictionary, max_atoms=20).occupancy
        < estimate_occupancy(planted(9), dictionary, max_atoms=20).occupancy
    )


def test_occupancy_is_deterministic():
    torch.manual_seed(2)
    dictionary = torch.randn(40, 8)
    target = torch.randn(8)
    first = estimate_occupancy(target, dictionary, seed=0)
    second = estimate_occupancy(target, dictionary, seed=0)
    assert first.occupancy == second.occupancy
    assert torch.equal(first.support, second.support)
    assert torch.allclose(first.control_captured_variance, second.control_captured_variance)


def test_occupancy_rejects_invalid_inputs():
    dictionary = torch.eye(6)
    with pytest.raises(ValueError, match="max_atoms must be between"):
        estimate_occupancy(torch.ones(6), dictionary, max_atoms=0)
    with pytest.raises(ValueError, match="max_atoms must be between"):
        estimate_occupancy(torch.ones(6), dictionary, max_atoms=99)
    with pytest.raises(ValueError, match="num_control_dictionaries must be at least 1"):
        estimate_occupancy(torch.ones(6), dictionary, max_atoms=6, num_control_dictionaries=0)
    with pytest.raises(ValueError, match="x must be 1-D of length"):
        estimate_occupancy(torch.ones(5), dictionary)
    with pytest.raises(ValueError, match="x must be 1-D of length"):
        estimate_occupancy(torch.ones(2, 6), dictionary)
    zero_norm_dictionary = torch.eye(6)
    zero_norm_dictionary[2] = 0.0
    with pytest.raises(ValueError, match="dictionary contains a non-finite or zero-norm atom"):
        estimate_occupancy(torch.ones(6), zero_norm_dictionary, max_atoms=6)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_occupancy_rejects_non_finite_target(invalid_value):
    target = torch.ones(6)
    target[0] = invalid_value
    with pytest.raises(ValueError, match="x contains non-finite entries"):
        estimate_occupancy(target, torch.eye(6), max_atoms=6)


def test_occupancy_rejects_zero_norm_target():
    with pytest.raises(ValueError, match="x must have non-zero norm"):
        estimate_occupancy(torch.zeros(6), torch.eye(6), max_atoms=6)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_occupancy_rejects_non_finite_dictionary(invalid_value):
    dictionary = torch.eye(6)
    dictionary[0, 0] = invalid_value
    with pytest.raises(ValueError, match="dictionary contains non-finite entries"):
        estimate_occupancy(torch.ones(6), dictionary, max_atoms=6)


def test_occupancy_rejects_non_finite_dictionary_norm():
    dictionary = torch.eye(6)
    dictionary[0] = torch.finfo(torch.float32).max
    with pytest.raises(ValueError, match="dictionary contains a non-finite or zero-norm atom"):
        estimate_occupancy(torch.ones(6), dictionary, max_atoms=6)


@pytest.mark.parametrize("complex_input", ["x", "dictionary"])
def test_occupancy_rejects_complex_inputs(complex_input):
    target = torch.ones(6)
    dictionary = torch.eye(6)
    if complex_input == "x":
        target = target.to(torch.complex64)
    else:
        dictionary = dictionary.to(torch.complex64)
    with pytest.raises(ValueError, match="x and dictionary must be real-valued"):
        estimate_occupancy(target, dictionary, max_atoms=6)


def test_occupancy_rejects_non_finite_target_norm():
    target = torch.full((6,), torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="x must have finite norm"):
        estimate_occupancy(target, torch.eye(6), max_atoms=6)
