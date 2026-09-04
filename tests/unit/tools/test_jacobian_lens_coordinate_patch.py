"""Model-free tests for anchored J-space coordinate patching."""

import math
import warnings
from dataclasses import replace

import pytest
import torch

from transformer_lens.tools.analysis.jacobian_lens_coordinate_patch import (
    CoordinatePatch,
    solve_coordinate_patch,
)
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    JSpaceDecomposition,
    get_sparse_decomposition,
)


def _problem() -> tuple[torch.Tensor, torch.Tensor]:
    dictionary = torch.eye(5)[:4]
    residual = torch.tensor([0.0, 0.0, 0.0, 0.0, -4.0])
    activation = 2.0 * dictionary[0] + 3.0 * dictionary[1] + residual
    return activation, dictionary


def test_substitute_appends_absent_target_and_preserves_frame() -> None:
    activation, dictionary = _problem()

    result = solve_coordinate_patch(activation, dictionary, source_idx=0, target_idx=2, k=2)

    assert isinstance(result, CoordinatePatch)
    assert result.support_before.tolist() == [1, 0]
    assert result.support_after.tolist() == [1, 0, 2]
    assert result.source_slot == 1
    assert result.target_slot == 2
    torch.testing.assert_close(result.coordinates_before, torch.tensor([3.0, 2.0, 0.0]))
    torch.testing.assert_close(result.coordinates_after, torch.tensor([3.0, 0.0, 2.0]))
    torch.testing.assert_close(result.residual, torch.tensor([0.0, 0.0, 0.0, 0.0, -4.0]))
    torch.testing.assert_close(result.reconstruction_after, torch.tensor([0.0, 3.0, 2.0, 0.0, 0.0]))
    torch.testing.assert_close(result.patched, result.residual + result.reconstruction_after)
    torch.testing.assert_close(result.delta, torch.tensor([-2.0, 0.0, 2.0, 0.0, 0.0]))
    assert result.target_was_appended is True
    assert result.overwritten_target_coordinate is None
    assert result.nonedited_coordinate_max_delta == pytest.approx(0.0, abs=1e-6)
    assert result.residual_max_delta == pytest.approx(0.0, abs=1e-6)


def test_active_target_substitute_overwrites_while_swap_exchanges() -> None:
    dictionary = torch.eye(3)
    activation = torch.tensor([2.0, 5.0, 0.0])
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)

    substitute = solve_coordinate_patch(
        activation, dictionary, 0, 1, decomposition=decomposition, mode="substitute"
    )
    swap = solve_coordinate_patch(
        activation, dictionary, 0, 1, decomposition=decomposition, mode="swap"
    )

    torch.testing.assert_close(substitute.patched, torch.tensor([0.0, 2.0, 0.0]))
    torch.testing.assert_close(swap.patched, torch.tensor([5.0, 2.0, 0.0]))
    assert substitute.target_was_appended is False
    assert swap.target_was_appended is False


def test_overwritten_target_coordinate_reports_exact_discarded_value() -> None:
    """Q2 value-level: substitute onto an active target reports the exact discarded coordinate."""
    dictionary = torch.eye(3)
    activation = torch.tensor([2.0, 7.25, 0.0])
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)

    substitute = solve_coordinate_patch(
        activation, dictionary, 0, 1, decomposition=decomposition, mode="substitute"
    )
    swap = solve_coordinate_patch(
        activation, dictionary, 0, 1, decomposition=decomposition, mode="swap"
    )
    absent = solve_coordinate_patch(
        activation, dictionary, 0, 2, decomposition=decomposition, mode="substitute"
    )

    assert substitute.overwritten_target_coordinate == pytest.approx(7.25)
    assert swap.overwritten_target_coordinate is None
    assert absent.overwritten_target_coordinate is None


def test_alpha_interpolates_and_zero_is_bit_exact() -> None:
    activation, dictionary = _problem()
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)
    full = solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=decomposition)
    half = solve_coordinate_patch(
        activation, dictionary, 0, 2, decomposition=decomposition, alpha=0.5
    )
    zero = solve_coordinate_patch(
        activation, dictionary, 0, 2, decomposition=decomposition, alpha=0.0
    )

    torch.testing.assert_close(half.delta, 0.5 * full.delta)
    torch.testing.assert_close(half.patched, activation + 0.5 * full.delta)
    assert torch.equal(zero.patched, activation)
    assert torch.equal(zero.delta, torch.zeros_like(activation))
    assert zero.patched.data_ptr() != activation.data_ptr()


def test_small_alpha_delta_is_proportional_not_floored() -> None:
    activation, dictionary = _problem()
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)
    full = solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=decomposition)
    tiny = solve_coordinate_patch(
        activation, dictionary, 0, 2, decomposition=decomposition, alpha=1e-4
    )

    # delta scales strictly with alpha: no alpha-independent recompute floor survives at 1e-4.
    torch.testing.assert_close(tiny.delta, 1e-4 * full.delta, rtol=1e-3, atol=1e-7)
    assert float(tiny.delta.abs().max()) < 1e-3


def test_preserves_reconstruction_residual_not_selected_span_residual() -> None:
    dictionary = torch.eye(3)
    activation = torch.tensor([2.0, -1.0, 3.0])
    decomposition = JSpaceDecomposition(
        support=torch.tensor([0]),
        coordinates=torch.tensor([2.0]),
        selected_support=torch.tensor([0, 1]),
        reconstruction=torch.tensor([2.0, 0.0, 0.0]),
        j_space_component=torch.tensor([2.0, -1.0, 0.0]),
        non_j_space_component=torch.tensor([0.0, 0.0, 3.0]),
    )

    result = solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=decomposition)

    torch.testing.assert_close(result.residual, torch.tensor([0.0, -1.0, 3.0]))
    torch.testing.assert_close(result.patched, torch.tensor([0.0, -1.0, 5.0]))
    assert not torch.equal(result.residual, decomposition.non_j_space_component)


def test_witnesses_certify_untouched_coordinates_and_residual() -> None:
    dictionary = torch.eye(4)
    activation = torch.tensor([2.0, 3.0, 4.0, 0.0])
    decomposition = get_sparse_decomposition(activation, dictionary, k=3)

    result = solve_coordinate_patch(activation, dictionary, 0, 3, decomposition=decomposition)

    # Atom 1 and atom 2 are neither source nor target, so their coordinates must be untouched.
    assert result.nonedited_coordinate_max_delta == pytest.approx(0.0, abs=1e-6)
    assert result.residual_max_delta == pytest.approx(0.0, abs=1e-6)


def test_fresh_and_precomputed_decompositions_match_without_mutating_inputs() -> None:
    activation, dictionary = _problem()
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)
    activation_before = activation.clone()
    dictionary_before = dictionary.clone()
    coordinates_before = decomposition.coordinates.clone()

    fresh = solve_coordinate_patch(activation, dictionary, 0, 2, k=2, mode="swap")
    reused = solve_coordinate_patch(
        activation, dictionary, 0, 2, decomposition=decomposition, mode="swap"
    )

    assert torch.equal(fresh.support_after, reused.support_after)
    torch.testing.assert_close(fresh.coordinates_after, reused.coordinates_after)
    torch.testing.assert_close(fresh.patched, reused.patched)
    assert torch.equal(activation, activation_before)
    assert torch.equal(dictionary, dictionary_before)
    assert torch.equal(decomposition.coordinates, coordinates_before)


def test_repeated_calls_are_deterministic() -> None:
    activation, dictionary = _problem()

    first = solve_coordinate_patch(activation, dictionary, 0, 2, k=2)
    second = solve_coordinate_patch(activation, dictionary, 0, 2, k=2)

    assert torch.equal(first.patched, second.patched)
    assert torch.equal(first.support_after, second.support_after)


def test_float64_input_produces_float32_outputs() -> None:
    activation, dictionary = _problem()

    result = solve_coordinate_patch(activation.double(), dictionary, 0, 2, k=2)

    assert result.patched.dtype == torch.float32
    assert result.coordinates_after.dtype == torch.float32
    torch.testing.assert_close(result.patched, torch.tensor([0.0, 3.0, 2.0, 0.0, -4.0]))


@pytest.mark.parametrize(
    "algorithm", ["nonnegative_orthogonal_matching_pursuit", "gradient_pursuit"]
)
def test_both_decomposition_algorithms_pass_through(algorithm: str) -> None:
    activation, dictionary = _problem()

    result = solve_coordinate_patch(activation, dictionary, 0, 2, k=2, algorithm=algorithm)

    torch.testing.assert_close(result.patched, torch.tensor([0.0, 3.0, 2.0, 0.0, -4.0]))


def test_rejects_incompatible_precomputed_decomposition() -> None:
    activation, dictionary = _problem()
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)

    with pytest.raises(ValueError, match="activation"):
        solve_coordinate_patch(
            activation + dictionary[3], dictionary, 0, 2, decomposition=decomposition
        )
    changed_dictionary = dictionary.clone()
    changed_dictionary[0] *= 2
    with pytest.raises(ValueError, match="dictionary|reconstruction"):
        solve_coordinate_patch(activation, changed_dictionary, 0, 2, decomposition=decomposition)


def test_rejects_scaled_but_self_consistent_decomposition() -> None:
    activation, dictionary = _problem()
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)

    # Scale the (reconstruction, coordinates) pair together. The pair stays internally
    # consistent with the dictionary and still agrees with the untouched j-space
    # components, so every self-referential check passes -- but the coordinates are no
    # longer the NNLS fit of the activation, which the x-coupled stationarity check must
    # catch instead of silently anchoring the edit to a wrong residual.
    scaled = replace(
        decomposition,
        coordinates=decomposition.coordinates * 2.0,
        reconstruction=decomposition.reconstruction * 2.0,
    )
    with pytest.raises(ValueError, match="activation"):
        solve_coordinate_patch(activation, dictionary, 0, 1, decomposition=scaled)


def test_rejects_malformed_precomputed_decomposition() -> None:
    activation, dictionary = _problem()
    decomposition = get_sparse_decomposition(activation, dictionary, k=2)

    duplicate = replace(
        decomposition, support=torch.tensor([0, 0]), coordinates=torch.tensor([1.0, 1.0])
    )
    with pytest.raises(ValueError, match="duplicates"):
        solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=duplicate)
    outside = replace(decomposition, selected_support=torch.tensor([0, 1, 9]))
    with pytest.raises(ValueError, match="outside"):
        solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=outside)
    zero_coordinate = replace(
        decomposition, coordinates=torch.zeros_like(decomposition.coordinates)
    )
    with pytest.raises(ValueError, match="positive"):
        solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=zero_coordinate)
    bad_shape = replace(decomposition, reconstruction=torch.zeros(2, 5))
    with pytest.raises(ValueError, match=r"shape \[d_model\]"):
        solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=bad_shape)
    nonfinite = replace(
        decomposition,
        non_j_space_component=torch.full_like(decomposition.non_j_space_component, torch.nan),
    )
    with pytest.raises(ValueError, match="non-finite"):
        solve_coordinate_patch(activation, dictionary, 0, 2, decomposition=nonfinite)


def test_reuse_validates_only_atoms_used_by_the_edit() -> None:
    dictionary = torch.eye(4)
    activation = 2.0 * dictionary[0]
    decomposition = get_sparse_decomposition(activation, dictionary, k=1)
    dictionary[3] = torch.nan

    reused = solve_coordinate_patch(activation, dictionary, 0, 1, decomposition=decomposition)

    assert torch.isfinite(reused.patched).all()
    with pytest.raises(ValueError, match="non-finite"):
        solve_coordinate_patch(activation, dictionary, 0, 1, k=1)


def test_tiny_nonzero_atoms_are_valid() -> None:
    dictionary = torch.tensor([[1e-8, 0.0], [0.0, 1.0]])
    activation = 2.0 * dictionary[0]

    result = solve_coordinate_patch(activation, dictionary, 0, 1, k=1)

    torch.testing.assert_close(result.patched, torch.tensor([0.0, 2.0]))


def test_rejects_zero_norm_target_atom() -> None:
    # The target atom is edited, so a zero-norm target is rejected once the edit frame is built.
    dictionary = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    activation = torch.tensor([2.0, 0.0])

    with pytest.raises(ValueError, match="zero-norm"):
        solve_coordinate_patch(activation, dictionary, 0, 1, k=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_idx": 2, "target_idx": 3}, "active support"),
        ({"source_idx": 0, "target_idx": 0}, "distinct"),
        ({"source_idx": -1, "target_idx": 2}, "source_idx"),
        ({"source_idx": 0, "target_idx": 9}, "target_idx"),
        ({"source_idx": 0, "target_idx": 2, "mode": "bad"}, "mode"),
        ({"source_idx": 0, "target_idx": 2, "alpha": math.inf}, "finite"),
    ],
)
def test_rejects_invalid_edit_requests(kwargs: dict[str, object], message: str) -> None:
    activation, dictionary = _problem()
    with pytest.raises(ValueError, match=message):
        solve_coordinate_patch(activation, dictionary, k=2, **kwargs)


@pytest.mark.parametrize(
    ("bad_x", "message"),
    [
        (torch.zeros(2, 4), "x must be 1-D"),
        (torch.zeros(3), "x must be 1-D"),
        (torch.tensor([2.0, float("nan"), 0.0, 0.0]), "non-finite"),
    ],
)
def test_rejects_malformed_activation(bad_x: torch.Tensor, message: str) -> None:
    dictionary = torch.eye(4)
    with pytest.raises(ValueError, match=message):
        solve_coordinate_patch(bad_x, dictionary, 0, 1, k=1)


def test_rejects_malformed_dictionary() -> None:
    activation = torch.zeros(4)
    with pytest.raises(ValueError, match="dictionary must be 2-D"):
        solve_coordinate_patch(activation, torch.zeros(4), 0, 1, k=1)
    with pytest.raises(ValueError, match="at least one atom"):
        solve_coordinate_patch(activation, torch.zeros(0, 4), 0, 1, k=1)


def test_rejects_device_mismatch_between_activation_and_dictionary() -> None:
    activation, dictionary = _problem()
    meta_dictionary = dictionary.to("meta")
    with pytest.raises(ValueError, match="same device"):
        solve_coordinate_patch(activation, meta_dictionary, 0, 2, k=2)


def test_near_parallel_atoms_warn_but_return_patch() -> None:
    """Q3-ext: near-parallel source/target warns (naming the cosine) and never raises."""
    activation = torch.tensor([2.0, 0.0, 0.0])
    dictionary = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.05, 0.0], [0.0, 0.0, 1.0]])

    with pytest.warns(UserWarning, match="near-parallel") as record:
        result = solve_coordinate_patch(activation, dictionary, 0, 1, k=1)

    assert 0.99 <= abs(result.source_target_cosine) < 0.999
    assert f"{abs(result.source_target_cosine):.6f}" in str(record[0].message)
    assert torch.isfinite(result.patched).all()

    # The parallel extreme (abs cosine >= 0.999) must still warn-and-complete, not raise.
    extreme = dictionary.clone()
    extreme[1] = 2.0 * extreme[0]
    with pytest.warns(UserWarning, match="near-parallel") as extreme_record:
        extreme_result = solve_coordinate_patch(activation, extreme, 0, 1, k=1)

    assert abs(extreme_result.source_target_cosine) >= 0.999
    assert f"{abs(extreme_result.source_target_cosine):.6f}" in str(extreme_record[0].message)
    assert torch.isfinite(extreme_result.patched).all()


def test_rank_deficient_edit_basis_warns_but_returns_patch() -> None:
    dictionary = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    activation = torch.tensor([2.0, 3.0, 0.0])
    decomposition = JSpaceDecomposition(
        support=torch.tensor([0, 1, 2]),
        coordinates=torch.tensor([1.0, 2.0, 1.0]),
        selected_support=torch.tensor([0, 1, 2]),
        reconstruction=activation.clone(),
        j_space_component=activation.clone(),
        non_j_space_component=torch.zeros(3),
    )

    with pytest.warns(UserWarning, match="rank deficient"):
        result = solve_coordinate_patch(activation, dictionary, 0, 3, decomposition=decomposition)

    assert result.basis_rank == 3
    assert math.isinf(result.basis_condition_number)
    assert torch.isfinite(result.patched).all()


def test_poorly_conditioned_full_rank_basis_warns_with_measurement() -> None:
    dictionary = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1e-4, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    activation = dictionary[0] + dictionary[1] + dictionary[2]
    selected_atoms = dictionary[[0, 1, 2]].T
    component = selected_atoms @ torch.linalg.pinv(selected_atoms) @ activation
    decomposition = JSpaceDecomposition(
        support=torch.tensor([0, 1, 2]),
        coordinates=torch.ones(3),
        selected_support=torch.tensor([0, 1, 2]),
        reconstruction=activation.clone(),
        j_space_component=component,
        non_j_space_component=activation - component,
    )

    with pytest.warns(UserWarning, match="condition number=") as record:
        result = solve_coordinate_patch(activation, dictionary, 0, 3, decomposition=decomposition)

    assert result.basis_rank == 4
    assert math.isfinite(result.basis_condition_number)
    assert f"{result.basis_condition_number:.6g}" in str(record[0].message)


def test_well_conditioned_control_emits_no_warning() -> None:
    """Over-warning guard: an orthonormal, non-parallel edit must fire no warning at all."""
    activation, dictionary = _problem()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = solve_coordinate_patch(activation, dictionary, 0, 2, k=2)

    assert torch.isfinite(result.patched).all()


def test_large_d_model_full_rank_basis_reports_finite_condition() -> None:
    """C2: the rank tolerance must scale by the column count, not the d_model row count.

    A full-rank but poorly conditioned edit basis stays finite at model scale. Scaling the
    rank tolerance by ``max(shape)`` (the d_model row count) instead crosses the smallest
    singular value once ``d_model >= ~2896``, reporting the basis as rank deficient with an
    infinite condition number on Llama/Mistral/Gemma-scale dictionaries.
    """
    d_model = 3000

    def _atom(*entries: tuple[int, float]) -> torch.Tensor:
        atom = torch.zeros(d_model)
        for index, value in entries:
            atom[index] = value
        return atom

    dictionary = torch.stack(
        [_atom((0, 1.0)), _atom((1, 1.0)), _atom((0, 1.0), (1, 1.0), (2, 1e-4)), _atom((3, 1.0))]
    )
    activation = dictionary[0] + dictionary[1] + dictionary[2]
    selected_atoms = dictionary[[0, 1, 2]].T
    component = selected_atoms @ torch.linalg.pinv(selected_atoms) @ activation
    decomposition = JSpaceDecomposition(
        support=torch.tensor([0, 1, 2]),
        coordinates=torch.ones(3),
        selected_support=torch.tensor([0, 1, 2]),
        reconstruction=activation.clone(),
        j_space_component=component,
        non_j_space_component=activation - component,
    )

    with pytest.warns(UserWarning, match="condition number="):
        result = solve_coordinate_patch(activation, dictionary, 0, 3, decomposition=decomposition)

    assert result.basis_rank == 4
    assert math.isfinite(result.basis_condition_number)


def test_high_condition_decomposition_is_accepted() -> None:
    """C8: the compatibility tolerance must scale with the selected span's condition number.

    An exactly valid, hand-built decomposition is rejected by a fixed 32-eps tolerance when the
    activation lies along the near-null direction of an ill-conditioned selected span: the
    float32 pinv recompute of the projection then diverges by ~cond*eps. Scaling the tolerance
    with the measured condition number keeps the genuine decomposition valid.
    """
    separation = 1e-3  # atom1/atom2 near-parallel; their difference direction is e2
    atom0 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # active support and edit source
    atom1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    atom2 = torch.tensor([0.0, 1.0, separation, 0.0])
    atom2 = atom2 / atom2.norm()
    target_atom = torch.tensor([0.0, 0.0, 0.0, 1.0])  # appended by the edit
    dictionary = torch.stack([atom0, atom1, atom2, target_atom])

    # The activation sits in the selected span but concentrates on e2, the ill-conditioned
    # (near-null) direction of the atom1/atom2 pair, so recovering it needs the amplifying inverse.
    activation = atom0 + torch.tensor([0.0, 0.0, 1.0, 0.0])
    decomposition = JSpaceDecomposition(
        support=torch.tensor([0]),
        coordinates=torch.tensor([1.0]),  # NNLS fit of the activation over the active atom e0
        selected_support=torch.tensor([0, 1, 2]),
        reconstruction=atom0.clone(),
        j_space_component=activation.clone(),  # activation lies entirely in the selected span
        non_j_space_component=torch.zeros(4),
    )

    result = solve_coordinate_patch(activation, dictionary, 0, 3, decomposition=decomposition)

    assert torch.isfinite(result.patched).all()
