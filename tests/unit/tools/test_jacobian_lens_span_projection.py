"""Model-free span invariants for sparse Jacobian-lens projections."""

import pytest
import torch

from tests.unit.tools.conftest import D_MODEL, _ToyBridge
from transformer_lens.tools.analysis import JacobianLens
from transformer_lens.tools.analysis.jacobian_lens_coordinate_patch import (
    solve_coordinate_patch,
)
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    _project_onto_span,
    estimate_occupancy,
    get_sparse_decomposition,
)


def _orthogonal_problem(dim: int, scale: float):
    dictionary = torch.zeros(3, dim)
    dictionary[0, 0] = 1
    dictionary[1, 1] = scale
    dictionary[2, 2] = 1
    target = torch.zeros(dim)
    target[:2] = torch.tensor([2.0, 1.0])
    return target, dictionary


@pytest.mark.parametrize("dim", [3, 4096])
@pytest.mark.parametrize("scale", [1.0, 3000.0])
def test_decomposition_retains_orthogonal_selected_directions(dim: int, scale: float):
    target, dictionary = _orthogonal_problem(dim, scale)
    result = get_sparse_decomposition(target, dictionary, k=2)
    assert result.support.tolist() == [0, 1]
    torch.testing.assert_close(result.reconstruction, target)
    torch.testing.assert_close(result.j_space_component, target)
    torch.testing.assert_close(result.non_j_space_component, torch.zeros_like(target))
    torch.testing.assert_close(dictionary @ result.non_j_space_component, torch.zeros(3))


@pytest.mark.parametrize("dim", [3, 4096])
@pytest.mark.parametrize("scale", [1.0, 3000.0])
def test_occupancy_gains_preserve_nested_spans(dim: int, scale: float):
    target, dictionary = _orthogonal_problem(dim, scale)
    result = estimate_occupancy(target, dictionary, max_atoms=2, num_control_dictionaries=1)
    torch.testing.assert_close(result.marginal_captured_variance, torch.tensor([0.8, 0.2]))


def test_coordinate_reuse_accepts_correct_wide_span_projection():
    target, dictionary = _orthogonal_problem(4096, 3000.0)
    decomposition = get_sparse_decomposition(target, dictionary, k=2)
    # Supply the exact projection independently of the implementation under test.
    decomposition.j_space_component = target.clone()
    decomposition.non_j_space_component = torch.zeros_like(target)
    patch = solve_coordinate_patch(target, dictionary, 0, 2, decomposition=decomposition)
    expected = target.clone()
    expected[0], expected[2] = 0, 2
    torch.testing.assert_close(patch.patched, expected)
    decomposition.j_space_component[0] = 0
    decomposition.non_j_space_component = target - decomposition.j_space_component
    with pytest.raises(ValueError, match="selected span"):
        solve_coordinate_patch(target, dictionary, 0, 2, decomposition=decomposition)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_span_projection_is_invariant_to_atom_rescaling_and_zero_padding(scale: float):
    atoms = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    target = torch.tensor([2.0, 1.0, 3.0])
    atoms[:, 1] *= scale
    expected = torch.tensor([2.0, 1.0, 0.0])
    torch.testing.assert_close(_project_onto_span(atoms, target), expected)
    padded_atoms = torch.nn.functional.pad(atoms, (0, 0, 0, 4093))
    padded_target = torch.nn.functional.pad(target, (0, 4093))
    projected = _project_onto_span(padded_atoms, padded_target)
    torch.testing.assert_close(projected[:3], expected)
    torch.testing.assert_close(projected[3:], torch.zeros(4093))


@pytest.mark.parametrize("separation, retained", [(1e-5, True), (1e-10, False)])
def test_span_projection_has_explicit_near_dependency_policy(separation: float, retained: bool):
    atoms = torch.tensor([[1.0, 1.0], [0.0, separation], [0.0, 0.0]])
    target = torch.tensor([0.0, 1.0, 0.0])
    projected = _project_onto_span(atoms, target)
    expected = target if retained else torch.zeros_like(target)
    torch.testing.assert_close(projected, expected, atol=1e-8, rtol=1e-6)


def test_span_projection_handles_dependent_and_empty_columns():
    atoms = torch.tensor([[1.0, 2.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([2.0, 1.0, 3.0])
    torch.testing.assert_close(_project_onto_span(atoms, target), torch.tensor([2.0, 1.0, 0.0]))
    torch.testing.assert_close(_project_onto_span(atoms[:, :0], target), torch.zeros(3))


def test_lens_variance_profile_uses_the_full_selected_span():
    model = _ToyBridge()
    with torch.no_grad():
        model.embed.weight.zero_()
        model.embed.weight[:, :2] = torch.tensor([2.0, 1.0])
        for block in model.blocks:
            block.linear.weight.zero_()
        model.unembed.weight.zero_()
        model.unembed.weight[:, 2] = 1
        model.unembed.weight[0] = 0
        model.unembed.weight[0, 0] = 1
        model.unembed.weight[1] = 0
        model.unembed.weight[1, 1] = 3e6
    lens = JacobianLens({0: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL)
    result = lens.decompose(model, "test", layer=0, position=-1, k=2)
    torch.testing.assert_close(result.non_j_space_component, torch.zeros(D_MODEL))
    profile = lens.fraction_of_variance(model, ["test"], layers=[0], k=2, skip_first=0)
    assert profile.median[0] == pytest.approx(1.0)
    assert profile.pooled[0] == pytest.approx(1.0)
    torch.testing.assert_close(profile.per_position[0], torch.ones_like(profile.per_position[0]))
