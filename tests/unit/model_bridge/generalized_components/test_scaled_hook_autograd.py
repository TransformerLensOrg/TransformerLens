"""Equal-valued interventions must retain their own autograd graph."""

import pytest
import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


@pytest.mark.parametrize("scale", [0.22, 1.0, 2.0, -3.0])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_zero_initialized_intervention_matches_direct_graph(scale, frozen, in_place):
    component = GeneralizedComponent("scaled")
    point = HookPoint()
    output = torch.tensor([0.3, -1.7, 2.1], requires_grad=not frozen) * 1
    delta = torch.nn.Parameter(torch.zeros_like(output))

    def intervene(value, hook):
        return value.add_(delta) if in_place else value + delta

    point.add_hook(intervene)
    result = component._hook_scaled_output(output, point, scale)
    torch.testing.assert_close(result, output, rtol=0, atol=0)
    weights = torch.tensor([1.0, -2.0, 0.5])
    assert result.requires_grad
    actual = torch.autograd.grad((result * weights).sum(), delta, allow_unused=True)[0]
    assert actual is not None
    torch.testing.assert_close(actual, weights / scale)


@pytest.mark.parametrize("scale", [0.22, 2.0, -3.0])
@pytest.mark.parametrize("kind", ["straight_through", "detach"])
def test_equal_values_preserve_changed_input_derivative(scale, kind):
    component = GeneralizedComponent("scaled")
    point = HookPoint()
    output = torch.tensor([0.3, -1.7, 2.1], requires_grad=True)

    def intervene(value, hook):
        if kind == "detach":
            return value.detach()
        return value.detach() + 3 * (value - value.detach())

    point.add_hook(intervene)
    result = component._hook_scaled_output(output, point, scale)
    torch.testing.assert_close(result, output, rtol=0, atol=0)
    if kind == "detach":
        assert not result.requires_grad
    else:
        result.sum().backward()
        torch.testing.assert_close(output.grad, torch.full_like(output, 3.0))


@pytest.mark.parametrize("scale", [0.0, 0.22, 2.0, -3.0])
@pytest.mark.parametrize("returns_clone", [False, True])
def test_observation_preserves_exact_values(scale, returns_clone):
    component = GeneralizedComponent("scaled")
    point = HookPoint()
    output = torch.randn(32, requires_grad=True)
    point.add_hook(lambda value, hook: value.clone() if returns_clone else None)
    result = component._hook_scaled_output(output, point, scale)
    torch.testing.assert_close(result, output, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(output.grad, torch.ones_like(output))


def test_unhooked_scaled_output_keeps_original_tensor():
    component = GeneralizedComponent("scaled")
    output = torch.randn(8, requires_grad=True)
    assert component._hook_scaled_output(output, HookPoint(), 0.22) is output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_equal_value_edit_preserves_dtype_and_gradients(dtype):
    component = GeneralizedComponent("scaled")
    point = HookPoint()
    output = torch.tensor([0.3, -1.7, 2.1], dtype=dtype, requires_grad=True)
    delta = torch.nn.Parameter(torch.zeros_like(output))
    point.add_hook(lambda value, hook: value + delta)
    result = component._hook_scaled_output(output, point, 0.22)
    torch.testing.assert_close(result, output, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(delta.grad, torch.ones_like(delta) / 0.22)


def test_overflowed_scaled_observation_keeps_finite_raw_value_and_graph():
    component = GeneralizedComponent("scaled")
    point = HookPoint()
    output = torch.tensor([40000.0], dtype=torch.float16, requires_grad=True)
    delta = torch.nn.Parameter(torch.zeros_like(output))
    point.add_hook(lambda value, hook: value + delta)
    result = component._hook_scaled_output(output, point, 2.0)
    torch.testing.assert_close(result, output, rtol=0, atol=0)
    result.sum().backward()
    torch.testing.assert_close(output.grad, torch.ones_like(output))
    torch.testing.assert_close(delta.grad, torch.full_like(delta, 0.5))


def test_equal_value_edit_preserves_second_derivative():
    component = GeneralizedComponent("scaled")
    point = HookPoint()
    output = torch.ones(2, dtype=torch.float64)
    delta = torch.nn.Parameter(torch.zeros_like(output))
    point.add_hook(lambda value, hook: value + delta.square())
    result = component._hook_scaled_output(output, point, 2.0)
    first = torch.autograd.grad(result.sum(), delta, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), delta)[0]
    torch.testing.assert_close(first, torch.zeros_like(delta))
    torch.testing.assert_close(second, torch.ones_like(delta))


def test_equal_value_edit_supports_functional_differentiation():
    component = GeneralizedComponent("scaled")

    def objective(delta):
        point = HookPoint()
        point.add_hook(lambda value, hook: value + delta)
        return component._hook_scaled_output(torch.ones_like(delta), point, 2.0).sum()

    delta = torch.zeros(3)
    torch.testing.assert_close(torch.func.jacrev(objective)(delta), torch.full_like(delta, 0.5))
    primal, tangent = torch.func.jvp(objective, (delta,), (torch.ones_like(delta),))
    torch.testing.assert_close(primal, torch.tensor(3.0))
    torch.testing.assert_close(tangent, torch.tensor(1.5))


def test_equal_value_edit_supports_forward_mode_jacobian():
    component = GeneralizedComponent("scaled")

    def outputs(delta):
        point = HookPoint()
        point.add_hook(lambda value, hook: value + delta)
        return component._hook_scaled_output(torch.ones_like(delta), point, 2.0)

    delta = torch.zeros(3)
    # jacfwd vmaps the Function over basis tangents; jacrev of a scalar never does.
    torch.testing.assert_close(torch.func.jacfwd(outputs)(delta), 0.5 * torch.eye(3))
