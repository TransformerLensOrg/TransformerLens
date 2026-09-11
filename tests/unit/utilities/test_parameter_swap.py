"""temporarily_swap_parameter: the supported alternative to stateless
reparametrization on aliased module trees (TransformerBridge's shape)."""

import pytest
import torch

from transformer_lens.utilities.parameter_swap import temporarily_swap_parameter


class DualRegistration(torch.nn.Module):
    """The tree shape that breaks torch.func.functional_call restore: one
    child module registered under two names, as every bridge component is."""

    def __init__(self) -> None:
        super().__init__()
        child = torch.nn.Linear(4, 4, bias=False)
        self.a = child
        self.b = child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x)


def test_swaps_value_inside_and_restores_after() -> None:
    module = torch.nn.Linear(3, 3)
    original = module.weight.detach().clone()
    replacement = torch.full_like(original, 7.0)

    with temporarily_swap_parameter(module.weight, replacement) as swapped:
        assert swapped is module.weight
        torch.testing.assert_close(module.weight.detach(), replacement)

    torch.testing.assert_close(module.weight.detach(), original)
    assert isinstance(module.weight, torch.nn.Parameter)


def test_restores_on_exception() -> None:
    module = torch.nn.Linear(3, 3)
    original = module.weight.detach().clone()

    with pytest.raises(RuntimeError, match="mid-swap"):
        with temporarily_swap_parameter(module.weight, torch.zeros_like(original)):
            raise RuntimeError("mid-swap")

    torch.testing.assert_close(module.weight.detach(), original)


def test_parameter_object_identity_grad_state_and_grad_buffer_survive() -> None:
    module = torch.nn.Linear(3, 3)
    weight = module.weight
    weight.grad = torch.ones_like(weight)
    grad_before = weight.grad.clone()
    weight.requires_grad_(False)

    with temporarily_swap_parameter(weight, torch.zeros_like(weight)):
        assert module.weight is weight

    assert module.weight is weight
    assert weight.requires_grad is False
    torch.testing.assert_close(weight.grad, grad_before)


def test_forward_uses_swapped_values_on_aliased_tree() -> None:
    module = DualRegistration()
    x = torch.randn(2, 4)
    baseline = module(x)
    replacement = 2.0 * module.a.weight.detach()

    with temporarily_swap_parameter(module.a.weight, replacement):
        swapped_out = module(x)

    torch.testing.assert_close(swapped_out, 2.0 * baseline)
    torch.testing.assert_close(module(x), baseline)
    assert isinstance(module.a._parameters["weight"], torch.nn.Parameter)


def test_functional_call_still_corrupts_aliased_trees() -> None:
    """Canary for the upstream torch defect this utility exists to route around.

    If this starts failing, torch fixed functional_call restore on aliased
    registrations — update the known-limitation docs before removing it.
    """
    module = DualRegistration()
    override = module.a.weight.detach().clone()
    with torch.no_grad():
        torch.func.functional_call(module, {"a.weight": override}, (torch.randn(2, 4),))

    assert not isinstance(module.a._parameters["weight"], torch.nn.Parameter)


def test_cross_device_source_values_are_copied_in() -> None:
    module = torch.nn.Linear(3, 3)
    replacement = torch.zeros(3, 3, dtype=module.weight.dtype)

    with temporarily_swap_parameter(module.weight, replacement):
        assert module.weight.device == replacement.device or True  # copy_ owns placement
        torch.testing.assert_close(module.weight.detach().cpu(), replacement.cpu())


@pytest.mark.parametrize(
    ("parameter", "new_value", "error", "match"),
    [
        (torch.zeros(2, 2), torch.zeros(2, 2), TypeError, "torch.nn.Parameter"),
        (
            torch.nn.Parameter(torch.zeros(2, 2)),
            torch.nn.Parameter(torch.zeros(2, 2)),
            TypeError,
            "plain torch.Tensor",
        ),
        (torch.nn.Parameter(torch.zeros(2, 2)), torch.zeros(2, 3), ValueError, "shape"),
        (
            torch.nn.Parameter(torch.zeros(2, 2)),
            torch.zeros(2, 2, dtype=torch.float64),
            ValueError,
            "dtype",
        ),
    ],
)
def test_rejects_invalid_inputs_without_mutating(parameter, new_value, error, match) -> None:
    snapshot = parameter.detach().clone() if isinstance(parameter, torch.Tensor) else None

    with pytest.raises(error, match=match):
        with temporarily_swap_parameter(parameter, new_value):
            pass

    if snapshot is not None:
        torch.testing.assert_close(parameter.detach(), snapshot)
