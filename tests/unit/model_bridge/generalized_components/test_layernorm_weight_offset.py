"""LayerNorm1P affine offsets are independent of RMS normalization semantics."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.nemotron.modeling_nemotron import NemotronLayerNorm1P

from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)


def _make_norm(native=True):
    component = NemotronLayerNorm1P(16, eps=1e-5)
    with torch.no_grad():
        component.weight.copy_(torch.linspace(-0.3, 0.6, 16))
        component.bias.copy_(torch.linspace(-0.1, 0.2, 16))
    bridge = NormalizationBridge(
        "ln1",
        SimpleNamespace(uses_rms_norm=False, rmsnorm_uses_offset=True, eps=1e-5),
        use_native_layernorm_autograd=native,
        uses_weight_offset=True,
    )
    bridge.set_original_component(component)
    assert not bridge.uses_rms_norm
    assert bridge.uses_weight_offset
    return bridge, component


@pytest.mark.parametrize("hook_name", ["hook_scale", "hook_normalized"])
@pytest.mark.parametrize("direction", ["fwd", "bwd"])
def test_layernorm1p_identity_hooks_preserve_outputs_and_gradients(hook_name, direction):
    bridge, component = _make_norm()
    x = torch.randn(2, 4, 16, requires_grad=True)
    probe = torch.randn_like(x)
    parameters = (x, component.weight, component.bias)
    expected = component(x)
    expected_grads = torch.autograd.grad(expected, parameters, probe)
    fired = []

    def identity(value, hook):
        fired.append(True)
        return value.clone()

    hook_point = getattr(bridge, hook_name)
    hook_point.add_hook(identity, dir=direction)
    message = "Backward hooks" if direction == "bwd" else "reconstructed from the hooked values"
    with pytest.warns(UserWarning, match=message):
        actual = bridge(x)
    actual_grads = torch.autograd.grad(actual, parameters, probe)
    hook_point.remove_hooks()
    assert fired
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_python_layernorm1p_preserves_offset():
    bridge, component = _make_norm(native=False)
    x = torch.randn(2, 4, 16)
    torch.testing.assert_close(bridge(x), component(x), atol=1e-6, rtol=1e-5)
