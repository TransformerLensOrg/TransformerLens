"""LayerNorm1P affine offsets are independent of RMS normalization semantics."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.nemotron.modeling_nemotron import NemotronLayerNorm1P

from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
    norm_uses_weight_offset,
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


@pytest.mark.parametrize("cfg_offset", [True, False])
def test_norm_uses_weight_offset_prefers_mounted_norm(cfg_offset):
    """A mounted norm's own declaration wins over the config; others fall back to it."""
    cfg = SimpleNamespace(uses_rms_norm=True, rmsnorm_uses_offset=cfg_offset)
    for declared in (True, False):
        bridge = NormalizationBridge("ln2", cfg, uses_weight_offset=declared)
        assert norm_uses_weight_offset(bridge, cfg) is declared
    assert norm_uses_weight_offset(torch.nn.Identity(), cfg) is cfg_offset
    assert norm_uses_weight_offset(None, cfg) is cfg_offset


def test_python_fallback_keeps_the_wrapped_norms_eps():
    """A backward hook routes the native path through the python norm, which must keep the
    module's eps (Muse Glimmer's post_norm_eps) rather than the config's rms_norm_eps."""
    component = NemotronLayerNorm1P(16, eps=1e-2)
    bridge = NormalizationBridge(
        "ln1_post",
        SimpleNamespace(uses_rms_norm=False, rmsnorm_uses_offset=True, eps=1e-5),
        use_native_layernorm_autograd=True,
        uses_weight_offset=True,
    )
    bridge.set_original_component(component)
    bridge.hook_scale.add_hook(lambda grad, hook: None, dir="bwd")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16) * 1e-2
    with pytest.warns(UserWarning, match="python-norm path"):
        out = bridge(x)
    torch.testing.assert_close(out, component(x), atol=1e-6, rtol=0)
