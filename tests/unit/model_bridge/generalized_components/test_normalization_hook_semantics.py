"""Hook semantics on NormalizationBridge's native-autograd path.

Regression coverage for #1526: with ``use_native_layernorm_autograd=True`` the
bridge fired ``hook_scale`` / ``hook_normalized`` on detached observation values
and discarded their return values — forward edits silently no-oped and backward
hooks never fired, while the default python-norm path honored both.
"""

import warnings

import pytest
import torch
import torch.nn as nn

# Warnings are matched by message substring, not imported constants, so this file
# collects (and fails red) on unfixed code where the constants don't exist yet.
from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)


class _Cfg:
    def __init__(
        self,
        uses_rms_norm: bool = False,
        eps: float = 1e-5,
        rmsnorm_uses_offset: bool = False,
    ):
        self.uses_rms_norm = uses_rms_norm
        self.eps = eps
        self.rmsnorm_uses_offset = rmsnorm_uses_offset


class _TinyRMSNorm(nn.Module):
    """Minimal RMSNorm mirroring LlamaRMSNorm's forward."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.variance_epsilon)


class _TinyGemmaRMSNorm(nn.Module):
    """Minimal Gemma-style RMSNorm: weight is stored as an offset from 1."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.full((d,), 0.5))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.variance_epsilon)
        return x_normed * (1.0 + self.weight)


def _layernorm(d: int) -> nn.LayerNorm:
    layer = nn.LayerNorm(d, eps=1e-5)
    nn.init.normal_(layer.weight, std=0.1)
    nn.init.normal_(layer.bias, std=0.1)
    layer.eval()
    return layer


def _make_bridge(
    native: bool, rms: bool = False, d: int = 16, offset: bool = False
) -> NormalizationBridge:
    layer: nn.Module
    if offset:
        layer = _TinyGemmaRMSNorm(d)
    elif rms:
        layer = _TinyRMSNorm(d)
    else:
        layer = _layernorm(d)
    bridge = NormalizationBridge(
        name="ln",
        config=_Cfg(uses_rms_norm=rms or offset, rmsnorm_uses_offset=offset),
        use_native_layernorm_autograd=native,
    )
    bridge.set_original_component(layer)
    return bridge


def _zero_hook(tensor, hook=None):
    return torch.zeros_like(tensor)


@pytest.mark.parametrize("rms", [False, True], ids=["layernorm", "rmsnorm"])
def test_native_path_normalized_edit_propagates(rms):
    """Editing hook_normalized must change the output on the native-autograd path."""
    bridge = _make_bridge(native=True, rms=rms)
    x = torch.randn(2, 5, 16)
    baseline = bridge(x)
    bridge.hook_normalized.add_hook(_zero_hook)
    with pytest.warns(UserWarning, match="reconstructed from the hooked values"):
        patched = bridge(x)
    bridge.hook_normalized.remove_hooks()
    assert not torch.allclose(baseline, patched)


def test_native_path_scale_edit_propagates():
    """Editing hook_scale must change the output on the native-autograd path."""
    bridge = _make_bridge(native=True)
    x = torch.randn(2, 5, 16)
    baseline = bridge(x)
    bridge.hook_scale.add_hook(lambda t, hook=None: t * 2.0)
    with pytest.warns(UserWarning, match="reconstructed from the hooked values"):
        patched = bridge(x)
    bridge.hook_scale.remove_hooks()
    assert not torch.allclose(baseline, patched)


def test_native_path_bwd_hook_fires():
    """Backward hooks on hook_normalized must fire on the native-autograd path."""
    bridge = _make_bridge(native=True)
    x = torch.randn(2, 5, 16, requires_grad=True)
    fired = {}
    bridge.hook_normalized.add_hook(
        lambda grad, hook=None: fired.setdefault("bwd", True) and grad, dir="bwd"
    )
    with pytest.warns(UserWarning, match="Backward hooks"):
        out = bridge(x)
    out.sum().backward()
    bridge.hook_normalized.remove_hooks()
    assert fired.get("bwd", False)


def test_native_path_edited_output_is_grad_connected():
    """An edited forward must stay differentiable back to the input."""
    bridge = _make_bridge(native=True)
    x = torch.randn(2, 5, 16, requires_grad=True)
    bridge.hook_scale.add_hook(lambda t, hook=None: t * 2.0)
    with pytest.warns(UserWarning):
        out = bridge(x)
    bridge.hook_scale.remove_hooks()
    out.sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_native_path_observation_hooks_keep_native_numerics():
    """run_with_cache-style hooks (return None) must not change output numerics.

    Guards the fallback dispatch: routing on "any hook attached" would silently
    switch caching runs from HF's native forward to the python-norm path.
    """
    bridge = _make_bridge(native=True)
    x = torch.randn(2, 5, 16)
    baseline = bridge(x)
    cache = {}

    def observe(tensor, hook=None):
        cache["normalized"] = tensor.detach()
        return None

    bridge.hook_normalized.add_hook(observe)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any fallback warning fails the test
        observed = bridge(x)
    bridge.hook_normalized.remove_hooks()
    assert torch.equal(baseline, observed)
    assert "normalized" in cache


def test_native_path_no_hooks_matches_original_component():
    """Without hooks, the native path is exactly the wrapped module's output."""
    bridge = _make_bridge(native=True)
    x = torch.randn(2, 5, 16)
    assert torch.equal(bridge(x), bridge.original_component(x))


def test_native_path_edit_fallback_applies_rmsnorm_offset():
    """Gemma-family RMSNorm multiplies by (1 + weight); the edit fallback must too."""
    bridge = _make_bridge(native=True, offset=True)
    x = torch.randn(2, 5, 16)
    bridge.hook_normalized.add_hook(lambda t, hook=None: torch.ones_like(t))
    with pytest.warns(UserWarning, match="reconstructed from the hooked values"):
        patched = bridge(x)
    bridge.hook_normalized.remove_hooks()
    expected = torch.ones(2, 5, 16) * (1.0 + bridge.original_component.weight)
    assert torch.allclose(patched, expected)


def test_native_path_bwd_fallback_applies_rmsnorm_offset():
    """The python-norm fallback for backward hooks must match Gemma-style output."""
    bridge = _make_bridge(native=True, offset=True)
    x = torch.randn(2, 5, 16, requires_grad=True)
    bridge.hook_normalized.add_hook(lambda grad, hook=None: grad, dir="bwd")
    with pytest.warns(UserWarning, match="Backward hooks"):
        out = bridge(x)
    bridge.hook_normalized.remove_hooks()
    assert torch.allclose(out, bridge.original_component(x), atol=1e-6)


def test_default_path_edit_still_propagates():
    """Control: the python-norm path's editing semantics are unchanged."""
    bridge = _make_bridge(native=False)
    x = torch.randn(2, 5, 16)
    baseline = bridge(x)
    bridge.hook_normalized.add_hook(_zero_hook)
    patched = bridge(x)
    bridge.hook_normalized.remove_hooks()
    assert not torch.allclose(baseline, patched)


def test_default_path_emits_no_fallback_warnings():
    """Control: python-norm path edits never warn — warnings are native-path only."""
    bridge = _make_bridge(native=False)
    x = torch.randn(2, 5, 16)
    bridge.hook_normalized.add_hook(_zero_hook)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bridge(x)
    bridge.hook_normalized.remove_hooks()
