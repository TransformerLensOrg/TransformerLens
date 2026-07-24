"""RMSNorm-vs-LayerNorm dispatch in NormalizationBridge.

Regression coverage for Gemma 3 multimodal: SigLIP's post_layernorm was wrapped
under the LM's uses_rms_norm=True config, silently dropping mean-centering and
bias and producing gibberish completions.
"""

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)


class _Cfg:
    def __init__(self, uses_rms_norm: bool, eps: float = 1e-5):
        self.uses_rms_norm = uses_rms_norm
        self.eps = eps


def _make_bridge(layer: nn.Module, cfg: _Cfg, **kwargs) -> NormalizationBridge:
    bridge = NormalizationBridge(name="ln", config=cfg, **kwargs)
    bridge.set_original_component(layer)
    return bridge


def _layernorm(d: int) -> nn.LayerNorm:
    layer = nn.LayerNorm(d, eps=1e-5)
    nn.init.normal_(layer.weight, std=0.1)
    nn.init.normal_(layer.bias, std=0.1)
    layer.eval()
    return layer


def test_override_forces_layernorm_when_config_says_rmsnorm():
    d = 16
    layer = _layernorm(d)
    bridge = _make_bridge(layer, _Cfg(uses_rms_norm=True), uses_rms_norm=False)
    x = torch.randn(2, 5, d)
    torch.testing.assert_close(bridge(x), layer(x), rtol=1e-5, atol=1e-5)


def test_introspects_layernorm_when_config_says_rmsnorm():
    d = 16
    layer = _layernorm(d)
    bridge = _make_bridge(layer, _Cfg(uses_rms_norm=True))
    assert bridge.uses_rms_norm is False
    x = torch.randn(2, 5, d)
    torch.testing.assert_close(bridge(x), layer(x), rtol=1e-5, atol=1e-5)


def test_introspects_rmsnorm_class_by_name():
    class FakeRMSNorm(nn.Module):
        def __init__(self, d, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.variance_epsilon = eps

        def forward(self, x):
            rms = (x.float().pow(2).mean(-1, keepdim=True) + self.variance_epsilon).sqrt()
            return (x.float() / rms * self.weight).to(x.dtype)

    rms = FakeRMSNorm(16)
    nn.init.normal_(rms.weight, std=0.1)
    rms.eval()
    bridge = _make_bridge(rms, _Cfg(uses_rms_norm=False))
    assert bridge.uses_rms_norm is True


def test_falls_back_to_config_when_component_unset():
    bridge = NormalizationBridge(name="ln", config=_Cfg(uses_rms_norm=True))
    assert bridge.original_component is None
    assert bridge.uses_rms_norm is True

    bridge2 = NormalizationBridge(name="ln", config=_Cfg(uses_rms_norm=False))
    assert bridge2.uses_rms_norm is False


def test_override_takes_precedence_over_config():
    # Override forces the *math path*, beating both config and introspection.
    d = 16
    layer = _layernorm(d)  # a real nn.LayerNorm: introspection would say uses_rms_norm=False
    x = torch.randn(2, 5, d)

    # uses_rms_norm=False override over config=True -> LayerNorm math (mean-centred + bias).
    ln_bridge = _make_bridge(layer, _Cfg(uses_rms_norm=True), uses_rms_norm=False)
    torch.testing.assert_close(ln_bridge(x), layer(x), rtol=1e-5, atol=1e-5)

    # uses_rms_norm=True override over config=False AND over the LayerNorm introspection
    # -> RMSNorm math: no mean-centring, no bias.
    rms_bridge = _make_bridge(layer, _Cfg(uses_rms_norm=False), uses_rms_norm=True)
    rms = (x.pow(2).mean(-1, keepdim=True) + layer.eps).sqrt()
    expected_rms = x / rms * layer.weight
    out = rms_bridge(x)
    torch.testing.assert_close(out, expected_rms, rtol=1e-5, atol=1e-5)
    # And it must diverge from the LayerNorm output (otherwise the override did nothing).
    assert not torch.allclose(out, layer(x), rtol=1e-3, atol=1e-3)


def test_siglip_post_layernorm_resolves_to_layernorm_under_rmsnorm_config():
    from transformer_lens.model_bridge.generalized_components.siglip_vision_encoder import (
        SiglipVisionEncoderBridge,
    )

    encoder = SiglipVisionEncoderBridge(name="vision_tower", config=_Cfg(uses_rms_norm=True))
    post_ln = encoder.submodules["post_layernorm"]
    post_ln.set_original_component(nn.LayerNorm(8))
    assert post_ln.uses_rms_norm is False


def test_clip_layernorms_resolve_to_layernorm_under_rmsnorm_config():
    from transformer_lens.model_bridge.generalized_components.clip_vision_encoder import (
        CLIPVisionEncoderBridge,
    )

    encoder = CLIPVisionEncoderBridge(name="vision_tower", config=_Cfg(uses_rms_norm=True))
    pre = encoder.submodules["pre_layernorm"]
    post = encoder.submodules["post_layernorm"]
    pre.set_original_component(nn.LayerNorm(8))
    post.set_original_component(nn.LayerNorm(8))
    assert pre.uses_rms_norm is False
    assert post.uses_rms_norm is False


def test_native_autograd_path_also_respects_override():
    d = 16
    layer = _layernorm(d)
    bridge = _make_bridge(
        layer, _Cfg(uses_rms_norm=True), uses_rms_norm=False, use_native_layernorm_autograd=True
    )
    x = torch.randn(2, 5, d)
    torch.testing.assert_close(bridge(x), layer(x), rtol=1e-5, atol=1e-5)


def test_layernorm_with_none_bias_matches_torch():
    # MPT's MptBlock and Cohere set norm.bias = None for Hub-weight compat; the bridge must
    # treat the missing bias as zero (as F.layer_norm does), not crash or NaN.
    d = 16
    layer = nn.LayerNorm(d, eps=1e-5)
    nn.init.normal_(layer.weight, std=0.1)
    layer.bias = None
    layer.eval()
    bridge = _make_bridge(layer, _Cfg(uses_rms_norm=False))
    x = torch.randn(2, 4, d)
    out = bridge(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, layer(x), rtol=1e-5, atol=1e-5)
