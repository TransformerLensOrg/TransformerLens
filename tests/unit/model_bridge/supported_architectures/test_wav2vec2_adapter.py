"""Wav2Vec2 adapter: only the deltas vs the inherited HuBERT structure.

Wav2Vec2Model's module tree is identical to HubertModel's, so the structural
mapping is covered by test_hubert_adapter.py; what needs proof here is the
factory routing, the ForCTC nesting detection, and that a real forward matches
HF end to end.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.model_bridge.supported_architectures.wav2vec2 import (
    Wav2Vec2ArchitectureAdapter,
)


def _tiny_config() -> Wav2Vec2Config:
    return Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(32,) * 7,
        vocab_size=40,
    )


def _tiny_bridge(model, arch):
    cfg = model.config
    return build_bridge_from_module(
        model.eval(), arch, hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    )


@pytest.mark.parametrize(
    "architecture", ["Wav2Vec2Model", "Wav2Vec2ForCTC", "Wav2Vec2ForPreTraining"]
)
def test_factory_routes_wav2vec2_architectures(architecture):
    from transformer_lens.config import TransformerBridgeConfig

    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=2,
        n_ctx=512,
        d_vocab=40,
        d_mlp=64,
        architecture=architecture,
    )
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    assert isinstance(adapter, Wav2Vec2ArchitectureAdapter)


def test_forctc_nesting_is_detected():
    """ForCTC nests the encoder under 'wav2vec2.' and adds the CTC head."""
    torch.manual_seed(0)
    model = Wav2Vec2ForCTC(_tiny_config())
    bridge = _tiny_bridge(model, "Wav2Vec2ForCTC")
    assert bridge.blocks[0].attn.q.original_component is not None
    assert hasattr(bridge, "unembed")


def test_tiny_forward_matches_hf():
    torch.manual_seed(0)
    model = Wav2Vec2Model(_tiny_config())
    bridge = _tiny_bridge(model, "Wav2Vec2Model")
    wave = torch.randn(1, 4000) * 0.1
    with torch.no_grad():
        ref = model(wave).last_hidden_state
        out = bridge(wave)
    out_t = out if isinstance(out, torch.Tensor) else out["last_hidden_state"]
    torch.testing.assert_close(out_t, ref, atol=1e-5, rtol=1e-5)


def test_hooks_fire_on_wav2vec2():
    torch.manual_seed(0)
    bridge = _tiny_bridge(Wav2Vec2Model(_tiny_config()), "Wav2Vec2Model")
    wave = torch.randn(1, 4000) * 0.1
    _, cache = bridge.run_with_cache(wave, names_filter=lambda n: n.endswith("hook_out"))
    assert any(k.startswith("blocks.0.attn") for k in cache)
    assert "feat_proj.hook_out" in cache


def test_pretraining_class_is_refused_with_a_pointer():
    """Wrapping Wav2Vec2ForPreTraining directly is unsupported (its forward
    returns quantizer states, not hidden states) — refuse with the alternative,
    never die inside component setup. Checkpoint boots still work: the arch
    string routes to AutoModel -> Wav2Vec2Model (integration coverage)."""
    from transformers import Wav2Vec2ForPreTraining

    torch.manual_seed(0)
    model = Wav2Vec2ForPreTraining(_tiny_config())
    with pytest.raises(NotImplementedError, match="without model_class"):
        _tiny_bridge(model, "Wav2Vec2ForPreTraining")


def test_stable_layer_norm_frame_entry_matches_hf():
    """Stable-LN encoders apply encoder.layer_norm AFTER the blocks; frame entry
    must mirror that order, not the post-LN hardcoding."""
    from transformers import Wav2Vec2Model

    torch.manual_seed(0)
    cfg = _tiny_config()
    cfg.do_stable_layer_norm = True
    model = Wav2Vec2Model(cfg).eval()
    assert type(model.encoder).__name__.endswith("StableLayerNorm")
    bridge = _tiny_bridge(model, "Wav2Vec2Model")
    wave = torch.randn(1, 4000) * 0.1
    with torch.no_grad():
        ref = model(wave).last_hidden_state
        feats = model.feature_extractor(wave).transpose(1, 2)
        proj = model.feature_projection(feats)
        frames = proj[0] if isinstance(proj, tuple) else proj
        out = bridge.encoder_output(frames)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
