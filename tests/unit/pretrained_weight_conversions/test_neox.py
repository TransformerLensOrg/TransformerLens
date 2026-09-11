"""Tests for NeoX weight conversion.

transformers >= ~5.14 renamed GPTNeoXForCausalLM.embed_out to lm_head, but the
repo's locked 5.13.0 (uv.lock) still exposes embed_out. convert_neox_weights
must resolve the unembedding on either layout.
"""

from types import SimpleNamespace

import torch

from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.neox import convert_neox_weights


def _make_cfg(n_layers=1, d_model=8, n_heads=2, d_mlp=16, d_vocab=32):
    return HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_ctx=32,
        d_vocab=d_vocab,
        act_fn="gelu",
        normalization_type="LN",
        dtype=torch.float32,
        device="cpu",
    )


def _make_layer(cfg):
    d, n_heads, d_mlp = cfg.d_model, cfg.n_heads, cfg.d_mlp
    attention = SimpleNamespace(
        query_key_value=SimpleNamespace(weight=torch.randn(3 * d, d), bias=torch.randn(3 * d)),
        dense=SimpleNamespace(weight=torch.randn(d, d), bias=torch.randn(d)),
    )
    mlp = SimpleNamespace(
        dense_h_to_4h=SimpleNamespace(weight=torch.randn(d_mlp, d), bias=torch.randn(d_mlp)),
        dense_4h_to_h=SimpleNamespace(weight=torch.randn(d, d_mlp), bias=torch.randn(d)),
    )
    return SimpleNamespace(
        input_layernorm=SimpleNamespace(weight=torch.randn(d), bias=torch.randn(d)),
        post_attention_layernorm=SimpleNamespace(weight=torch.randn(d), bias=torch.randn(d)),
        attention=attention,
        mlp=mlp,
    )


def _make_model(cfg, unembed_attr: str):
    """Build a minimal fake NeoX model exposing the unembedding under the given attr."""
    gpt_neox = SimpleNamespace(
        embed_in=SimpleNamespace(weight=torch.randn(cfg.d_vocab, cfg.d_model)),
        layers=[_make_layer(cfg) for _ in range(cfg.n_layers)],
        final_layer_norm=SimpleNamespace(
            weight=torch.randn(cfg.d_model), bias=torch.randn(cfg.d_model)
        ),
    )
    unembed = SimpleNamespace(weight=torch.randn(cfg.d_vocab, cfg.d_model))
    kwargs = {"gpt_neox": gpt_neox, unembed_attr: unembed}
    return SimpleNamespace(**kwargs), unembed


class TestNeoxUnembedResolution:
    """convert_neox_weights must resolve the unembed weight on both HF layouts."""

    def test_resolves_lm_head_layout(self):
        """transformers >= ~5.14 layout: only lm_head is present."""
        cfg = _make_cfg()
        model, unembed = _make_model(cfg, "lm_head")
        assert not hasattr(model, "embed_out")

        state_dict = convert_neox_weights(model, cfg)

        assert torch.equal(state_dict["unembed.W_U"], unembed.weight.T)

    def test_resolves_embed_out_layout(self):
        """transformers <= 5.13.0 layout (the repo's locked version): only embed_out is present."""
        cfg = _make_cfg()
        model, unembed = _make_model(cfg, "embed_out")
        assert not hasattr(model, "lm_head")

        state_dict = convert_neox_weights(model, cfg)

        assert torch.equal(state_dict["unembed.W_U"], unembed.weight.T)

    def test_lm_head_preferred_when_both_present(self):
        cfg = _make_cfg()
        model, lm_head = _make_model(cfg, "lm_head")
        model.embed_out = SimpleNamespace(weight=torch.randn(cfg.d_vocab, cfg.d_model))

        state_dict = convert_neox_weights(model, cfg)

        assert torch.equal(state_dict["unembed.W_U"], lm_head.weight.T)
