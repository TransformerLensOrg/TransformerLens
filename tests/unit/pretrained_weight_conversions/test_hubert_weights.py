"""Tests for HuBERT weight conversion."""

from unittest import mock

import pytest
import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.hubert import convert_hubert_weights


def make_cfg():
    return HookedTransformerConfig(
        n_layers=1,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        n_ctx=32,
        d_vocab=100,
        act_fn="gelu",
        normalization_type="LN",
        dtype=torch.float32,
    )


def make_mock_hubert(cfg, has_biases=True):
    """Build a minimal mock HF HuBERT model."""
    d = cfg.d_model
    d_mlp = cfg.d_mlp
    n_heads = cfg.n_heads
    d_head = cfg.d_head

    model = mock.Mock()

    layer = mock.Mock()

    # Attention projections
    layer.attention.q_proj.weight = torch.randn(n_heads * d_head, d)
    layer.attention.k_proj.weight = torch.randn(n_heads * d_head, d)
    layer.attention.v_proj.weight = torch.randn(n_heads * d_head, d)
    layer.attention.out_proj.weight = torch.randn(d, n_heads * d_head)

    if has_biases:
        layer.attention.q_proj.bias = torch.randn(n_heads * d_head)
        layer.attention.k_proj.bias = torch.randn(n_heads * d_head)
        layer.attention.v_proj.bias = torch.randn(n_heads * d_head)
        layer.attention.out_proj.bias = torch.randn(d)
    else:
        layer.attention.q_proj.bias = None
        layer.attention.k_proj.bias = None
        layer.attention.v_proj.bias = None
        layer.attention.out_proj.bias = None

    # Layer norms
    layer.layer_norm.weight = torch.randn(d)
    layer.layer_norm.bias = torch.randn(d)
    layer.final_layer_norm.weight = torch.randn(d)
    layer.final_layer_norm.bias = torch.randn(d)

    # Feed-forward
    layer.feed_forward.intermediate_dense.weight = torch.randn(d_mlp, d)
    layer.feed_forward.intermediate_dense.bias = torch.randn(d_mlp)
    layer.feed_forward.output_dense.weight = torch.randn(d, d_mlp)
    layer.feed_forward.output_dense.bias = torch.randn(d)

    # Remove alternate attribute names so mock doesn't auto-create them
    layer.self_attn = None

    model.encoder.layers = [layer]
    model.encoder.layer_norm.weight = torch.randn(d)
    model.encoder.layer_norm.bias = torch.randn(d)

    return model


class TestHubertWeightConversion:
    def test_attention_weight_shapes(self):
        cfg = make_cfg()
        model = make_mock_hubert(cfg)
        sd = convert_hubert_weights(model, cfg)

        assert sd["blocks.0.attn.W_Q"].shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
        assert sd["blocks.0.attn.W_K"].shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
        assert sd["blocks.0.attn.W_V"].shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
        assert sd["blocks.0.attn.W_O"].shape == (cfg.n_heads, cfg.d_head, cfg.d_model)

    def test_attention_bias_shapes(self):
        cfg = make_cfg()
        model = make_mock_hubert(cfg, has_biases=True)
        sd = convert_hubert_weights(model, cfg)

        assert sd["blocks.0.attn.b_Q"].shape == (cfg.n_heads, cfg.d_head)
        assert sd["blocks.0.attn.b_K"].shape == (cfg.n_heads, cfg.d_head)
        assert sd["blocks.0.attn.b_V"].shape == (cfg.n_heads, cfg.d_head)
        assert sd["blocks.0.attn.b_O"].shape == (cfg.d_model,)

    def test_no_bias_omits_bias_keys(self):
        cfg = make_cfg()
        model = make_mock_hubert(cfg, has_biases=False)
        sd = convert_hubert_weights(model, cfg)

        assert "blocks.0.attn.b_Q" not in sd
        assert "blocks.0.attn.b_K" not in sd
        assert "blocks.0.attn.b_V" not in sd
        assert "blocks.0.attn.b_O" not in sd

    def test_layer_norm_extraction(self):
        cfg = make_cfg()
        model = make_mock_hubert(cfg)
        sd = convert_hubert_weights(model, cfg)

        assert "blocks.0.ln1.w" in sd
        assert "blocks.0.ln1.b" in sd
        assert "blocks.0.ln2.w" in sd
        assert "blocks.0.ln2.b" in sd
        assert sd["blocks.0.ln1.w"].shape == (cfg.d_model,)

    def test_ffn_weight_shapes(self):
        """FFN weights should be transposed to TL convention."""
        cfg = make_cfg()
        model = make_mock_hubert(cfg)
        sd = convert_hubert_weights(model, cfg)

        # W_in: (d_model, d_mlp) — transposed from HF's (d_mlp, d_model)
        assert sd["blocks.0.mlp.W_in"].shape == (cfg.d_model, cfg.d_mlp)
        # W_out: (d_mlp, d_model) — transposed from HF's (d_model, d_mlp)
        assert sd["blocks.0.mlp.W_out"].shape == (cfg.d_mlp, cfg.d_model)
        assert sd["blocks.0.mlp.b_in"].shape == (cfg.d_mlp,)
        assert sd["blocks.0.mlp.b_out"].shape == (cfg.d_model,)

    def test_final_layer_norm(self):
        cfg = make_cfg()
        model = make_mock_hubert(cfg)
        sd = convert_hubert_weights(model, cfg)

        assert "ln_final.w" in sd
        assert "ln_final.b" in sd
        assert sd["ln_final.w"].shape == (cfg.d_model,)

    def test_no_embedding_keys(self):
        """HuBERT is encoder-only — no embed or unembed keys."""
        cfg = make_cfg()
        model = make_mock_hubert(cfg)
        sd = convert_hubert_weights(model, cfg)

        assert "embed.W_E" not in sd
        assert "unembed.W_U" not in sd

    def test_raises_on_missing_encoder(self):
        model = mock.Mock(spec=[])
        model.encoder = None
        cfg = make_cfg()
        with pytest.raises((ValueError, AttributeError)):
            convert_hubert_weights(model, cfg)
