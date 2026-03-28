"""Tests for Apertus weight conversion."""

from unittest import mock

import torch

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.apertus import (
    convert_apertus_weights,
)


def make_cfg(use_qk_norm=True, n_key_value_heads=4):
    return HookedTransformerConfig(
        n_layers=1,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        n_ctx=32,
        d_vocab=100,
        act_fn="xielu",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        gated_mlp=False,
        final_rms=True,
        use_qk_norm=use_qk_norm,
        n_key_value_heads=n_key_value_heads,
        dtype=torch.float32,
        device="cpu",
    )


def make_mock_model(cfg, has_act_fn=True):
    """Build a minimal mock HF Apertus model."""
    d = cfg.d_model
    d_mlp = cfg.d_mlp
    n_heads = cfg.n_heads
    n_kv = cfg.n_key_value_heads or cfg.n_heads
    d_head = cfg.d_head

    model = mock.Mock()
    model.model.embed_tokens.weight = torch.randn(cfg.d_vocab, d)
    model.model.norm.weight = torch.randn(d)
    model.lm_head.weight = torch.randn(cfg.d_vocab, d)

    layer = mock.Mock()
    layer.attention_layernorm.weight = torch.randn(d)
    layer.feedforward_layernorm.weight = torch.randn(d)

    # Attention
    layer.self_attn.q_proj.weight = torch.randn(n_heads * d_head, d)
    layer.self_attn.k_proj.weight = torch.randn(n_kv * d_head, d)
    layer.self_attn.v_proj.weight = torch.randn(n_kv * d_head, d)
    layer.self_attn.o_proj.weight = torch.randn(d, n_heads * d_head)

    # QK norm
    layer.self_attn.q_norm.weight = torch.randn(d_head)
    layer.self_attn.k_norm.weight = torch.randn(d_head)

    # Non-gated MLP
    layer.mlp.up_proj.weight = torch.randn(d_mlp, d)
    layer.mlp.down_proj.weight = torch.randn(d, d_mlp)

    # XIeLU activation parameters
    if has_act_fn:
        layer.mlp.act_fn.alpha_p = torch.tensor(0.3)
        layer.mlp.act_fn.alpha_n = torch.tensor(0.4)
        layer.mlp.act_fn.beta = torch.tensor(0.6)
    else:
        # Remove act_fn to test fallback
        del layer.mlp.act_fn

    model.model.layers = [layer]
    return model


class TestApertusWeightConversion:
    def test_embedding_extraction(self):
        cfg = make_cfg()
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert "embed.W_E" in sd
        assert sd["embed.W_E"].shape == (cfg.d_vocab, cfg.d_model)

    def test_attention_weight_shapes(self):
        cfg = make_cfg()
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)

        assert sd["blocks.0.attn.W_Q"].shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
        assert sd["blocks.0.attn._W_K"].shape == (cfg.n_key_value_heads, cfg.d_model, cfg.d_head)
        assert sd["blocks.0.attn._W_V"].shape == (cfg.n_key_value_heads, cfg.d_model, cfg.d_head)
        assert sd["blocks.0.attn.W_O"].shape == (cfg.n_heads, cfg.d_head, cfg.d_model)

    def test_gqa_underscore_keys(self):
        """GQA models should have _W_K/_W_V (with underscore prefix)."""
        cfg = make_cfg(n_key_value_heads=4)
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert "blocks.0.attn._W_K" in sd
        assert "blocks.0.attn._W_V" in sd
        assert "blocks.0.attn._b_K" in sd
        assert "blocks.0.attn._b_V" in sd

    def test_no_gqa_no_underscore(self):
        """Non-GQA models should have W_K/W_V (no underscore)."""
        cfg = make_cfg(n_key_value_heads=None)
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert "blocks.0.attn.W_K" in sd
        assert "blocks.0.attn.W_V" in sd

    def test_qk_norm_extracted(self):
        cfg = make_cfg(use_qk_norm=True)
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert "blocks.0.attn.q_norm.w" in sd
        assert "blocks.0.attn.k_norm.w" in sd
        assert sd["blocks.0.attn.q_norm.w"].shape == (cfg.d_head,)

    def test_no_qk_norm_when_disabled(self):
        cfg = make_cfg(use_qk_norm=False)
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert "blocks.0.attn.q_norm.w" not in sd
        assert "blocks.0.attn.k_norm.w" not in sd

    def test_non_gated_mlp_weights(self):
        """Apertus uses non-gated MLP (up_proj + down_proj, no gate_proj)."""
        cfg = make_cfg()
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert sd["blocks.0.mlp.W_in"].shape == (cfg.d_model, cfg.d_mlp)
        assert sd["blocks.0.mlp.W_out"].shape == (cfg.d_mlp, cfg.d_model)
        assert "blocks.0.mlp.W_gate" not in sd

    def test_xielu_params_extracted(self):
        cfg = make_cfg()
        model = make_mock_model(cfg, has_act_fn=True)
        sd = convert_apertus_weights(model, cfg)
        torch.testing.assert_close(sd["blocks.0.mlp.act_fn.alpha_p"], torch.tensor(0.3))
        torch.testing.assert_close(sd["blocks.0.mlp.act_fn.alpha_n"], torch.tensor(0.4))
        torch.testing.assert_close(sd["blocks.0.mlp.act_fn.beta"], torch.tensor(0.6))

    def test_xielu_params_fallback_defaults(self):
        """When activation params aren't found, defaults should be used."""
        cfg = make_cfg()
        model = make_mock_model(cfg, has_act_fn=False)
        # Also remove alternate attribute paths
        layer = model.model.layers[0]
        if hasattr(layer.mlp, "act"):
            del layer.mlp.act
        if hasattr(layer.mlp, "alpha_p"):
            del layer.mlp.alpha_p

        sd = convert_apertus_weights(model, cfg)
        torch.testing.assert_close(
            sd["blocks.0.mlp.act_fn.alpha_p"], torch.tensor(0.8, dtype=cfg.dtype)
        )
        torch.testing.assert_close(
            sd["blocks.0.mlp.act_fn.alpha_n"], torch.tensor(0.8, dtype=cfg.dtype)
        )
        torch.testing.assert_close(
            sd["blocks.0.mlp.act_fn.beta"], torch.tensor(0.5, dtype=cfg.dtype)
        )

    def test_layer_norm_names(self):
        """Apertus uses attention_layernorm/feedforward_layernorm."""
        cfg = make_cfg()
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert "blocks.0.ln1.w" in sd
        assert "blocks.0.ln2.w" in sd

    def test_zero_biases_have_correct_device(self):
        cfg = make_cfg()
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        for key in [
            "blocks.0.attn.b_Q",
            "blocks.0.attn.b_O",
            "blocks.0.mlp.b_in",
            "blocks.0.mlp.b_out",
            "unembed.b_U",
        ]:
            assert sd[key].device.type == cfg.device, f"{key} on wrong device"

    def test_unembed_shapes(self):
        cfg = make_cfg()
        model = make_mock_model(cfg)
        sd = convert_apertus_weights(model, cfg)
        assert sd["unembed.W_U"].shape == (cfg.d_model, cfg.d_vocab)
        assert sd["ln_final.w"].shape == (cfg.d_model,)
