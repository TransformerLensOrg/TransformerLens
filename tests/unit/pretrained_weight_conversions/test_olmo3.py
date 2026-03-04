"""
Unit tests for OLMo 3/3.1 weight conversion.

Tests cover:
1. Basic weight conversion produces expected keys
2. GQA (Grouped Query Attention) with underscore prefix for K/V weights
3. Q/K normalization weight loading
4. Device consistency across all tensors
5. Weight shapes match expected dimensions
"""

import torch
import torch.nn as nn

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.olmo3 import convert_olmo3_weights


def get_olmo3_config(
    n_layers: int = 2,
    use_qk_norm: bool = True,
    n_key_value_heads: int = 1,
):
    """Create an OLMo 3 style config for testing."""
    return HookedTransformerConfig(
        d_model=128,
        d_head=64,
        n_heads=2,
        n_key_value_heads=n_key_value_heads,
        d_mlp=512,
        n_ctx=2048,
        n_layers=n_layers,
        d_vocab=50304,
        act_fn="silu",
        use_qk_norm=use_qk_norm,
        use_normalization_before_and_after=True,
        normalization_type="RMS",
        positional_embedding_type="rotary",
        gated_mlp=True,
    )


class MockOlmo3Layer(nn.Module):
    """A mock OLMo 3 layer with real nn.Module components."""

    def __init__(self, cfg: HookedTransformerConfig, use_qk_norm: bool = True):
        super().__init__()

        # OLMo 3 uses post-attention and post-feedforward layer norms (RMSNorm)
        self.post_attention_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.post_feedforward_layernorm = nn.RMSNorm(cfg.d_model, eps=cfg.eps)

        # Self attention
        n_kv_heads = cfg.n_key_value_heads if cfg.n_key_value_heads else cfg.n_heads
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.self_attn.k_proj = nn.Linear(cfg.d_model, n_kv_heads * cfg.d_head, bias=False)
        self.self_attn.v_proj = nn.Linear(cfg.d_model, n_kv_heads * cfg.d_head, bias=False)
        self.self_attn.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        # Q/K norms (OLMo 3)
        if use_qk_norm:
            self.self_attn.q_norm = nn.RMSNorm(cfg.d_head, eps=cfg.eps)
            self.self_attn.k_norm = nn.RMSNorm(cfg.d_head, eps=cfg.eps)

        # MLP (Gated / SwiGLU-style)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.mlp.gate_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.mlp.down_proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)


class MockOlmo3Model(nn.Module):
    """A mock Olmo3ForCausalLM model."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.model.norm = nn.RMSNorm(cfg.d_model, eps=cfg.eps)
        self.model.layers = nn.ModuleList(
            [MockOlmo3Layer(cfg, use_qk_norm=cfg.use_qk_norm) for _ in range(cfg.n_layers)]
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)


# ============================================================================
# Test: Basic weight conversion
# ============================================================================


class TestOlmo3BasicConversion:
    """Test basic weight conversion for OLMo 3 models."""

    def test_convert_weights_basic(self):
        """Test basic weight conversion produces expected keys."""
        cfg = get_olmo3_config()
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        # Check essential keys exist
        assert "embed.W_E" in state_dict
        assert "ln_final.w" in state_dict
        assert "unembed.W_U" in state_dict
        assert "unembed.b_U" in state_dict

    def test_convert_weights_layer_keys(self):
        """Test that layer-specific keys are generated."""
        cfg = get_olmo3_config(n_layers=2)
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        for layer in range(cfg.n_layers):
            # Attention weights
            assert f"blocks.{layer}.attn.W_Q" in state_dict
            assert f"blocks.{layer}.attn._W_K" in state_dict  # GQA underscore prefix
            assert f"blocks.{layer}.attn._W_V" in state_dict  # GQA underscore prefix
            assert f"blocks.{layer}.attn.W_O" in state_dict

            # Attention biases
            assert f"blocks.{layer}.attn.b_Q" in state_dict
            assert f"blocks.{layer}.attn._b_K" in state_dict
            assert f"blocks.{layer}.attn._b_V" in state_dict
            assert f"blocks.{layer}.attn.b_O" in state_dict

            # MLP weights
            assert f"blocks.{layer}.mlp.W_in" in state_dict
            assert f"blocks.{layer}.mlp.W_gate" in state_dict
            assert f"blocks.{layer}.mlp.W_out" in state_dict

            # Layer norms
            assert f"blocks.{layer}.ln1.w" in state_dict
            assert f"blocks.{layer}.ln2.w" in state_dict

    def test_convert_weights_no_biases_in_source(self):
        """Test that OLMo 3 has no bias weights in the source model."""
        cfg = get_olmo3_config()
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        # All biases should be zero tensors created during conversion
        assert torch.all(state_dict["blocks.0.attn.b_Q"] == 0)
        assert torch.all(state_dict["blocks.0.attn._b_K"] == 0)
        assert torch.all(state_dict["blocks.0.attn.b_O"] == 0)
        assert torch.all(state_dict["unembed.b_U"] == 0)


# ============================================================================
# Test: Q/K normalization
# ============================================================================


class TestOlmo3QKNorm:
    """Test Q/K normalization weight loading."""

    def test_qk_norm_weights_extracted(self):
        """Test that Q/K norm weights are extracted when use_qk_norm=True."""
        cfg = get_olmo3_config(use_qk_norm=True)
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        for layer in range(cfg.n_layers):
            assert f"blocks.{layer}.attn.q_norm.w" in state_dict
            assert f"blocks.{layer}.attn.k_norm.w" in state_dict
            # Verify weights are not zeros (actual model weights)
            assert not torch.all(state_dict[f"blocks.{layer}.attn.q_norm.w"] == 0)
            assert not torch.all(state_dict[f"blocks.{layer}.attn.k_norm.w"] == 0)


# ============================================================================
# Test: GQA (Grouped Query Attention)
# ============================================================================


class TestOlmo3GQA:
    """Test Grouped Query Attention (GQA) handling."""

    def test_gqa_underscore_prefix(self):
        """Test that GQA models have underscore prefix for K/V weights."""
        cfg = get_olmo3_config(n_key_value_heads=1)  # GQA
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        for layer in range(cfg.n_layers):
            # GQA should have underscore prefix in key names
            assert f"blocks.{layer}.attn._W_K" in state_dict
            assert f"blocks.{layer}.attn._W_V" in state_dict
            assert f"blocks.{layer}.attn._b_K" in state_dict
            assert f"blocks.{layer}.attn._b_V" in state_dict

    def test_no_gqa_no_underscore_prefix(self):
        """Test that non-GQA models don't have underscore prefix."""
        # Note: OLMo 3 always uses GQA (n_key_value_heads < n_heads)
        # This test documents the expected behavior for MHA models
        cfg = get_olmo3_config(n_key_value_heads=2)  # MHA (n_kv_heads == n_heads)
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        for layer in range(cfg.n_layers):
            # MHA should not have underscore prefix
            assert f"blocks.{layer}.attn.W_K" in state_dict
            assert f"blocks.{layer}.attn.W_V" in state_dict


# ============================================================================
# Test: Device consistency
# ============================================================================


class TestOlmo3DeviceConsistency:
    """Test device consistency across converted weights."""

    def test_all_tensors_on_same_device(self):
        """Test that all converted tensors are on the same device as source."""
        cfg = get_olmo3_config()

        # Test on CPU
        model = MockOlmo3Model(cfg)
        state_dict = convert_olmo3_weights(model, cfg)

        for key, value in state_dict.items():
            assert value.device == model.model.embed_tokens.weight.device, (
                f"Tensor {key} is on {value.device} but should be on "
                f"{model.model.embed_tokens.weight.device}"
            )

    def test_bias_tensors_use_correct_device(self):
        """Test that zero bias tensors use device from source weights."""
        cfg = get_olmo3_config()
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        # Check that bias tensors use the device from source weights
        expected_device = model.model.layers[0].self_attn.q_proj.weight.device
        assert state_dict["blocks.0.attn.b_Q"].device == expected_device
        assert state_dict["blocks.0.attn._b_K"].device == expected_device


# ============================================================================
# Test: Weight shapes
# ============================================================================


class TestOlmo3WeightShapes:
    """Test that converted weights have correct shapes."""

    def test_attention_weight_shapes(self):
        """Test attention weight shapes are correct."""
        cfg = get_olmo3_config(n_layers=1)
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        # W_Q: (n_heads, d_model, d_head)
        assert state_dict["blocks.0.attn.W_Q"].shape == (cfg.n_heads, cfg.d_model, cfg.d_head)
        # W_K: (n_kv_heads, d_model, d_head)
        assert state_dict["blocks.0.attn._W_K"].shape == (
            cfg.n_key_value_heads,
            cfg.d_model,
            cfg.d_head,
        )
        # W_V: (n_kv_heads, d_model, d_head)
        assert state_dict["blocks.0.attn._W_V"].shape == (
            cfg.n_key_value_heads,
            cfg.d_model,
            cfg.d_head,
        )
        # W_O: (n_heads, d_head, d_model)
        assert state_dict["blocks.0.attn.W_O"].shape == (cfg.n_heads, cfg.d_head, cfg.d_model)

    def test_mlp_weight_shapes(self):
        """Test MLP weight shapes are correct."""
        cfg = get_olmo3_config(n_layers=1)
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        # W_in: (d_model, d_mlp) - transposed from HF (d_mlp, d_model)
        assert state_dict["blocks.0.mlp.W_in"].shape == (cfg.d_model, cfg.d_mlp)
        # W_gate: (d_model, d_mlp) - transposed from HF (d_mlp, d_model)
        assert state_dict["blocks.0.mlp.W_gate"].shape == (cfg.d_model, cfg.d_mlp)
        # W_out: (d_mlp, d_model) - transposed from HF (d_model, d_mlp)
        assert state_dict["blocks.0.mlp.W_out"].shape == (cfg.d_mlp, cfg.d_model)

    def test_embedding_shapes(self):
        """Test embedding weight shapes are correct."""
        cfg = get_olmo3_config()
        model = MockOlmo3Model(cfg)

        state_dict = convert_olmo3_weights(model, cfg)

        assert state_dict["embed.W_E"].shape == (cfg.d_vocab, cfg.d_model)
        assert state_dict["unembed.W_U"].shape == (cfg.d_model, cfg.d_vocab)
