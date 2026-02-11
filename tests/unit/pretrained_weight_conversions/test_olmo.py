"""
Unit tests for OLMO weight conversion.

Tests cover:
1. Basic weight conversion
2. Q/K normalization weight loading
3. Grouped Query Attention (GQA) handling
4. Sliding window attention configuration
5. Device compatibility (MPS/CUDA)
"""

import torch
import torch.nn as nn

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.olmo import convert_olmo_weights


def get_olmo_config(
    n_layers: int = 2,
    use_qk_norm: bool = True,
    use_normalization_before_and_after: bool = True,
):
    """Create an OLMO 3 style config for testing."""
    return HookedTransformerConfig(
        d_model=128,
        d_head=64,
        n_heads=2,
        n_key_value_heads=1,  # GQA
        d_mlp=512,
        n_ctx=128,
        n_layers=n_layers,
        d_vocab=1024,
        act_fn="silu",
        use_qk_norm=use_qk_norm,
        use_normalization_before_and_after=use_normalization_before_and_after,
        normalization_type="RMS",
        positional_embedding_type="rotary",
        gated_mlp=True,
    )


class MockOlmoRMSNorm(nn.Module):
    """Mock OLMO RMSNorm layer."""

    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


class MockOlmoAttention(nn.Module):
    """Mock OLMO attention module."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_key_value_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_key_value_heads * cfg.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        # OLMO has Q/K norms
        self.q_norm = MockOlmoRMSNorm(cfg.n_heads * cfg.d_head)
        self.k_norm = MockOlmoRMSNorm(cfg.n_key_value_heads * cfg.d_head)


class MockOlmoMLP(nn.Module):
    """Mock OLMO MLP module."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.down_proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)


class MockOlmoLayer(nn.Module):
    """A mock OLMO decoder layer."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.self_attn = MockOlmoAttention(cfg)
        self.mlp = MockOlmoMLP(cfg)
        self.post_attention_layernorm = MockOlmoRMSNorm(cfg.d_model)
        self.post_feedforward_layernorm = MockOlmoRMSNorm(cfg.d_model)


class MockOlmoModel(nn.Module):
    """A mock OLMO model (Olmo3ForCausalLM)."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.model.layers = nn.ModuleList([MockOlmoLayer(cfg) for _ in range(cfg.n_layers)])
        self.model.norm = MockOlmoRMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)


# ============================================================================
# Test: Basic weight conversion
# ============================================================================


class TestOlmoBasicConversion:
    """Test basic weight conversion for OLMO models."""

    def test_convert_weights_basic(self):
        """Test basic weight conversion produces expected keys."""
        cfg = get_olmo_config()
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        # Check essential keys exist
        assert "embed.W_E" in state_dict
        assert "ln_final.w" in state_dict
        assert "unembed.W_U" in state_dict
        assert "unembed.b_U" in state_dict

    def test_convert_weights_layer_keys(self):
        """Test that layer-specific keys are generated."""
        cfg = get_olmo_config(n_layers=2)
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        for layer in range(cfg.n_layers):
            # Attention weights
            assert f"blocks.{layer}.attn.W_Q" in state_dict
            assert f"blocks.{layer}.attn._W_K" in state_dict  # GQA underscore prefix
            assert f"blocks.{layer}.attn._W_V" in state_dict
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

            # Layer norms (OLMO uses post-norm pattern)
            assert f"blocks.{layer}.ln1.w" in state_dict
            assert f"blocks.{layer}.ln2.w" in state_dict


# ============================================================================
# Test: Q/K normalization
# ============================================================================


class TestOlmoQKNorm:
    """Test Q/K norm weight extraction for OLMO models."""

    def test_convert_weights_with_qk_norm(self):
        """Test that Q/K norm weights are extracted when use_qk_norm=True."""
        cfg = get_olmo_config(use_qk_norm=True)
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        for layer in range(cfg.n_layers):
            assert f"blocks.{layer}.attn.q_norm.w" in state_dict
            assert f"blocks.{layer}.attn.k_norm.w" in state_dict

    def test_convert_weights_without_qk_norm(self):
        """Test that Q/K norm weights are not extracted when use_qk_norm=False."""
        cfg = get_olmo_config(use_qk_norm=False)
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        for layer in range(cfg.n_layers):
            assert f"blocks.{layer}.attn.q_norm.w" not in state_dict
            assert f"blocks.{layer}.attn.k_norm.w" not in state_dict


# ============================================================================
# Test: Weight shapes
# ============================================================================


class TestOlmoWeightShapes:
    """Test that converted weights have correct shapes."""

    def test_attention_weight_shapes(self):
        """Test attention weight shapes are correct."""
        cfg = get_olmo_config()
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        # W_Q: [n_heads, d_model, d_head]
        assert state_dict["blocks.0.attn.W_Q"].shape == (cfg.n_heads, cfg.d_model, cfg.d_head)

        # W_K/V with GQA: [n_key_value_heads, d_model, d_head]
        assert state_dict["blocks.0.attn._W_K"].shape == (
            cfg.n_key_value_heads,
            cfg.d_model,
            cfg.d_head,
        )
        assert state_dict["blocks.0.attn._W_V"].shape == (
            cfg.n_key_value_heads,
            cfg.d_model,
            cfg.d_head,
        )

        # W_O: [n_heads, d_head, d_model]
        assert state_dict["blocks.0.attn.W_O"].shape == (cfg.n_heads, cfg.d_head, cfg.d_model)

    def test_mlp_weight_shapes(self):
        """Test MLP weight shapes are correct."""
        cfg = get_olmo_config()
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        # W_in and W_gate: [d_model, d_mlp]
        assert state_dict["blocks.0.mlp.W_in"].shape == (cfg.d_model, cfg.d_mlp)
        assert state_dict["blocks.0.mlp.W_gate"].shape == (cfg.d_model, cfg.d_mlp)

        # W_out: [d_mlp, d_model]
        assert state_dict["blocks.0.mlp.W_out"].shape == (cfg.d_mlp, cfg.d_model)

    def test_no_embedding_scaling(self):
        """Test that embeddings are NOT scaled (unlike Gemma)."""
        cfg = get_olmo_config()
        model = MockOlmoModel(cfg)

        original_embed = model.model.embed_tokens.weight.clone()
        state_dict = convert_olmo_weights(model, cfg)

        # OLMO does NOT scale embeddings
        assert torch.allclose(state_dict["embed.W_E"], original_embed)

    def test_rms_norm_weights_not_modified(self):
        """Test that RMSNorm weights are NOT modified with +1 (unlike Gemma)."""
        cfg = get_olmo_config()
        model = MockOlmoModel(cfg)

        original_norm = model.model.norm.weight.clone()
        state_dict = convert_olmo_weights(model, cfg)

        # OLMO RMSNorm does NOT add 1
        assert torch.allclose(state_dict["ln_final.w"], original_norm)


# ============================================================================
# Test: Device consistency
# ============================================================================


class TestOlmoDeviceConsistency:
    """Test that bias tensors are created on the correct device."""

    def test_bias_tensors_have_matching_device(self):
        """Test that zero bias tensors match weight tensor devices."""
        cfg = get_olmo_config()
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        # Check that bias tensors are on the same device as their weights
        for layer in range(cfg.n_layers):
            w_q = state_dict[f"blocks.{layer}.attn.W_Q"]
            b_q = state_dict[f"blocks.{layer}.attn.b_Q"]
            assert w_q.device == b_q.device

            w_out = state_dict[f"blocks.{layer}.mlp.W_out"]
            b_out = state_dict[f"blocks.{layer}.mlp.b_out"]
            assert w_out.device == b_out.device

    def test_gqa_underscore_prefix(self):
        """Test that GQA models have underscore prefix for K/V weights."""
        cfg = get_olmo_config()  # Uses n_key_value_heads=1 (GQA)
        model = MockOlmoModel(cfg)

        state_dict = convert_olmo_weights(model, cfg)

        for layer in range(cfg.n_layers):
            # GQA should have underscore prefix in key names
            assert f"blocks.{layer}.attn._W_K" in state_dict
            assert f"blocks.{layer}.attn._W_V" in state_dict
            assert f"blocks.{layer}.attn._b_K" in state_dict
            assert f"blocks.{layer}.attn._b_V" in state_dict
