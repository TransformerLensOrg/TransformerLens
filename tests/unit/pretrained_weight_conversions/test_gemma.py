"""
Unit tests for Gemma weight conversion.

Tests cover:
1. Multimodal vs text-only model detection
2. Weight extraction from multimodal models
3. Q/K normalization weight loading
4. Device compatibility (MPS/CUDA)
"""

import torch
import torch.nn as nn

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.pretrained.weight_conversions.gemma import convert_gemma_weights


def get_gemma3_config(
    n_layers: int = 2,
    use_qk_norm: bool = True,
    use_normalization_before_and_after: bool = True,
):
    """Create a Gemma 3 style config for testing."""
    return HookedTransformerConfig(
        d_model=128,
        d_head=64,
        n_heads=2,
        n_key_value_heads=1,
        d_mlp=512,
        n_ctx=128,
        n_layers=n_layers,
        d_vocab=1024,
        act_fn="gelu_pytorch_tanh",
        use_qk_norm=use_qk_norm,
        use_normalization_before_and_after=use_normalization_before_and_after,
        normalization_type="RMS",
        positional_embedding_type="rotary",
        gated_mlp=True,
    )


class MockGemmaLayer(nn.Module):
    """A mock Gemma layer with real nn.Module components."""

    def __init__(self, cfg: HookedTransformerConfig, use_qk_norm: bool = True):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(cfg.d_model)
        self.post_attention_layernorm = nn.LayerNorm(cfg.d_model)

        if cfg.use_normalization_before_and_after:
            self.pre_feedforward_layernorm = nn.LayerNorm(cfg.d_model)
            self.post_feedforward_layernorm = nn.LayerNorm(cfg.d_model)

        # Self attention
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.self_attn.k_proj = nn.Linear(
            cfg.d_model, cfg.n_key_value_heads * cfg.d_head, bias=False
        )
        self.self_attn.v_proj = nn.Linear(
            cfg.d_model, cfg.n_key_value_heads * cfg.d_head, bias=False
        )
        self.self_attn.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        # Q/K norms (Gemma 3)
        if use_qk_norm:
            self.self_attn.q_norm = nn.LayerNorm(cfg.d_head)
            self.self_attn.k_norm = nn.LayerNorm(cfg.d_head)

        # MLP
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.mlp.gate_proj = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.mlp.down_proj = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)


class MockGemmaTextModel(nn.Module):
    """A mock text-only Gemma3ForCausalLM model."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.d_vocab, cfg.d_model)
        self.model.norm = nn.LayerNorm(cfg.d_model)
        self.model.layers = nn.ModuleList(
            [MockGemmaLayer(cfg, use_qk_norm=cfg.use_qk_norm) for _ in range(cfg.n_layers)]
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)


class MockGemmaMultimodalModel(nn.Module):
    """A mock multimodal Gemma3ForConditionalGeneration model."""

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        # Multimodal structure: model.language_model.model
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()

        base = self.language_model.model
        base.embed_tokens = nn.Embedding(cfg.d_vocab, cfg.d_model)
        base.norm = nn.LayerNorm(cfg.d_model)
        base.layers = nn.ModuleList(
            [MockGemmaLayer(cfg, use_qk_norm=cfg.use_qk_norm) for _ in range(cfg.n_layers)]
        )
        # Note: No lm_head in multimodal, uses tied embeddings


# ============================================================================
# Test: Text-only model weight conversion
# ============================================================================


class TestGemmaTextOnlyConversion:
    """Test weight conversion for text-only Gemma3ForCausalLM models."""

    def test_convert_weights_basic(self):
        """Test basic weight conversion produces expected keys."""
        cfg = get_gemma3_config()
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        # Check essential keys exist
        assert "embed.W_E" in state_dict
        assert "ln_final.w" in state_dict
        assert "unembed.W_U" in state_dict
        assert "unembed.b_U" in state_dict

    def test_convert_weights_layer_keys(self):
        """Test that layer-specific keys are generated."""
        cfg = get_gemma3_config(n_layers=2)
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        for layer in range(cfg.n_layers):
            # Attention weights
            assert f"blocks.{layer}.attn.W_Q" in state_dict
            assert f"blocks.{layer}.attn._W_K" in state_dict
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

            # Layer norms
            assert f"blocks.{layer}.ln1.w" in state_dict
            assert f"blocks.{layer}.ln2.w" in state_dict

    def test_convert_weights_qk_norm(self):
        """Test that Q/K norm weights are extracted when use_qk_norm=True."""
        cfg = get_gemma3_config(use_qk_norm=True)
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        for layer in range(cfg.n_layers):
            assert f"blocks.{layer}.attn.q_norm.w" in state_dict
            assert f"blocks.{layer}.attn.k_norm.w" in state_dict

    def test_convert_weights_no_qk_norm(self):
        """Test that Q/K norm weights are not extracted when use_qk_norm=False."""
        cfg = get_gemma3_config(use_qk_norm=False)
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        for layer in range(cfg.n_layers):
            assert f"blocks.{layer}.attn.q_norm.w" not in state_dict
            assert f"blocks.{layer}.attn.k_norm.w" not in state_dict

    def test_convert_weights_normalization_before_and_after(self):
        """Test Gemma 2/3 style normalization weights."""
        cfg = get_gemma3_config(use_normalization_before_and_after=True)
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        for layer in range(cfg.n_layers):
            # Post-attention and post-feedforward norms (Gemma 2/3 style)
            assert f"blocks.{layer}.ln1_post.w" in state_dict
            assert f"blocks.{layer}.ln2_post.w" in state_dict


# ============================================================================
# Test: Multimodal model weight conversion
# ============================================================================


class TestGemmaMultimodalConversion:
    """Test weight conversion for multimodal Gemma3ForConditionalGeneration models."""

    def test_multimodal_detection(self):
        """Test that multimodal models are correctly detected."""
        cfg = get_gemma3_config()
        model = MockGemmaMultimodalModel(cfg)

        # Should have language_model attribute
        assert hasattr(model, "language_model")

        # Should produce valid state dict
        state_dict = convert_gemma_weights(model, cfg)
        assert "embed.W_E" in state_dict

    def test_multimodal_embedding_extraction(self):
        """Test that embeddings are extracted from language_model.model."""
        cfg = get_gemma3_config()
        model = MockGemmaMultimodalModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        # Check embedding exists and has correct shape
        assert "embed.W_E" in state_dict
        assert state_dict["embed.W_E"].shape[0] == cfg.d_vocab

    def test_multimodal_tied_embeddings_for_unembed(self):
        """Test that multimodal models use tied embeddings for unembed."""
        cfg = get_gemma3_config()
        model = MockGemmaMultimodalModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        # Unembed should exist
        assert "unembed.W_U" in state_dict


# ============================================================================
# Test: Weight shapes
# ============================================================================


class TestGemmaWeightShapes:
    """Test that converted weights have correct shapes."""

    def test_attention_weight_shapes(self):
        """Test attention weight shapes are correct."""
        cfg = get_gemma3_config()
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

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
        cfg = get_gemma3_config()
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        # W_in and W_gate: [d_model, d_mlp]
        assert state_dict["blocks.0.mlp.W_in"].shape == (cfg.d_model, cfg.d_mlp)
        assert state_dict["blocks.0.mlp.W_gate"].shape == (cfg.d_model, cfg.d_mlp)

        # W_out: [d_mlp, d_model]
        assert state_dict["blocks.0.mlp.W_out"].shape == (cfg.d_mlp, cfg.d_model)

    def test_embedding_scaling(self):
        """Test that embeddings are scaled by sqrt(d_model)."""
        cfg = get_gemma3_config()
        model = MockGemmaTextModel(cfg)

        original_embed = model.model.embed_tokens.weight.clone()
        state_dict = convert_gemma_weights(model, cfg)

        # Embedding should be scaled by sqrt(d_model)
        expected_scale = cfg.d_model**0.5
        # Check approximately equal (allowing for dtype conversion)
        ratio = state_dict["embed.W_E"] / original_embed
        assert torch.allclose(ratio, torch.full_like(ratio, expected_scale), rtol=1e-4)


# ============================================================================
# Test: Device consistency
# ============================================================================


class TestGemmaDeviceConsistency:
    """Test that bias tensors are created on the correct device."""

    def test_bias_tensors_have_matching_device(self):
        """Test that zero bias tensors match weight tensor devices."""
        cfg = get_gemma3_config()
        model = MockGemmaTextModel(cfg)

        state_dict = convert_gemma_weights(model, cfg)

        # Check that bias tensors are on the same device as their weights
        for layer in range(cfg.n_layers):
            w_q = state_dict[f"blocks.{layer}.attn.W_Q"]
            b_q = state_dict[f"blocks.{layer}.attn.b_Q"]
            assert w_q.device == b_q.device

            w_out = state_dict[f"blocks.{layer}.mlp.W_out"]
            b_out = state_dict[f"blocks.{layer}.mlp.b_out"]
            assert w_out.device == b_out.device
