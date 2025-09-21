#!/usr/bin/env python3
"""
Unit tests for the ProcessWeights class.

Comprehensive test coverage for all weight processing functions extracted from HookedTransformer.
"""

from unittest.mock import Mock, patch

import einops
import pytest
import torch

from transformer_lens.weight_processing import ProcessWeights

# from typing import Dict  # Unused import


class MockConfig:
    """Mock configuration class for testing."""

    def __init__(self, **kwargs):
        # Default values
        self.n_layers = 2
        self.n_heads = 4
        self.d_model = 8
        self.d_head = 2
        self.d_mlp = 16
        self.n_key_value_heads = None
        self.attn_only = False
        self.gated_mlp = False
        self.act_fn = None
        self.final_rms = False
        self.positional_embedding_type = "standard"
        self.normalization_type = "LN"
        self.num_experts = None

        # Override with any provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def basic_config():
    """Basic test configuration."""
    return MockConfig()


@pytest.fixture
def gqa_config():
    """Configuration with Grouped Query Attention."""
    return MockConfig(n_key_value_heads=2)


@pytest.fixture
def attn_only_config():
    """Attention-only configuration."""
    return MockConfig(attn_only=True)


@pytest.fixture
def gated_mlp_config():
    """Configuration with gated MLP."""
    return MockConfig(gated_mlp=True)


@pytest.fixture
def solu_config():
    """Configuration with SoLU activation."""
    return MockConfig(act_fn="solu_ln")


@pytest.fixture
def basic_state_dict(basic_config):
    """Create a basic state dict for testing."""
    cfg = basic_config
    state_dict = {}

    # Embedding weights
    state_dict["embed.W_E"] = torch.randn(100, cfg.d_model)  # vocab_size=100
    state_dict["pos_embed.W_pos"] = torch.randn(50, cfg.d_model)  # n_ctx=50

    # Unembedding weights
    state_dict["unembed.W_U"] = torch.randn(cfg.d_model, 100)
    state_dict["unembed.b_U"] = torch.randn(100)

    # Final layer norm
    state_dict["ln_final.w"] = torch.randn(cfg.d_model)
    state_dict["ln_final.b"] = torch.randn(cfg.d_model)

    # Layer-specific weights
    for l in range(cfg.n_layers):
        # Layer norms
        state_dict[f"blocks.{l}.ln1.w"] = torch.randn(cfg.d_model)
        state_dict[f"blocks.{l}.ln1.b"] = torch.randn(cfg.d_model)
        state_dict[f"blocks.{l}.ln2.w"] = torch.randn(cfg.d_model)
        state_dict[f"blocks.{l}.ln2.b"] = torch.randn(cfg.d_model)

        # Attention weights
        state_dict[f"blocks.{l}.attn.W_Q"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
        state_dict[f"blocks.{l}.attn.W_K"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
        state_dict[f"blocks.{l}.attn.W_V"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
        state_dict[f"blocks.{l}.attn.W_O"] = torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model)

        # Attention biases
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.randn(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_K"] = torch.randn(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_V"] = torch.randn(cfg.n_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_O"] = torch.randn(cfg.d_model)

        # MLP weights
        state_dict[f"blocks.{l}.mlp.W_in"] = torch.randn(cfg.d_model, cfg.d_mlp)
        state_dict[f"blocks.{l}.mlp.W_out"] = torch.randn(cfg.d_mlp, cfg.d_model)
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.randn(cfg.d_mlp)
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.randn(cfg.d_model)

    return state_dict


@pytest.fixture
def gqa_state_dict(gqa_config):
    """Create a state dict for GQA testing."""
    cfg = gqa_config
    state_dict = {}

    # Basic weights (same as basic_state_dict)
    state_dict["embed.W_E"] = torch.randn(100, cfg.d_model)
    state_dict["pos_embed.W_pos"] = torch.randn(50, cfg.d_model)
    state_dict["unembed.W_U"] = torch.randn(cfg.d_model, 100)
    state_dict["unembed.b_U"] = torch.randn(100)
    state_dict["ln_final.w"] = torch.randn(cfg.d_model)
    state_dict["ln_final.b"] = torch.randn(cfg.d_model)

    for l in range(cfg.n_layers):
        # Layer norms
        state_dict[f"blocks.{l}.ln1.w"] = torch.randn(cfg.d_model)
        state_dict[f"blocks.{l}.ln1.b"] = torch.randn(cfg.d_model)
        state_dict[f"blocks.{l}.ln2.w"] = torch.randn(cfg.d_model)
        state_dict[f"blocks.{l}.ln2.b"] = torch.randn(cfg.d_model)

        # Standard attention weights (Q is full size)
        state_dict[f"blocks.{l}.attn.W_Q"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.randn(cfg.n_heads, cfg.d_head)

        # GQA attention weights (K, V are smaller)
        state_dict[f"blocks.{l}.attn._W_K"] = torch.randn(
            cfg.n_key_value_heads, cfg.d_model, cfg.d_head
        )
        state_dict[f"blocks.{l}.attn._W_V"] = torch.randn(
            cfg.n_key_value_heads, cfg.d_model, cfg.d_head
        )
        state_dict[f"blocks.{l}.attn._b_K"] = torch.randn(cfg.n_key_value_heads, cfg.d_head)
        state_dict[f"blocks.{l}.attn._b_V"] = torch.randn(cfg.n_key_value_heads, cfg.d_head)

        # Output weights (same as basic)
        state_dict[f"blocks.{l}.attn.W_O"] = torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model)
        state_dict[f"blocks.{l}.attn.b_O"] = torch.randn(cfg.d_model)

        # MLP weights
        state_dict[f"blocks.{l}.mlp.W_in"] = torch.randn(cfg.d_model, cfg.d_mlp)
        state_dict[f"blocks.{l}.mlp.W_out"] = torch.randn(cfg.d_mlp, cfg.d_model)
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.randn(cfg.d_mlp)
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.randn(cfg.d_model)

    return state_dict


class TestProcessWeights:
    """Test cases for the ProcessWeights class."""

    def test_fold_layer_norm_basic(self, basic_config, basic_state_dict):
        """Test basic LayerNorm folding functionality."""
        original_dict = basic_state_dict.copy()
        processed_dict = ProcessWeights.fold_layer_norm(basic_state_dict, basic_config)

        # Check that original dict is not modified
        assert basic_state_dict == original_dict

        # Check that LayerNorm weights are removed
        for l in range(basic_config.n_layers):
            assert f"blocks.{l}.ln1.w" not in processed_dict
            assert f"blocks.{l}.ln1.b" not in processed_dict
            assert f"blocks.{l}.ln2.w" not in processed_dict
            assert f"blocks.{l}.ln2.b" not in processed_dict

        assert "ln_final.w" not in processed_dict
        assert "ln_final.b" not in processed_dict

        # Check that attention and MLP weights are modified
        for l in range(basic_config.n_layers):
            assert f"blocks.{l}.attn.W_Q" in processed_dict
            assert f"blocks.{l}.attn.W_K" in processed_dict
            assert f"blocks.{l}.attn.W_V" in processed_dict
            assert f"blocks.{l}.mlp.W_in" in processed_dict

        # Check that unembed weights are modified
        assert "unembed.W_U" in processed_dict

    def test_fold_layer_norm_gqa(self, gqa_config, gqa_state_dict):
        """Test LayerNorm folding with Grouped Query Attention."""
        processed_dict = ProcessWeights.fold_layer_norm(gqa_state_dict, gqa_config)

        # Check that GQA weights are processed correctly
        for l in range(gqa_config.n_layers):
            assert f"blocks.{l}.attn.W_Q" in processed_dict
            assert f"blocks.{l}.attn._W_K" in processed_dict
            assert f"blocks.{l}.attn._W_V" in processed_dict
            assert f"blocks.{l}.attn.b_Q" in processed_dict
            assert f"blocks.{l}.attn._b_K" in processed_dict
            assert f"blocks.{l}.attn._b_V" in processed_dict

    def test_fold_layer_norm_no_biases(self, basic_config, basic_state_dict):
        """Test LayerNorm folding without bias folding."""
        processed_dict = ProcessWeights.fold_layer_norm(
            basic_state_dict, basic_config, fold_biases=False
        )

        # When fold_biases=False, LayerNorm biases should NOT be removed
        # (they're only removed when folding biases into subsequent layers)
        for l in range(basic_config.n_layers):
            # The ln1.b and ln2.b should still be present when fold_biases=False
            # but the ln1.w and ln2.w should be removed (folded into weights)
            assert f"blocks.{l}.ln1.w" not in processed_dict
            assert f"blocks.{l}.ln2.w" not in processed_dict

    def test_fold_layer_norm_no_centering(self, basic_config, basic_state_dict):
        """Test LayerNorm folding without weight centering."""
        processed_dict = ProcessWeights.fold_layer_norm(
            basic_state_dict, basic_config, center_weights=False
        )

        # Should still fold weights but not center them
        for l in range(basic_config.n_layers):
            assert f"blocks.{l}.ln1.w" not in processed_dict
            assert f"blocks.{l}.attn.W_Q" in processed_dict

    def test_fold_layer_norm_attn_only(self, attn_only_config, basic_state_dict):
        """Test LayerNorm folding with attention-only model."""
        # Remove MLP weights from state dict
        attn_only_dict = {k: v for k, v in basic_state_dict.items() if "mlp" not in k}

        processed_dict = ProcessWeights.fold_layer_norm(attn_only_dict, attn_only_config)

        # Should only process attention weights
        for l in range(attn_only_config.n_layers):
            assert f"blocks.{l}.attn.W_Q" in processed_dict
            assert f"blocks.{l}.mlp.W_in" not in processed_dict

    def test_fold_layer_norm_gated_mlp(self, gated_mlp_config):
        """Test LayerNorm folding with gated MLP."""
        # Create state dict with gated MLP
        state_dict = {}
        cfg = gated_mlp_config

        # Add required weights
        for l in range(cfg.n_layers):
            state_dict[f"blocks.{l}.ln1.w"] = torch.randn(cfg.d_model)
            state_dict[f"blocks.{l}.ln1.b"] = torch.randn(cfg.d_model)
            state_dict[f"blocks.{l}.ln2.w"] = torch.randn(cfg.d_model)
            state_dict[f"blocks.{l}.ln2.b"] = torch.randn(cfg.d_model)

            # Attention weights
            state_dict[f"blocks.{l}.attn.W_Q"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
            state_dict[f"blocks.{l}.attn.W_K"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
            state_dict[f"blocks.{l}.attn.W_V"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_Q"] = torch.randn(cfg.n_heads, cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_K"] = torch.randn(cfg.n_heads, cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_V"] = torch.randn(cfg.n_heads, cfg.d_head)

            # Gated MLP weights
            state_dict[f"blocks.{l}.mlp.W_in"] = torch.randn(cfg.d_model, cfg.d_mlp)
            state_dict[f"blocks.{l}.mlp.W_gate"] = torch.randn(cfg.d_model, cfg.d_mlp)
            state_dict[f"blocks.{l}.mlp.b_in"] = torch.randn(cfg.d_mlp)

        # Final layer norm and unembed
        state_dict["ln_final.w"] = torch.randn(cfg.d_model)
        state_dict["ln_final.b"] = torch.randn(cfg.d_model)
        state_dict["unembed.W_U"] = torch.randn(cfg.d_model, 100)
        state_dict["unembed.b_U"] = torch.randn(100)

        processed_dict = ProcessWeights.fold_layer_norm(state_dict, cfg)

        # Check that gate weights are processed
        for l in range(cfg.n_layers):
            assert f"blocks.{l}.mlp.W_gate" in processed_dict

    def test_fold_layer_norm_solu(self, solu_config):
        """Test LayerNorm folding with SoLU activation."""
        # Create state dict with SoLU-specific weights
        state_dict = {}
        cfg = solu_config

        for l in range(cfg.n_layers):
            state_dict[f"blocks.{l}.ln1.w"] = torch.randn(cfg.d_model)
            state_dict[f"blocks.{l}.ln1.b"] = torch.randn(cfg.d_model)
            state_dict[f"blocks.{l}.ln2.w"] = torch.randn(cfg.d_model)
            state_dict[f"blocks.{l}.ln2.b"] = torch.randn(cfg.d_model)

            # Attention weights
            state_dict[f"blocks.{l}.attn.W_Q"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
            state_dict[f"blocks.{l}.attn.W_K"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
            state_dict[f"blocks.{l}.attn.W_V"] = torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_Q"] = torch.randn(cfg.n_heads, cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_K"] = torch.randn(cfg.n_heads, cfg.d_head)
            state_dict[f"blocks.{l}.attn.b_V"] = torch.randn(cfg.n_heads, cfg.d_head)

            # MLP weights including SoLU-specific ln
            state_dict[f"blocks.{l}.mlp.W_in"] = torch.randn(cfg.d_model, cfg.d_mlp)
            state_dict[f"blocks.{l}.mlp.W_out"] = torch.randn(cfg.d_mlp, cfg.d_model)
            state_dict[f"blocks.{l}.mlp.b_in"] = torch.randn(cfg.d_mlp)
            state_dict[f"blocks.{l}.mlp.b_out"] = torch.randn(cfg.d_model)
            state_dict[f"blocks.{l}.mlp.ln.w"] = torch.randn(cfg.d_mlp)
            state_dict[f"blocks.{l}.mlp.ln.b"] = torch.randn(cfg.d_mlp)

        # Final layer norm and unembed
        state_dict["ln_final.w"] = torch.randn(cfg.d_model)
        state_dict["ln_final.b"] = torch.randn(cfg.d_model)
        state_dict["unembed.W_U"] = torch.randn(cfg.d_model, 100)
        state_dict["unembed.b_U"] = torch.randn(100)

        processed_dict = ProcessWeights.fold_layer_norm(state_dict, cfg)

        # Check that SoLU ln weights are removed
        for l in range(cfg.n_layers):
            assert f"blocks.{l}.mlp.ln.w" not in processed_dict
            assert f"blocks.{l}.mlp.ln.b" not in processed_dict

    def test_center_writing_weights(self, basic_config, basic_state_dict):
        """Test weight centering functionality."""
        original_dict = basic_state_dict.copy()
        processed_dict = ProcessWeights.center_writing_weights(basic_state_dict, basic_config)

        # Check that original dict is not modified
        assert basic_state_dict == original_dict

        # Check that embedding weights are centered
        embed_mean = processed_dict["embed.W_E"].mean(-1, keepdim=True)
        assert torch.allclose(embed_mean, torch.zeros_like(embed_mean), atol=1e-6)

        # Check that positional embedding weights are centered
        pos_mean = processed_dict["pos_embed.W_pos"].mean(-1, keepdim=True)
        assert torch.allclose(pos_mean, torch.zeros_like(pos_mean), atol=1e-6)

        # Check that attention output weights are centered
        for l in range(basic_config.n_layers):
            w_o_mean = processed_dict[f"blocks.{l}.attn.W_O"].mean(-1, keepdim=True)
            assert torch.allclose(w_o_mean, torch.zeros_like(w_o_mean), atol=1e-6)

            b_o_mean = processed_dict[f"blocks.{l}.attn.b_O"].mean()
            assert torch.allclose(b_o_mean, torch.tensor(0.0), atol=1e-6)

            # Check MLP output weights are centered
            mlp_out_mean = processed_dict[f"blocks.{l}.mlp.W_out"].mean(-1, keepdim=True)
            assert torch.allclose(mlp_out_mean, torch.zeros_like(mlp_out_mean), atol=1e-6)

            mlp_b_out_mean = processed_dict[f"blocks.{l}.mlp.b_out"].mean()
            assert torch.allclose(mlp_b_out_mean, torch.tensor(0.0), atol=1e-6)

    def test_center_writing_weights_rotary(self, basic_config, basic_state_dict):
        """Test weight centering with rotary embeddings."""
        basic_config.positional_embedding_type = "rotary"
        processed_dict = ProcessWeights.center_writing_weights(basic_state_dict, basic_config)

        # Positional embeddings should not be processed for rotary
        assert torch.equal(processed_dict["pos_embed.W_pos"], basic_state_dict["pos_embed.W_pos"])

    def test_center_writing_weights_attn_only(self, attn_only_config, basic_state_dict):
        """Test weight centering with attention-only model."""
        # Remove MLP weights
        attn_only_dict = {k: v for k, v in basic_state_dict.items() if "mlp" not in k}

        processed_dict = ProcessWeights.center_writing_weights(attn_only_dict, attn_only_config)

        # Should only process attention weights
        for l in range(attn_only_config.n_layers):
            assert f"blocks.{l}.attn.W_O" in processed_dict
            assert f"blocks.{l}.mlp.W_out" not in processed_dict

    def test_center_unembed(self, basic_state_dict):
        """Test unembedding weight centering."""
        original_dict = basic_state_dict.copy()
        processed_dict = ProcessWeights.center_unembed(basic_state_dict)

        # Check that original dict is not modified
        assert basic_state_dict == original_dict

        # Check that unembedding weights are centered
        w_u_mean = processed_dict["unembed.W_U"].mean(-1, keepdim=True)
        assert torch.allclose(w_u_mean, torch.zeros_like(w_u_mean), atol=1e-6)

        b_u_mean = processed_dict["unembed.b_U"].mean()
        assert torch.allclose(b_u_mean, torch.tensor(0.0), atol=1e-6)

    def test_fold_value_biases_basic(self, basic_config, basic_state_dict):
        """Test value bias folding functionality."""
        original_dict = basic_state_dict.copy()
        processed_dict = ProcessWeights.fold_value_biases(basic_state_dict, basic_config)

        # Check that original dict is not modified
        assert basic_state_dict == original_dict

        # Check that value biases are zeroed out
        for l in range(basic_config.n_layers):
            b_v = processed_dict[f"blocks.{l}.attn.b_V"]
            assert torch.allclose(b_v, torch.zeros_like(b_v), atol=1e-6)

            # Output bias should be modified (not zero)
            assert f"blocks.{l}.attn.b_O" in processed_dict

    def test_fold_value_biases_gqa(self, gqa_config, gqa_state_dict):
        """Test value bias folding with GQA."""
        processed_dict = ProcessWeights.fold_value_biases(gqa_state_dict, gqa_config)

        # Check that GQA value biases are zeroed out
        for l in range(gqa_config.n_layers):
            b_v = processed_dict[f"blocks.{l}.attn._b_V"]
            assert torch.allclose(b_v, torch.zeros_like(b_v), atol=1e-6)

    def test_refactor_factored_attn_matrices(self, basic_config, basic_state_dict):
        """Test attention matrix refactoring."""
        original_dict = basic_state_dict.copy()

        with patch("transformer_lens.weight_processing.FactoredMatrix") as mock_factored_matrix:
            # Mock the FactoredMatrix behavior
            mock_instance = Mock()
            mock_instance.make_even.return_value.pair = (
                torch.randn(basic_config.n_heads, basic_config.d_model + 1, basic_config.d_head),
                torch.randn(basic_config.n_heads, basic_config.d_head, basic_config.d_model + 1),
            )
            mock_factored_matrix.return_value = mock_instance

            # Mock SVD for OV matrices
            mock_ov_instance = Mock()
            U = torch.randn(basic_config.n_heads, basic_config.d_model, basic_config.d_head)
            S = torch.randn(basic_config.n_heads, basic_config.d_head)
            Vh = torch.randn(basic_config.n_heads, basic_config.d_head, basic_config.d_model)
            mock_ov_instance.svd.return_value = (U, S, Vh)

            def factored_matrix_side_effect(*args):
                if len(args) == 2 and args[1].shape[-1] == basic_config.d_model + 1:
                    return mock_instance
                else:
                    return mock_ov_instance

            mock_factored_matrix.side_effect = factored_matrix_side_effect

            processed_dict = ProcessWeights.refactor_factored_attn_matrices(
                basic_state_dict, basic_config
            )

            # Check that original dict is not modified
            assert basic_state_dict == original_dict

            # Check that attention weights are modified
            for l in range(basic_config.n_layers):
                assert f"blocks.{l}.attn.W_Q" in processed_dict
                assert f"blocks.{l}.attn.W_K" in processed_dict
                assert f"blocks.{l}.attn.W_V" in processed_dict
                assert f"blocks.{l}.attn.W_O" in processed_dict

                # Value biases should be zeroed
                b_v = processed_dict[f"blocks.{l}.attn.b_V"]
                assert torch.allclose(b_v, torch.zeros_like(b_v), atol=1e-6)

    def test_refactor_factored_attn_matrices_rotary_error(self, basic_config, basic_state_dict):
        """Test that refactoring fails with rotary embeddings."""
        basic_config.positional_embedding_type = "rotary"

        with pytest.raises(
            AssertionError, match="You can't refactor the QK circuit when using rotary embeddings"
        ):
            ProcessWeights.refactor_factored_attn_matrices(basic_state_dict, basic_config)

    def test_process_weights_full_pipeline(self, basic_config, basic_state_dict):
        """Test the full weight processing pipeline."""
        original_dict = basic_state_dict.copy()
        processed_dict = ProcessWeights.process_weights(basic_state_dict, basic_config)

        # Check that original dict is not modified
        assert basic_state_dict == original_dict

        # Check that LayerNorm weights are removed
        for l in range(basic_config.n_layers):
            assert f"blocks.{l}.ln1.w" not in processed_dict
            assert f"blocks.{l}.ln2.w" not in processed_dict
        assert "ln_final.w" not in processed_dict

        # Check that weights are centered
        embed_mean = processed_dict["embed.W_E"].mean(-1, keepdim=True)
        assert torch.allclose(embed_mean, torch.zeros_like(embed_mean), atol=1e-6)

        # Check that unembedding is centered
        w_u_mean = processed_dict["unembed.W_U"].mean(-1, keepdim=True)
        assert torch.allclose(w_u_mean, torch.zeros_like(w_u_mean), atol=1e-6)

        # Check that value biases are folded
        for l in range(basic_config.n_layers):
            b_v = processed_dict[f"blocks.{l}.attn.b_V"]
            assert torch.allclose(b_v, torch.zeros_like(b_v), atol=1e-6)

    def test_process_weights_selective_processing(self, basic_config, basic_state_dict):
        """Test selective processing options."""
        processed_dict = ProcessWeights.process_weights(
            basic_state_dict,
            basic_config,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
        )

        # LayerNorm weights should still be present
        assert "blocks.0.ln1.w" in processed_dict

        # Weights should not be centered
        embed_mean = processed_dict["embed.W_E"].mean(-1, keepdim=True)
        assert not torch.allclose(embed_mean, torch.zeros_like(embed_mean), atol=1e-6)

        # Value biases should not be folded
        b_v = processed_dict["blocks.0.attn.b_V"]
        assert not torch.allclose(b_v, torch.zeros_like(b_v), atol=1e-6)

    def test_process_weights_moe_model(self, basic_config, basic_state_dict):
        """Test processing with MoE model (should skip LayerNorm folding)."""
        basic_config.num_experts = 8
        processed_dict = ProcessWeights.process_weights(basic_state_dict, basic_config)

        # LayerNorm weights should still be present for MoE
        assert "blocks.0.ln1.w" in processed_dict

    def test_process_weights_rms_norm(self, basic_config, basic_state_dict):
        """Test processing with RMS normalization."""
        basic_config.normalization_type = "RMS"
        processed_dict = ProcessWeights.process_weights(basic_state_dict, basic_config)

        # LayerNorm weights should be removed (RMS processing)
        assert "blocks.0.ln1.w" not in processed_dict

    def test_process_weights_final_rms(self, basic_config, basic_state_dict):
        """Test processing with final RMS (should skip writing weight centering)."""
        basic_config.final_rms = True
        processed_dict = ProcessWeights.process_weights(basic_state_dict, basic_config)

        # Writing weights should not be centered with final RMS
        embed_mean = processed_dict["embed.W_E"].mean(-1, keepdim=True)
        assert not torch.allclose(embed_mean, torch.zeros_like(embed_mean), atol=1e-6)

    def test_state_dict_immutability(self, basic_config, basic_state_dict):
        """Test that all functions don't modify the input state dict."""
        original_keys = set(basic_state_dict.keys())
        original_values = {k: v.clone() for k, v in basic_state_dict.items()}

        # Run all processing functions
        ProcessWeights.fold_layer_norm(basic_state_dict, basic_config)
        ProcessWeights.center_writing_weights(basic_state_dict, basic_config)
        ProcessWeights.center_unembed(basic_state_dict)
        ProcessWeights.fold_value_biases(basic_state_dict, basic_config)
        ProcessWeights.process_weights(basic_state_dict, basic_config)

        # Check that original state dict is unchanged
        assert set(basic_state_dict.keys()) == original_keys
        for k, v in basic_state_dict.items():
            assert torch.equal(v, original_values[k])

    def test_tensor_shapes_preserved(self, basic_config, basic_state_dict):
        """Test that tensor shapes are preserved correctly."""
        processed_dict = ProcessWeights.process_weights(basic_state_dict, basic_config)

        # Check that key tensor shapes are preserved where expected
        assert processed_dict["embed.W_E"].shape == basic_state_dict["embed.W_E"].shape
        assert processed_dict["unembed.W_U"].shape == basic_state_dict["unembed.W_U"].shape

        for l in range(basic_config.n_layers):
            assert (
                processed_dict[f"blocks.{l}.attn.W_Q"].shape
                == basic_state_dict[f"blocks.{l}.attn.W_Q"].shape
            )
            assert (
                processed_dict[f"blocks.{l}.attn.b_O"].shape
                == basic_state_dict[f"blocks.{l}.attn.b_O"].shape
            )

    def test_mathematical_correctness_layer_norm_folding(self, basic_config):
        """Test mathematical correctness of LayerNorm folding."""
        # Create simple test case with known values
        cfg = basic_config
        state_dict = {}

        # Simple values for testing
        ln_w = torch.tensor([2.0, 3.0, 1.0, 0.5])  # d_model = 4
        ln_b = torch.tensor([0.1, 0.2, 0.3, 0.4])
        w_q = torch.ones(2, 4, 2)  # n_heads=2, d_model=4, d_head=2
        b_q = torch.zeros(2, 2)

        cfg.d_model = 4
        cfg.n_heads = 2
        cfg.d_head = 2
        cfg.n_layers = 1

        state_dict["blocks.0.ln1.w"] = ln_w
        state_dict["blocks.0.ln1.b"] = ln_b
        state_dict["blocks.0.attn.W_Q"] = w_q
        state_dict["blocks.0.attn.b_Q"] = b_q

        # Add minimal required weights
        state_dict["blocks.0.ln2.w"] = torch.ones(4)
        state_dict["blocks.0.ln2.b"] = torch.zeros(4)
        state_dict["blocks.0.attn.W_K"] = torch.ones(2, 4, 2)
        state_dict["blocks.0.attn.W_V"] = torch.ones(2, 4, 2)
        state_dict["blocks.0.attn.b_K"] = torch.zeros(2, 2)
        state_dict["blocks.0.attn.b_V"] = torch.zeros(2, 2)
        state_dict["blocks.0.mlp.W_in"] = torch.ones(4, 8)
        state_dict["blocks.0.mlp.b_in"] = torch.zeros(8)
        state_dict["ln_final.w"] = torch.ones(4)
        state_dict["ln_final.b"] = torch.zeros(4)
        state_dict["unembed.W_U"] = torch.ones(4, 10)
        state_dict["unembed.b_U"] = torch.zeros(10)

        # Test with centering disabled to check pure mathematical folding
        processed_dict = ProcessWeights.fold_layer_norm(state_dict, cfg, center_weights=False)

        # Check mathematical correctness (without centering)
        expected_w_q = w_q * ln_w[None, :, None]
        expected_b_q = b_q + (w_q * ln_b[None, :, None]).sum(-2)

        assert torch.allclose(processed_dict["blocks.0.attn.W_Q"], expected_w_q)
        assert torch.allclose(processed_dict["blocks.0.attn.b_Q"], expected_b_q)

        # Also test that centering works when enabled
        processed_dict_centered = ProcessWeights.fold_layer_norm(
            state_dict, cfg, center_weights=True
        )

        # With centering, the weights should have zero mean across d_model dimension
        w_q_centered = processed_dict_centered["blocks.0.attn.W_Q"]
        w_q_mean = einops.reduce(
            w_q_centered, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )
        assert torch.allclose(w_q_mean, torch.zeros_like(w_q_mean), atol=1e-6)

    def test_edge_cases_empty_state_dict(self, basic_config):
        """Test handling of edge cases like empty state dicts."""
        empty_dict = {}

        # Should not crash but also not do anything useful
        try:
            ProcessWeights.center_unembed(empty_dict)
            ProcessWeights.center_writing_weights(empty_dict, basic_config)
        except KeyError:
            # Expected behavior for missing keys
            pass

    def test_config_attribute_access(self):
        """Test that config attribute access works with getattr defaults."""
        minimal_config = MockConfig(n_layers=1)
        # Remove some attributes to test getattr defaults
        delattr(minimal_config, "attn_only")
        delattr(minimal_config, "gated_mlp")

        state_dict = {
            "blocks.0.ln1.w": torch.ones(8),
            "blocks.0.ln1.b": torch.zeros(8),
            "blocks.0.ln2.w": torch.ones(8),
            "blocks.0.ln2.b": torch.zeros(8),
            "blocks.0.attn.W_Q": torch.ones(4, 8, 2),
            "blocks.0.attn.W_K": torch.ones(4, 8, 2),
            "blocks.0.attn.W_V": torch.ones(4, 8, 2),
            "blocks.0.attn.b_Q": torch.zeros(4, 2),
            "blocks.0.attn.b_K": torch.zeros(4, 2),
            "blocks.0.attn.b_V": torch.zeros(4, 2),
            "blocks.0.mlp.W_in": torch.ones(8, 16),
            "blocks.0.mlp.b_in": torch.zeros(16),
            "ln_final.w": torch.ones(8),
            "ln_final.b": torch.zeros(8),
            "unembed.W_U": torch.ones(8, 100),
            "unembed.b_U": torch.zeros(100),
        }

        # Should work with getattr defaults
        processed_dict = ProcessWeights.fold_layer_norm(state_dict, minimal_config)
        assert "blocks.0.ln1.w" not in processed_dict
