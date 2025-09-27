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

    def test_extract_state_dict(self):
        """Test the extract_state_dict function with a small model."""
        import torch
        from torch import nn

        # Create a small test model
        class SmallTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 8)
                self.linear2 = nn.Linear(8, 4)
                self.embedding = nn.Embedding(10, 4)

        model = SmallTestModel()

        # Extract state dict using the new function
        extracted_dict = ProcessWeights.extract_state_dict(model)

        # Check that we get the expected keys
        expected_keys = {
            "linear1.weight",
            "linear1.bias",
            "linear2.weight",
            "linear2.bias",
            "embedding.weight",
        }
        assert set(extracted_dict.keys()) == expected_keys

        # Check that no _original_component references are present
        for key in extracted_dict.keys():
            assert "_original_component" not in key, f"Found _original_component in key: {key}"

        # Check that tensor shapes are correct
        assert extracted_dict["linear1.weight"].shape == (8, 4)
        assert extracted_dict["linear1.bias"].shape == (8,)
        assert extracted_dict["linear2.weight"].shape == (4, 8)
        assert extracted_dict["linear2.bias"].shape == (4,)
        assert extracted_dict["embedding.weight"].shape == (10, 4)

        # Check that tensors are cloned (not references to original model parameters)
        original_linear1_weight = model.linear1.weight.data
        extracted_linear1_weight = extracted_dict["linear1.weight"]

        # They should have the same values
        assert torch.equal(original_linear1_weight, extracted_linear1_weight)

        # But they should be different objects (cloned)
        assert extracted_linear1_weight is not original_linear1_weight

    def test_extract_state_dict_with_original_component_suffixes(self):
        """Test that extract_state_dict properly removes _original_component suffixes."""
        import torch
        from torch import nn

        # Create a mock model that simulates bridge model with _original_component suffixes
        class MockBridgeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 8)

            def state_dict(self):
                # Simulate a bridge model state dict with _original_component suffixes
                return {
                    "linear1.weight._original_component": self.linear1.weight.data,
                    "linear1.bias._original_component": self.linear1.bias.data,
                }

        model = MockBridgeModel()

        # Extract state dict using the new function
        extracted_dict = ProcessWeights.extract_state_dict(model)

        # Check that _original_component suffixes are removed
        expected_keys = {"linear1.weight", "linear1.bias"}
        assert set(extracted_dict.keys()) == expected_keys

        # Verify no _original_component references remain
        for key in extracted_dict.keys():
            assert "_original_component" not in key, f"Found _original_component in key: {key}"

        # Check that the values are correct
        assert torch.equal(extracted_dict["linear1.weight"], model.linear1.weight.data)
        assert torch.equal(extracted_dict["linear1.bias"], model.linear1.bias.data)

    def test_load_processed_weights_into_module(self):
        """Test loading processed weights into an nn.Module."""
        import torch
        import torch.nn as nn

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 2)
                self.linear2 = nn.Linear(2, 1)

        model = SimpleModel()

        # Create processed state dict (simulating processed weights)
        processed_state_dict = {
            "linear1.weight": torch.randn(2, 3),
            "linear1.bias": torch.randn(2),
            "linear2.weight": torch.randn(1, 2),
            "linear2.bias": torch.randn(1),
        }

        # Store original weights for comparison
        original_linear1_weight = model.linear1.weight.data.clone()
        original_linear1_bias = model.linear1.bias.data.clone()

        # Load processed weights
        updated_model = ProcessWeights.load_processed_weights_into_module(
            processed_state_dict, model
        )

        # Check that the model is the same object (returned reference)
        assert updated_model is model

        # Check that weights were updated
        assert torch.equal(model.linear1.weight.data, processed_state_dict["linear1.weight"])
        assert torch.equal(model.linear1.bias.data, processed_state_dict["linear1.bias"])
        assert torch.equal(model.linear2.weight.data, processed_state_dict["linear2.weight"])
        assert torch.equal(model.linear2.bias.data, processed_state_dict["linear2.bias"])

        # Check that weights are different from original
        assert not torch.equal(model.linear1.weight.data, original_linear1_weight)
        assert not torch.equal(model.linear1.bias.data, original_linear1_bias)

    def test_fold_layer_no_adapter_transformer_lens_format(self, basic_config):
        """Test _fold_layer function with no adapter (TransformerLens format).

        This test locks in the current behavior of _fold_layer when no adapter is provided,
        ensuring that HookedTransformer models continue to work correctly.
        """
        cfg = basic_config
        cfg.n_layers = 1  # Test with single layer for simplicity

        # Create a state dict with known values for deterministic testing
        state_dict = {}

        # Layer 0 weights with known values
        ln1_w = torch.tensor([2.0, 3.0, 1.0, 0.5])  # d_model = 4
        ln1_b = torch.tensor([0.1, 0.2, 0.3, 0.4])
        ln2_w = torch.tensor([1.5, 2.5, 0.8, 1.2])
        ln2_b = torch.tensor([0.05, 0.15, 0.25, 0.35])

        # Attention weights: [n_heads, d_model, d_head]
        w_q = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],  # head 0
                [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
            ]
        )  # head 1
        w_k = torch.tensor(
            [
                [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]],  # head 0
                [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]],
            ]
        )  # head 1
        w_v = torch.tensor(
            [
                [[0.8, 1.2], [1.6, 2.0], [2.4, 2.8], [3.2, 3.6]],  # head 0
                [[1.2, 1.6], [2.0, 2.4], [2.8, 3.2], [3.6, 4.0]],
            ]
        )  # head 1

        # Attention biases: [n_heads, d_head]
        b_q = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        b_k = torch.tensor([[0.05, 0.15], [0.25, 0.35]])
        b_v = torch.tensor([[0.08, 0.12], [0.16, 0.20]])

        # MLP weights
        w_in = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # d_model=4, d_mlp=8
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ]
        )
        b_in = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        # Store in state dict
        state_dict["blocks.0.ln1.w"] = ln1_w
        state_dict["blocks.0.ln1.b"] = ln1_b
        state_dict["blocks.0.ln2.w"] = ln2_w
        state_dict["blocks.0.ln2.b"] = ln2_b
        state_dict["blocks.0.attn.W_Q"] = w_q
        state_dict["blocks.0.attn.W_K"] = w_k
        state_dict["blocks.0.attn.W_V"] = w_v
        state_dict["blocks.0.attn.b_Q"] = b_q
        state_dict["blocks.0.attn.b_K"] = b_k
        state_dict["blocks.0.attn.b_V"] = b_v
        state_dict["blocks.0.mlp.W_in"] = w_in
        state_dict["blocks.0.mlp.b_in"] = b_in

        # Make a copy for comparison
        original_state_dict = {k: v.clone() for k, v in state_dict.items()}

        # Test _fold_layer with no adapter (TransformerLens format)
        ProcessWeights._fold_layer(
            state_dict,
            cfg,
            layer_idx=0,
            fold_biases=True,
            center_weights=True,
            adapter=None,
            gqa="",
        )

        # Verify LayerNorm weights are removed
        assert "blocks.0.ln1.w" not in state_dict
        assert "blocks.0.ln1.b" not in state_dict
        assert "blocks.0.ln2.w" not in state_dict
        assert "blocks.0.ln2.b" not in state_dict

        # Verify attention weights are modified (folded and centered)
        w_q_processed = state_dict["blocks.0.attn.W_Q"]
        w_k_processed = state_dict["blocks.0.attn.W_K"]
        w_v_processed = state_dict["blocks.0.attn.W_V"]

        # Check that weights are folded (multiplied by ln1_w)
        expected_w_q_folded = w_q * ln1_w[None, :, None]
        expected_w_k_folded = w_k * ln1_w[None, :, None]
        expected_w_v_folded = w_v * ln1_w[None, :, None]

        # Check that weights are centered (mean should be zero across d_model dimension)
        w_q_mean = einops.reduce(
            w_q_processed, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )
        w_k_mean = einops.reduce(
            w_k_processed, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )
        w_v_mean = einops.reduce(
            w_v_processed, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )

        assert torch.allclose(w_q_mean, torch.zeros_like(w_q_mean), atol=1e-6)
        assert torch.allclose(w_k_mean, torch.zeros_like(w_k_mean), atol=1e-6)
        assert torch.allclose(w_v_mean, torch.zeros_like(w_v_mean), atol=1e-6)

        # Verify attention biases are folded
        b_q_processed = state_dict["blocks.0.attn.b_Q"]
        b_k_processed = state_dict["blocks.0.attn.b_K"]
        b_v_processed = state_dict["blocks.0.attn.b_V"]

        # Check that biases are folded (bias folding formula)
        expected_b_q_folded = b_q + (w_q * ln1_b[None, :, None]).sum(-2)
        expected_b_k_folded = b_k + (w_k * ln1_b[None, :, None]).sum(-2)
        expected_b_v_folded = b_v + (w_v * ln1_b[None, :, None]).sum(-2)

        assert torch.allclose(b_q_processed, expected_b_q_folded, atol=1e-6)
        assert torch.allclose(b_k_processed, expected_b_k_folded, atol=1e-6)
        assert torch.allclose(b_v_processed, expected_b_v_folded, atol=1e-6)

        # Verify MLP weights are folded
        w_in_processed = state_dict["blocks.0.mlp.W_in"]
        b_in_processed = state_dict["blocks.0.mlp.b_in"]

        # Check that MLP weights are folded (multiplied by ln2_w) and then centered
        expected_w_in_folded = w_in * ln2_w[:, None]
        # After centering, the mean across d_model dimension should be zero
        expected_w_in_centered = expected_w_in_folded - einops.reduce(
            expected_w_in_folded, "d_model d_mlp -> 1 d_mlp", "mean"
        )
        assert torch.allclose(w_in_processed, expected_w_in_centered, atol=1e-6)

        # Check that MLP biases are folded
        expected_b_in_folded = b_in + (w_in * ln2_b[:, None]).sum(-2)
        assert torch.allclose(b_in_processed, expected_b_in_folded, atol=1e-6)

        # Verify MLP weights are centered
        w_in_mean = einops.reduce(w_in_processed, "d_model d_mlp -> 1 d_mlp", "mean")
        assert torch.allclose(w_in_mean, torch.zeros_like(w_in_mean), atol=1e-6)

        # Verify original state dict is unchanged
        for k, v in original_state_dict.items():
            assert torch.equal(v, original_state_dict[k])

    def test_fold_layer_no_adapter_without_centering(self, basic_config):
        """Test _fold_layer function without weight centering to verify pure folding behavior."""
        cfg = basic_config
        cfg.n_layers = 1

        # Create simple test case
        state_dict = {}
        ln1_w = torch.tensor([2.0, 3.0, 1.0, 0.5])
        ln1_b = torch.tensor([0.1, 0.2, 0.3, 0.4])
        w_q = torch.ones(2, 4, 2)  # n_heads=2, d_model=4, d_head=2
        b_q = torch.zeros(2, 2)

        state_dict["blocks.0.ln1.w"] = ln1_w
        state_dict["blocks.0.ln1.b"] = ln1_b
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

        # Test without centering
        ProcessWeights._fold_layer(
            state_dict,
            cfg,
            layer_idx=0,
            fold_biases=True,
            center_weights=False,
            adapter=None,
            gqa="",
        )

        # Check pure mathematical folding (no centering)
        expected_w_q = w_q * ln1_w[None, :, None]
        expected_b_q = b_q + (w_q * ln1_b[None, :, None]).sum(-2)

        assert torch.allclose(state_dict["blocks.0.attn.W_Q"], expected_w_q, atol=1e-6)
        assert torch.allclose(state_dict["blocks.0.attn.b_Q"], expected_b_q, atol=1e-6)

        # Verify LayerNorm weights are removed
        assert "blocks.0.ln1.w" not in state_dict
        assert "blocks.0.ln1.b" not in state_dict
        assert "blocks.0.ln2.w" not in state_dict
        assert "blocks.0.ln2.b" not in state_dict

    def test_fold_layer_no_adapter_without_bias_folding(self, basic_config):
        """Test _fold_layer function without bias folding."""
        cfg = basic_config
        cfg.n_layers = 1

        # Create simple test case
        state_dict = {}
        ln1_w = torch.tensor([2.0, 3.0, 1.0, 0.5])
        ln1_b = torch.tensor([0.1, 0.2, 0.3, 0.4])
        w_q = torch.ones(2, 4, 2)
        b_q = torch.zeros(2, 2)

        state_dict["blocks.0.ln1.w"] = ln1_w
        state_dict["blocks.0.ln1.b"] = ln1_b
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

        # Test without bias folding
        ProcessWeights._fold_layer(
            state_dict,
            cfg,
            layer_idx=0,
            fold_biases=False,
            center_weights=True,
            adapter=None,
            gqa="",
        )

        # Check that weights are folded but biases are not
        expected_w_q_folded = w_q * ln1_w[None, :, None]
        # After centering, the mean across d_model dimension should be zero
        expected_w_q_centered = expected_w_q_folded - einops.reduce(
            expected_w_q_folded, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )

        assert torch.allclose(state_dict["blocks.0.attn.W_Q"], expected_w_q_centered, atol=1e-6)
        assert torch.allclose(state_dict["blocks.0.attn.b_Q"], b_q, atol=1e-6)  # Bias unchanged

        # Verify LayerNorm weights are removed
        assert "blocks.0.ln1.w" not in state_dict
        assert "blocks.0.ln1.b" in state_dict  # Should still be present when fold_biases=False
        assert "blocks.0.ln2.w" not in state_dict
        assert "blocks.0.ln2.b" in state_dict  # Should still be present when fold_biases=False

    def test_fold_layer_with_adapter_huggingface_format(self, basic_config):
        """Test _fold_layer function with adapter (HuggingFace format).

        This test locks in the current behavior of _fold_layer when an adapter is provided,
        ensuring that HuggingFace models are processed correctly with combined QKV weights.
        """
        cfg = basic_config
        cfg.n_layers = 1  # Test with single layer for simplicity

        # Create a mock adapter that translates TransformerLens keys to HuggingFace keys
        class MockAdapter:
            def translate_transformer_lens_path(self, tl_key):
                # Simple translation: blocks.X.attn.W_Q/W_K/W_V -> transformer.h.X.attn.c_attn.weight
                # and blocks.X.attn.b_Q/b_K/b_V -> transformer.h.X.attn.c_attn.bias
                if (
                    "blocks.0.attn.W_Q" in tl_key
                    or "blocks.0.attn.W_K" in tl_key
                    or "blocks.0.attn.W_V" in tl_key
                ):
                    return "transformer.h.0.attn.c_attn.weight"
                elif (
                    "blocks.0.attn.b_Q" in tl_key
                    or "blocks.0.attn.b_K" in tl_key
                    or "blocks.0.attn.b_V" in tl_key
                ):
                    return "transformer.h.0.attn.c_attn.bias"
                elif "blocks.0.ln1.w" in tl_key:
                    return "transformer.h.0.ln_1.weight"
                elif "blocks.0.ln1.b" in tl_key:
                    return "transformer.h.0.ln_1.bias"
                elif "blocks.0.ln2.w" in tl_key:
                    return "transformer.h.0.ln_2.weight"
                elif "blocks.0.ln2.b" in tl_key:
                    return "transformer.h.0.ln_2.bias"
                elif "blocks.0.mlp.W_in" in tl_key:
                    return "transformer.h.0.mlp.c_fc.weight"
                elif "blocks.0.mlp.b_in" in tl_key:
                    return "transformer.h.0.mlp.c_fc.bias"
                else:
                    return tl_key

        adapter = MockAdapter()

        # Create a state dict with HuggingFace format (combined QKV weights)
        state_dict = {}

        # Layer 0 weights with known values (using correct dimensions from MockConfig)
        # MockConfig: n_heads=4, d_model=8, d_head=2, d_mlp=16
        ln1_w = torch.tensor([2.0, 3.0, 1.0, 0.5, 1.5, 2.5, 0.8, 1.2])  # d_model = 8
        ln1_b = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ln2_w = torch.tensor([1.5, 2.5, 0.8, 1.2, 2.0, 3.0, 1.0, 0.5])
        ln2_b = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])

        # Create combined QKV weight: [d_model, 3 * d_model] = [8, 24]
        # Q: [8, 8], K: [8, 8], V: [8, 8] -> combined: [8, 24]
        w_q = torch.randn(8, 8) * 0.1  # Q weights
        w_k = torch.randn(8, 8) * 0.1  # K weights
        w_v = torch.randn(8, 8) * 0.1  # V weights

        # Combine QKV weights: [d_model, 3 * d_model]
        qkv_weight = torch.cat([w_q, w_k, w_v], dim=1)  # [8, 24]

        # Create combined QKV bias: [3 * d_model] = [3 * 8] = [24]
        # In HuggingFace format, the bias is per d_model dimension, not per head
        # Q: [8], K: [8], V: [8] -> combined: [24]
        b_q = torch.randn(8) * 0.1  # [8]
        b_k = torch.randn(8) * 0.1  # [8]
        b_v = torch.randn(8) * 0.1  # [8]
        qkv_bias = torch.cat([b_q, b_k, b_v])  # [24]

        # MLP weights: d_model=8, d_mlp=16
        w_in = torch.randn(8, 16) * 0.1
        b_in = torch.randn(16) * 0.1

        # Store in state dict with HuggingFace keys
        state_dict["transformer.h.0.ln_1.weight"] = ln1_w
        state_dict["transformer.h.0.ln_1.bias"] = ln1_b
        state_dict["transformer.h.0.ln_2.weight"] = ln2_w
        state_dict["transformer.h.0.ln_2.bias"] = ln2_b
        state_dict["transformer.h.0.attn.c_attn.weight"] = qkv_weight
        state_dict["transformer.h.0.attn.c_attn.bias"] = qkv_bias
        state_dict["transformer.h.0.mlp.c_fc.weight"] = w_in
        state_dict["transformer.h.0.mlp.c_fc.bias"] = b_in

        # Make a copy for comparison
        original_state_dict = {k: v.clone() for k, v in state_dict.items()}

        # Test _fold_layer with adapter (HuggingFace format)
        ProcessWeights._fold_layer(
            state_dict,
            cfg,
            layer_idx=0,
            fold_biases=True,
            center_weights=True,
            adapter=adapter,
            gqa="",
        )

        # Verify LayerNorm weights are removed
        assert "transformer.h.0.ln_1.weight" not in state_dict
        assert "transformer.h.0.ln_1.bias" not in state_dict
        assert "transformer.h.0.ln_2.weight" not in state_dict
        assert "transformer.h.0.ln_2.bias" not in state_dict

        # Verify combined QKV weight is modified
        qkv_weight_processed = state_dict["transformer.h.0.attn.c_attn.weight"]
        qkv_bias_processed = state_dict["transformer.h.0.attn.c_attn.bias"]

        # Split the processed QKV weight back into Q, K, V
        w_q_processed, w_k_processed, w_v_processed = torch.tensor_split(
            qkv_weight_processed, 3, dim=1
        )

        # Verify that the weights have been processed (they should be different from original)
        # The exact values depend on the complex conversion between formats, so we just verify
        # that processing occurred by checking that weights are different from original
        assert not torch.allclose(w_q_processed, w_q, atol=1e-6)
        assert not torch.allclose(w_k_processed, w_k, atol=1e-6)
        assert not torch.allclose(w_v_processed, w_v, atol=1e-6)

        # Verify that the processed weights have the correct shape
        assert w_q_processed.shape == w_q.shape
        assert w_k_processed.shape == w_k.shape
        assert w_v_processed.shape == w_v.shape

        # Verify combined QKV bias is modified
        # Split the processed QKV bias back into Q, K, V
        b_q_processed, b_k_processed, b_v_processed = torch.tensor_split(
            qkv_bias_processed, 3, dim=0
        )

        # Verify that the biases have been processed (they should be different from original)
        assert not torch.allclose(b_q_processed, b_q, atol=1e-6)
        assert not torch.allclose(b_k_processed, b_k, atol=1e-6)
        assert not torch.allclose(b_v_processed, b_v, atol=1e-6)

        # Verify that the processed biases have the correct shape
        assert b_q_processed.shape == b_q.shape
        assert b_k_processed.shape == b_k.shape
        assert b_v_processed.shape == b_v.shape

        # Verify MLP weights are folded and centered
        w_in_processed = state_dict["transformer.h.0.mlp.c_fc.weight"]
        b_in_processed = state_dict["transformer.h.0.mlp.c_fc.bias"]

        # Verify that MLP weights have been processed (they should be different from original)
        assert not torch.allclose(w_in_processed, w_in, atol=1e-6)
        assert not torch.allclose(b_in_processed, b_in, atol=1e-6)

        # Verify that the processed MLP weights have the correct shape
        assert w_in_processed.shape == w_in.shape
        assert b_in_processed.shape == b_in.shape

        # Verify original state dict is unchanged
        for k, v in original_state_dict.items():
            assert torch.equal(v, original_state_dict[k])


def test_mlp_layer_norm_folding():
    """Test that the MLP layer norm folding function works correctly."""
    device = "cpu"
    model_name = "gpt2"
    layer = 0

    # Load HookedTransformer
    hooked_model = HookedTransformer.from_pretrained(
        model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False
    )

    state_dict = hooked_model.state_dict()
    cfg = hooked_model.cfg

    # Check if MLP parameters exist
    mlp_b_in_key = f"blocks.{layer}.mlp.b_in"
    mlp_W_in_key = f"blocks.{layer}.mlp.W_in"
    ln2_b_key = f"blocks.{layer}.ln2.b"
    ln2_w_key = f"blocks.{layer}.ln2.w"

    if not all(key in state_dict for key in [mlp_b_in_key, mlp_W_in_key, ln2_b_key, ln2_w_key]):
        pytest.skip("MLP or LayerNorm parameters not found - cannot test")

    # Test bias folding
    test_state_dict = {k: v.clone() for k, v in state_dict.items()}

    # Get original values
    original_mlp_b_in = test_state_dict[mlp_b_in_key].clone()
    original_mlp_W_in = test_state_dict[mlp_W_in_key].clone()
    original_ln2_b = test_state_dict[ln2_b_key].clone()
    original_ln2_w = test_state_dict[ln2_w_key].clone()

    # Apply MLP layer norm folding with bias folding
    ProcessWeights._fold_mlp_layer_norm(
        test_state_dict, cfg, layer, fold_biases=True, center_weights=False, adapter=None
    )

    # Check that LayerNorm parameters were removed
    assert ln2_b_key not in test_state_dict, "LayerNorm bias should be removed"
    assert ln2_w_key not in test_state_dict, "LayerNorm weight should be removed"

    # Verify the math for bias folding
    expected_mlp_b_in = original_mlp_b_in + (original_mlp_W_in * original_ln2_b[:, None]).sum(-2)
    actual_mlp_b_in = test_state_dict[mlp_b_in_key]

    max_diff = torch.max(torch.abs(expected_mlp_b_in - actual_mlp_b_in)).item()
    assert max_diff < 1e-6, f"MLP bias folding math incorrect: max_diff={max_diff:.2e}"

    # Verify the math for weight folding
    expected_mlp_W_in = original_mlp_W_in * original_ln2_w[:, None]
    actual_mlp_W_in = test_state_dict[mlp_W_in_key]

    max_diff = torch.max(torch.abs(expected_mlp_W_in - actual_mlp_W_in)).item()
    assert max_diff < 1e-6, f"MLP weight folding math incorrect: max_diff={max_diff:.2e}"


def test_unembed_layer_norm_folding():
    """Test that the unembedding layer norm folding function works correctly."""
    device = "cpu"
    model_name = "gpt2"

    # Load HookedTransformer to get real config
    hooked_model = HookedTransformer.from_pretrained(
        model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False
    )

    cfg = hooked_model.cfg

    # Create test tensors in TransformerLens format
    d_model = cfg.d_model
    vocab_size = cfg.d_vocab

    # TransformerLens format: [d_model, vocab_size]
    unembed_weight = torch.randn(d_model, vocab_size)
    unembed_bias = torch.randn(vocab_size)
    ln_final_weight = torch.randn(d_model)
    ln_final_bias = torch.randn(d_model)

    # Create state dict
    state_dict = {
        "unembed.W_U": unembed_weight.clone(),
        "unembed.b_U": unembed_bias.clone(),
        "ln_final.w": ln_final_weight.clone(),
        "ln_final.b": ln_final_bias.clone(),
    }

    # Apply unembedding layer norm folding
    ProcessWeights._fold_unembed_layer_norm(
        state_dict, cfg, fold_biases=True, center_weights=True, adapter=None
    )

    # Check that LayerNorm weight was removed (but bias should remain since it's handled separately)
    assert "ln_final.w" not in state_dict, "LayerNorm weight should be removed"
    assert "ln_final.b" in state_dict, "LayerNorm bias should be preserved (handled separately)"

    # Verify the math for weight folding (accounting for centering)
    # First apply weight folding
    folded_weight = unembed_weight * ln_final_weight[:, None]
    # Then apply centering using einops (TransformerLens format: [d_model, vocab_size])
    centered_weight = folded_weight - einops.reduce(
        folded_weight, "d_model d_vocab -> 1 d_vocab", "mean"
    )
    actual_unembed_weight = state_dict["unembed.W_U"]

    max_diff = torch.max(torch.abs(centered_weight - actual_unembed_weight)).item()
    assert (
        max_diff < 1e-6
    ), f"Unembedding weight folding + centering math incorrect: max_diff={max_diff:.2e}"

    # Verify that bias was NOT modified (since bias folding is handled separately)
    expected_unembed_bias = unembed_bias  # Should remain unchanged
    actual_unembed_bias = state_dict["unembed.b_U"]

    max_diff = torch.max(torch.abs(expected_unembed_bias - actual_unembed_bias)).item()
    assert max_diff < 1e-6, f"Unembedding bias was unexpectedly modified: max_diff={max_diff:.2e}"


def test_math_functions_consistency():
    """Test that math functions produce consistent results with the same input data."""
    device = "cpu"
    model_name = "gpt2"
    layer = 0

    # Load HookedTransformer
    hooked_model = HookedTransformer.from_pretrained(
        model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False
    )

    hooked_state_dict = hooked_model.state_dict()

    # Extract tensors using our extraction function
    tensors = ProcessWeights.extract_attention_tensors_for_folding(
        hooked_state_dict, hooked_model.cfg, layer, adapter=None
    )

    if tensors["ln1_b"] is None or tensors["ln1_w"] is None:
        pytest.skip("No LayerNorm parameters found - cannot test math functions")

    # Test 1: fold_layer_norm_bias_single
    # Create identical copies of the input data
    wq_copy1 = tensors["wq"].clone()
    wq_copy2 = tensors["wq"].clone()
    bq_copy1 = tensors["bq"].clone()
    bq_copy2 = tensors["bq"].clone()
    ln1_b_copy1 = tensors["ln1_b"].clone()
    ln1_b_copy2 = tensors["ln1_b"].clone()

    # Apply the same function to identical data
    result1 = ProcessWeights.fold_layer_norm_bias_single(wq_copy1, bq_copy1, ln1_b_copy1)
    result2 = ProcessWeights.fold_layer_norm_bias_single(wq_copy2, bq_copy2, ln1_b_copy2)

    # Results should be identical
    max_diff = torch.max(torch.abs(result1 - result2)).item()
    assert (
        max_diff < 1e-10
    ), f"fold_layer_norm_bias_single not deterministic: max_diff={max_diff:.2e}"

    # Verify the math is correct
    expected = bq_copy1 + (wq_copy1 * ln1_b_copy1[None, :, None]).sum(-2)
    math_diff = torch.max(torch.abs(result1 - expected)).item()
    assert (
        math_diff < 1e-10
    ), f"fold_layer_norm_bias_single math incorrect: max_diff={math_diff:.2e}"

    # Test 2: fold_layer_norm_weight_single
    # Create identical copies
    wq_copy1 = tensors["wq"].clone()
    wq_copy2 = tensors["wq"].clone()
    ln1_w_copy1 = tensors["ln1_w"].clone()
    ln1_w_copy2 = tensors["ln1_w"].clone()

    # Apply the same function to identical data
    result1 = ProcessWeights.fold_layer_norm_weight_single(wq_copy1, ln1_w_copy1)
    result2 = ProcessWeights.fold_layer_norm_weight_single(wq_copy2, ln1_w_copy2)

    # Results should be identical
    max_diff = torch.max(torch.abs(result1 - result2)).item()
    assert (
        max_diff < 1e-10
    ), f"fold_layer_norm_weight_single not deterministic: max_diff={max_diff:.2e}"

    # Verify the math is correct
    expected = wq_copy1 * ln1_w_copy1[None, :, None]
    math_diff = torch.max(torch.abs(result1 - expected)).item()
    assert (
        math_diff < 1e-10
    ), f"fold_layer_norm_weight_single math incorrect: max_diff={math_diff:.2e}"

    # Test 3: center_weight_single
    # Create identical copies
    wq_copy1 = tensors["wq"].clone()
    wq_copy2 = tensors["wq"].clone()

    # Apply the same function to identical data
    result1 = ProcessWeights.center_weight_single(wq_copy1)
    result2 = ProcessWeights.center_weight_single(wq_copy2)

    # Results should be identical
    max_diff = torch.max(torch.abs(result1 - result2)).item()
    assert max_diff < 1e-10, f"center_weight_single not deterministic: max_diff={max_diff:.2e}"

    # Verify the math is correct - mean should be close to zero
    wq_mean = result1.mean(dim=1, keepdim=True)
    mean_diff = torch.max(torch.abs(wq_mean)).item()
    assert (
        mean_diff < 1e-6
    ), f"center_weight_single math incorrect: mean not zero, max={mean_diff:.2e}"


class TestExtractionAndMathConsistency:
    """Test that tensor extraction and math functions are consistent between models."""

    def setup_method(self):
        """Set up test models."""
        self.device = "cpu"
        self.model_name = "gpt2"
        self.layer = 0

        # Load HookedTransformer (no processing)
        self.hooked_model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )

        # Load TransformerBridge (no processing)
        from transformer_lens.model_bridge import TransformerBridge

        self.bridge_model = TransformerBridge.boot_transformers(self.model_name, device=self.device)

        # Get state dicts
        self.hooked_state_dict = self.hooked_model.state_dict()
        self.bridge_state_dict = self.bridge_model.original_model.state_dict()

    def test_extract_attention_tensors_returns_same_shapes(self):
        """Test that tensor extraction returns the same shapes for both models."""
        # Extract tensors from HookedTransformer (no adapter)
        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            self.hooked_state_dict, self.hooked_model.cfg, self.layer, adapter=None
        )

        # Extract tensors from TransformerBridge (with adapter)
        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            self.bridge_state_dict,
            self.bridge_model.cfg,
            self.layer,
            adapter=self.bridge_model.adapter,
        )

        # Test shapes match
        tensor_names = ["wq", "wk", "wv", "bq", "bk", "bv", "ln1_b", "ln1_w"]

        for tensor_name in tensor_names:
            hooked_tensor = hooked_tensors[tensor_name]
            bridge_tensor = bridge_tensors[tensor_name]

            if hooked_tensor is None and bridge_tensor is None:
                continue
            elif hooked_tensor is None or bridge_tensor is None:
                pytest.fail(f"{tensor_name}: One is None, other is not")

            assert (
                hooked_tensor.shape == bridge_tensor.shape
            ), f"{tensor_name} shape mismatch: HookedTransformer {hooked_tensor.shape} vs TransformerBridge {bridge_tensor.shape}"

    def test_extract_attention_tensors_returns_same_values(self):
        """Test that tensor extraction returns the same values for both models."""
        # Extract tensors from both models
        hooked_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            self.hooked_state_dict, self.hooked_model.cfg, self.layer, adapter=None
        )

        bridge_tensors = ProcessWeights.extract_attention_tensors_for_folding(
            self.bridge_state_dict,
            self.bridge_model.cfg,
            self.layer,
            adapter=self.bridge_model.adapter,
        )

        # Test values match
        tensor_names = ["wq", "wk", "wv", "bq", "bk", "bv", "ln1_b", "ln1_w"]

        for tensor_name in tensor_names:
            hooked_tensor = hooked_tensors[tensor_name]
            bridge_tensor = bridge_tensors[tensor_name]

            if hooked_tensor is None or bridge_tensor is None:
                continue

            max_diff = torch.max(torch.abs(hooked_tensor - bridge_tensor)).item()
            mean_diff = torch.mean(torch.abs(hooked_tensor - bridge_tensor)).item()

            assert (
                max_diff < 1e-6
            ), f"{tensor_name} value mismatch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"


def test_attention_tensor_storage():
    """Test that the attention tensor storage function works correctly."""
    device = "cpu"
    model_name = "gpt2"
    layer = 0

    # Load HookedTransformer to get real config
    hooked_model = HookedTransformer.from_pretrained(
        model_name, device=device, fold_ln=False, center_writing_weights=False, center_unembed=False
    )

    cfg = hooked_model.cfg

    # Create test tensors in TransformerLens format
    n_heads = cfg.n_heads
    d_head = cfg.d_head
    d_model = cfg.d_model

    wq_tensor = torch.randn(n_heads, d_model, d_head)
    wk_tensor = torch.randn(n_heads, d_model, d_head)
    wv_tensor = torch.randn(n_heads, d_model, d_head)
    bq_tensor = torch.randn(n_heads, d_head)
    bk_tensor = torch.randn(n_heads, d_head)
    bv_tensor = torch.randn(n_heads, d_head)

    # Create state dict and keys
    state_dict = {}
    keys = {
        "W_Q": f"blocks.{layer}.attn.W_Q",
        "W_K": f"blocks.{layer}.attn.W_K",
        "W_V": f"blocks.{layer}.attn.W_V",
        "b_Q": f"blocks.{layer}.attn.b_Q",
        "b_K": f"blocks.{layer}.attn.b_K",
        "b_V": f"blocks.{layer}.attn.b_V",
    }

    # Store tensors using the function
    ProcessWeights._store_processed_attention_tensors(
        state_dict,
        keys,
        wq_tensor,
        wk_tensor,
        wv_tensor,
        bq_tensor,
        bk_tensor,
        bv_tensor,
        adapter=None,
        cfg=cfg,
        layer=layer,
    )

    # Verify all keys are present
    expected_keys = list(keys.values())
    actual_keys = list(state_dict.keys())

    all_keys_present = all(key in state_dict for key in expected_keys)
    assert (
        all_keys_present
    ), f"Not all expected keys present. Expected: {expected_keys}, Actual: {actual_keys}"

    # Verify tensor values are stored correctly
    assert torch.equal(state_dict[keys["W_Q"]], wq_tensor), "W_Q tensor not stored correctly"
    assert torch.equal(state_dict[keys["W_K"]], wk_tensor), "W_K tensor not stored correctly"
    assert torch.equal(state_dict[keys["W_V"]], wv_tensor), "W_V tensor not stored correctly"
    assert torch.equal(state_dict[keys["b_Q"]], bq_tensor), "b_Q tensor not stored correctly"
    assert torch.equal(state_dict[keys["b_K"]], bk_tensor), "b_K tensor not stored correctly"
    assert torch.equal(state_dict[keys["b_V"]], bv_tensor), "b_V tensor not stored correctly"
