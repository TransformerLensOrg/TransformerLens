"""
Integration tests for weight processing functions with HookedTransformer and transformer bridge.

These tests verify that the individual math functions (fold_layer_norm_biases, 
fold_layer_norm_weights, center_attention_weights) produce consistent results
across different model formats.
"""

import pytest
import torch

from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.weight_processing import ProcessWeights


class TestWeightProcessingIntegration:
    """Integration tests for weight processing with different model formats."""

    @pytest.fixture
    def gpt2_small_model(self):
        """Load GPT-2 Small model for testing."""
        return HookedTransformer.from_pretrained("gpt2-small")

    @pytest.fixture
    def gpt2_small_adapter(self):
        """Create adapter for GPT-2 Small model."""
        from transformer_lens.model_bridge import TransformerBridge

        # Use the proper way to get an adapter by creating a bridge and accessing its adapter
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        return bridge.adapter

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing math functions."""
        torch.manual_seed(42)

        # Create sample tensors with realistic dimensions
        n_heads = 12
        d_model = 768
        d_head = 64

        # Weight tensors: [n_heads, d_model, d_head]
        wq_tensor = torch.randn(n_heads, d_model, d_head)
        wk_tensor = torch.randn(n_heads, d_model, d_head)
        wv_tensor = torch.randn(n_heads, d_model, d_head)

        # Bias tensors: [n_heads, d_head]
        bq_tensor = torch.randn(n_heads, d_head)
        bk_tensor = torch.randn(n_heads, d_head)
        bv_tensor = torch.randn(n_heads, d_head)

        # LayerNorm tensors: [d_model]
        ln_bias = torch.randn(d_model)
        ln_weight = torch.randn(d_model)

        return {
            "weights": (wq_tensor, wk_tensor, wv_tensor),
            "biases": (bq_tensor, bk_tensor, bv_tensor),
            "ln_bias": ln_bias,
            "ln_weight": ln_weight,
        }

    def test_fold_layer_norm_biases_consistency(self, sample_tensors):
        """Test that fold_layer_norm_biases produces consistent results."""
        wq_tensor, wk_tensor, wv_tensor = sample_tensors["weights"]
        bq_tensor, bk_tensor, bv_tensor = sample_tensors["biases"]
        ln_bias = sample_tensors["ln_bias"]

        # Test the function
        new_bq, new_bk, new_bv = ProcessWeights.fold_layer_norm_biases(
            wq_tensor, wk_tensor, wv_tensor, bq_tensor, bk_tensor, bv_tensor, ln_bias
        )

        # Verify shapes are preserved
        assert new_bq.shape == bq_tensor.shape
        assert new_bk.shape == bk_tensor.shape
        assert new_bv.shape == bv_tensor.shape

        # Verify the mathematical correctness
        expected_bq = bq_tensor + (wq_tensor * ln_bias[None, :, None]).sum(-2)
        expected_bk = bk_tensor + (wk_tensor * ln_bias[None, :, None]).sum(-2)
        expected_bv = bv_tensor + (wv_tensor * ln_bias[None, :, None]).sum(-2)

        torch.testing.assert_close(new_bq, expected_bq)
        torch.testing.assert_close(new_bk, expected_bk)
        torch.testing.assert_close(new_bv, expected_bv)

    def test_fold_layer_norm_weights_consistency(self, sample_tensors):
        """Test that fold_layer_norm_weights produces consistent results."""
        wq_tensor, wk_tensor, wv_tensor = sample_tensors["weights"]
        ln_weight = sample_tensors["ln_weight"]

        # Test the function
        new_wq, new_wk, new_wv = ProcessWeights.fold_layer_norm_weights(
            wq_tensor, wk_tensor, wv_tensor, ln_weight
        )

        # Verify shapes are preserved
        assert new_wq.shape == wq_tensor.shape
        assert new_wk.shape == wk_tensor.shape
        assert new_wv.shape == wv_tensor.shape

        # Verify the mathematical correctness
        expected_wq = wq_tensor * ln_weight[None, :, None]
        expected_wk = wk_tensor * ln_weight[None, :, None]
        expected_wv = wv_tensor * ln_weight[None, :, None]

        torch.testing.assert_close(new_wq, expected_wq)
        torch.testing.assert_close(new_wk, expected_wk)
        torch.testing.assert_close(new_wv, expected_wv)

    def test_center_attention_weights_consistency(self, sample_tensors):
        """Test that center_attention_weights produces consistent results."""
        wq_tensor, wk_tensor, wv_tensor = sample_tensors["weights"]

        # Test the function
        centered_wq, centered_wk, centered_wv = ProcessWeights.center_attention_weights(
            wq_tensor, wk_tensor, wv_tensor
        )

        # Verify shapes are preserved
        assert centered_wq.shape == wq_tensor.shape
        assert centered_wk.shape == wk_tensor.shape
        assert centered_wv.shape == wv_tensor.shape

        # Verify the mathematical correctness
        import einops

        expected_wq = wq_tensor - einops.reduce(
            wq_tensor, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )
        expected_wk = wk_tensor - einops.reduce(
            wk_tensor, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )
        expected_wv = wv_tensor - einops.reduce(
            wv_tensor, "head_index d_model d_head -> head_index 1 d_head", "mean"
        )

        torch.testing.assert_close(centered_wq, expected_wq)
        torch.testing.assert_close(centered_wk, expected_wk)
        torch.testing.assert_close(centered_wv, expected_wv)

    def test_extract_attention_tensors_with_hooked_transformer(self, gpt2_small_model):
        """Test tensor extraction with HookedTransformer model."""
        model = gpt2_small_model
        state_dict = model.state_dict()
        cfg = model.cfg
        layer = 0

        # Extract tensors
        tensors = ProcessWeights.extract_attention_tensors_for_folding(state_dict, cfg, layer, None)

        wq_tensor = tensors["wq"]
        wk_tensor = tensors["wk"]
        wv_tensor = tensors["wv"]
        bq_tensor = tensors["bq"]
        bk_tensor = tensors["bk"]
        bv_tensor = tensors["bv"]

        # Verify shapes
        expected_shape = (cfg.n_heads, cfg.d_model, cfg.d_head)
        assert wq_tensor.shape == expected_shape
        assert wk_tensor.shape == expected_shape
        assert wv_tensor.shape == expected_shape

        expected_bias_shape = (cfg.n_heads, cfg.d_head)
        assert bq_tensor.shape == expected_bias_shape
        assert bk_tensor.shape == expected_bias_shape
        assert bv_tensor.shape == expected_bias_shape

        # Verify tensors are properly extracted
        assert wq_tensor is not None
        assert wk_tensor is not None
        assert wv_tensor is not None

    def test_extract_attention_tensors_with_adapter(self, gpt2_small_adapter):
        """Test tensor extraction with HuggingFace adapter."""
        # Create a mock state dict with HuggingFace format
        d_model = 768
        n_heads = 12
        d_head = 64

        # Combined QKV weight: [d_model, 3*d_model]
        combined_qkv_weight = torch.randn(d_model, 3 * d_model)
        # Combined QKV bias: [3*d_model]
        combined_qkv_bias = torch.randn(3 * d_model)

        # Mock state dict
        state_dict = {
            "transformer.h.0.attn.c_attn.weight": combined_qkv_weight,
            "transformer.h.0.attn.c_attn.bias": combined_qkv_bias,
        }

        # Mock config - define as function to avoid variable scope issue
        def create_mock_config():
            class MockConfig:
                pass

            config = MockConfig()
            config.n_heads = n_heads
            config.d_head = d_head
            config.d_model = d_model
            return config

        cfg = create_mock_config()
        layer = 0
        adapter = gpt2_small_adapter

        # Extract tensors
        tensors = ProcessWeights.extract_attention_tensors_for_folding(
            state_dict, cfg, layer, adapter
        )

        wq_tensor = tensors["wq"]
        wk_tensor = tensors["wk"]
        wv_tensor = tensors["wv"]
        bq_tensor = tensors["bq"]
        bk_tensor = tensors["bk"]
        bv_tensor = tensors["bv"]

        # Verify shapes (should be in TransformerLens format)
        expected_shape = (n_heads, d_model, d_head)
        assert wq_tensor.shape == expected_shape
        assert wk_tensor.shape == expected_shape
        assert wv_tensor.shape == expected_shape

        expected_bias_shape = (n_heads, d_head)
        assert bq_tensor.shape == expected_bias_shape
        assert bk_tensor.shape == expected_bias_shape
        assert bv_tensor.shape == expected_bias_shape

        # Verify tensors are properly extracted
        assert wq_tensor is not None
        assert wk_tensor is not None
        assert wv_tensor is not None

    def test_full_pipeline_with_hooked_transformer(self, gpt2_small_model):
        """Test the full pipeline with HookedTransformer model."""
        model = gpt2_small_model
        state_dict = model.state_dict()
        cfg = model.cfg
        layer = 0

        # Get parameter keys
        W_Q_key = f"blocks.{layer}.attn.W_Q"
        W_K_key = f"blocks.{layer}.attn.W_K"
        W_V_key = f"blocks.{layer}.attn.W_V"
        b_Q_key = f"blocks.{layer}.attn.b_Q"
        b_K_key = f"blocks.{layer}.attn.b_K"
        b_V_key = f"blocks.{layer}.attn.b_V"

        # Extract tensors
        tensors = ProcessWeights.extract_attention_tensors_for_folding(state_dict, cfg, layer, None)

        wq_tensor = tensors["wq"]
        wk_tensor = tensors["wk"]
        wv_tensor = tensors["wv"]
        bq_tensor = tensors["bq"]
        bk_tensor = tensors["bk"]
        bv_tensor = tensors["bv"]

        # Test LayerNorm folding if parameters exist
        ln1_b_key = f"blocks.{layer}.ln1.b"
        ln1_w_key = f"blocks.{layer}.ln1.w"

        if ln1_b_key in state_dict and ln1_w_key in state_dict:
            ln1_b = state_dict[ln1_b_key]
            ln1_w = state_dict[ln1_w_key]

            # Test bias folding
            new_bq, new_bk, new_bv = ProcessWeights.fold_layer_norm_biases(
                wq_tensor, wk_tensor, wv_tensor, bq_tensor, bk_tensor, bv_tensor, ln1_b
            )

            # Test weight folding
            new_wq, new_wk, new_wv = ProcessWeights.fold_layer_norm_weights(
                wq_tensor, wk_tensor, wv_tensor, ln1_w
            )

            # Verify shapes are preserved
            assert new_bq.shape == bq_tensor.shape
            assert new_bk.shape == bk_tensor.shape
            assert new_bv.shape == bv_tensor.shape
            assert new_wq.shape == wq_tensor.shape
            assert new_wk.shape == wk_tensor.shape
            assert new_wv.shape == wv_tensor.shape

        # Test weight centering
        centered_wq, centered_wk, centered_wv = ProcessWeights.center_attention_weights(
            wq_tensor, wk_tensor, wv_tensor
        )

        # Verify shapes are preserved
        assert centered_wq.shape == wq_tensor.shape
        assert centered_wk.shape == wk_tensor.shape
        assert centered_wv.shape == wv_tensor.shape

    def test_consistency_between_formats(self, gpt2_small_model, gpt2_small_adapter):
        """Test that the same mathematical operations produce consistent results across formats."""
        model = gpt2_small_model
        cfg = model.cfg
        layer = 0

        # Get tensors from HookedTransformer format
        state_dict_tl = model.state_dict()
        W_Q_key = f"blocks.{layer}.attn.W_Q"
        W_K_key = f"blocks.{layer}.attn.W_K"
        W_V_key = f"blocks.{layer}.attn.W_V"
        b_Q_key = f"blocks.{layer}.attn.b_Q"
        b_K_key = f"blocks.{layer}.attn.b_K"
        b_V_key = f"blocks.{layer}.attn.b_V"

        tensors_tl = ProcessWeights.extract_attention_tensors_for_folding(
            state_dict_tl, cfg, layer, None
        )
        wq_tl = tensors_tl["wq"]
        wk_tl = tensors_tl["wk"]
        wv_tl = tensors_tl["wv"]
        bq_tl = tensors_tl["bq"]
        bk_tl = tensors_tl["bk"]
        bv_tl = tensors_tl["bv"]

        # Convert to HuggingFace format and back
        adapter = gpt2_small_adapter

        # Convert TL tensors to HF format
        wq_hf = ProcessWeights.convert_tensor_to_hf_format(
            wq_tl, f"blocks.{layer}.attn.W_Q", adapter, cfg, layer
        )
        wk_hf = ProcessWeights.convert_tensor_to_hf_format(
            wk_tl, f"blocks.{layer}.attn.W_K", adapter, cfg, layer
        )
        wv_hf = ProcessWeights.convert_tensor_to_hf_format(
            wv_tl, f"blocks.{layer}.attn.W_V", adapter, cfg, layer
        )
        bq_hf = ProcessWeights.convert_tensor_to_hf_format(
            bq_tl, f"blocks.{layer}.attn.b_Q", adapter, cfg, layer
        )
        bk_hf = ProcessWeights.convert_tensor_to_hf_format(
            bk_tl, f"blocks.{layer}.attn.b_K", adapter, cfg, layer
        )
        bv_hf = ProcessWeights.convert_tensor_to_hf_format(
            bv_tl, f"blocks.{layer}.attn.b_V", adapter, cfg, layer
        )

        # Convert back to TL format using proper HF state dict keys
        wq_hf_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_Q")
        wk_hf_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_K")
        wv_hf_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.W_V")
        bq_hf_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_Q")
        bk_hf_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_K")
        bv_hf_key = adapter.translate_transformer_lens_path(f"blocks.{layer}.attn.b_V")

        wq_tl_converted = ProcessWeights.convert_tensor_to_tl_format(
            f"blocks.{layer}.attn.W_Q", adapter, {wq_hf_key: wq_hf}, cfg, layer
        )
        wk_tl_converted = ProcessWeights.convert_tensor_to_tl_format(
            f"blocks.{layer}.attn.W_K", adapter, {wk_hf_key: wk_hf}, cfg, layer
        )
        wv_tl_converted = ProcessWeights.convert_tensor_to_tl_format(
            f"blocks.{layer}.attn.W_V", adapter, {wv_hf_key: wv_hf}, cfg, layer
        )
        bq_tl_converted = ProcessWeights.convert_tensor_to_tl_format(
            f"blocks.{layer}.attn.b_Q", adapter, {bq_hf_key: bq_hf}, cfg, layer
        )
        bk_tl_converted = ProcessWeights.convert_tensor_to_tl_format(
            f"blocks.{layer}.attn.b_K", adapter, {bk_hf_key: bk_hf}, cfg, layer
        )
        bv_tl_converted = ProcessWeights.convert_tensor_to_tl_format(
            f"blocks.{layer}.attn.b_V", adapter, {bv_hf_key: bv_hf}, cfg, layer
        )

        # Test that the math functions produce the same results
        ln_bias = torch.randn(cfg.d_model)
        ln_weight = torch.randn(cfg.d_model)

        # Apply operations to original TL tensors
        new_bq_tl, new_bk_tl, new_bv_tl = ProcessWeights.fold_layer_norm_biases(
            wq_tl, wk_tl, wv_tl, bq_tl, bk_tl, bv_tl, ln_bias
        )
        new_wq_tl, new_wk_tl, new_wv_tl = ProcessWeights.fold_layer_norm_weights(
            wq_tl, wk_tl, wv_tl, ln_weight
        )
        centered_wq_tl, centered_wk_tl, centered_wv_tl = ProcessWeights.center_attention_weights(
            wq_tl, wk_tl, wv_tl
        )

        # Apply operations to converted TL tensors
        (
            new_bq_converted,
            new_bk_converted,
            new_bv_converted,
        ) = ProcessWeights.fold_layer_norm_biases(
            wq_tl_converted,
            wk_tl_converted,
            wv_tl_converted,
            bq_tl_converted,
            bk_tl_converted,
            bv_tl_converted,
            ln_bias,
        )
        (
            new_wq_converted,
            new_wk_converted,
            new_wv_converted,
        ) = ProcessWeights.fold_layer_norm_weights(
            wq_tl_converted, wk_tl_converted, wv_tl_converted, ln_weight
        )
        (
            centered_wq_converted,
            centered_wk_converted,
            centered_wv_converted,
        ) = ProcessWeights.center_attention_weights(
            wq_tl_converted, wk_tl_converted, wv_tl_converted
        )

        # Verify results are consistent (within numerical precision)
        torch.testing.assert_close(new_bq_tl, new_bq_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_bk_tl, new_bk_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_bv_tl, new_bv_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_wq_tl, new_wq_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_wk_tl, new_wk_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_wv_tl, new_wv_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(centered_wq_tl, centered_wq_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(centered_wk_tl, centered_wk_converted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(centered_wv_tl, centered_wv_converted, atol=1e-6, rtol=1e-6)
