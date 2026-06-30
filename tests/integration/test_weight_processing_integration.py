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
        return HookedTransformer.from_pretrained("distilgpt2")

    @pytest.fixture
    def gpt2_small_adapter(self):
        """Create adapter for GPT-2 Small model."""
        from transformer_lens.model_bridge import TransformerBridge

        # Use the proper way to get an adapter by creating a bridge and accessing its adapter
        bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
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
        """Folding LN bias adds the W-projected LN bias to each attention bias."""
        wq_tensor, wk_tensor, wv_tensor = sample_tensors["weights"]
        bq_tensor, bk_tensor, bv_tensor = sample_tensors["biases"]
        ln_bias = sample_tensors["ln_bias"]

        new_bq, new_bk, new_bv = ProcessWeights.fold_layer_norm_biases(
            wq_tensor, wk_tensor, wv_tensor, bq_tensor, bk_tensor, bv_tensor, ln_bias
        )

        assert new_bq.shape == bq_tensor.shape
        assert new_bk.shape == bk_tensor.shape
        assert new_bv.shape == bv_tensor.shape

        # Effect: new_b[h, j] == b[h, j] + sum_i W[h, i, j] * ln_bias[i].
        # Use einsum as an independent reference (different op + axis spec than the
        # implementation's (w * ln_bias[None, :, None]).sum(-2)), so a wrong axis or
        # broadcast in the impl diverges from this expected value.
        for w_tensor, b_tensor, new_b in (
            (wq_tensor, bq_tensor, new_bq),
            (wk_tensor, bk_tensor, new_bk),
            (wv_tensor, bv_tensor, new_bv),
        ):
            expected = b_tensor + torch.einsum("hij,i->hj", w_tensor, ln_bias)
            # The einsum reduces the d_model sum in a different order than the impl's
            # .sum(-2), so allow fp32 summation-order slack (a wrong axis is order-1 off).
            torch.testing.assert_close(new_b, expected, atol=1e-4, rtol=1e-3)

        # Independent scalar spot-check on one (head, d_head) entry: a plain dot product
        # over d_model, no broadcasting, to pin down the contraction axis.
        h, j = 3, 7
        manual = bq_tensor[h, j] + (wq_tensor[h, :, j] * ln_bias).sum()
        torch.testing.assert_close(new_bq[h, j], manual, atol=1e-4, rtol=1e-3)

        # Centering/zero-bias guard: with zero LN bias the attention biases are unchanged.
        zero_bq, zero_bk, zero_bv = ProcessWeights.fold_layer_norm_biases(
            wq_tensor,
            wk_tensor,
            wv_tensor,
            bq_tensor,
            bk_tensor,
            bv_tensor,
            torch.zeros_like(ln_bias),
        )
        torch.testing.assert_close(zero_bq, bq_tensor)
        torch.testing.assert_close(zero_bk, bk_tensor)
        torch.testing.assert_close(zero_bv, bv_tensor)

    def test_fold_layer_norm_weights_consistency(self, sample_tensors):
        """Folding LN weight scales each d_model input row of W by ln_weight[i]."""
        wq_tensor, wk_tensor, wv_tensor = sample_tensors["weights"]
        ln_weight = sample_tensors["ln_weight"]

        new_wq, new_wk, new_wv = ProcessWeights.fold_layer_norm_weights(
            wq_tensor, wk_tensor, wv_tensor, ln_weight
        )

        assert new_wq.shape == wq_tensor.shape
        assert new_wk.shape == wk_tensor.shape
        assert new_wv.shape == wv_tensor.shape

        # Effect: new_W[h, i, j] == W[h, i, j] * ln_weight[i], i.e. the scaling indexes
        # the d_model axis (axis 1), NOT d_head. Use einsum as an independent reference
        # (different op than the impl's ln_weight[None, :, None] broadcast); a wrong
        # broadcast index (e.g. scaling d_head) diverges from this expected value.
        for w_tensor, new_w in (
            (wq_tensor, new_wq),
            (wk_tensor, new_wk),
            (wv_tensor, new_wv),
        ):
            expected = torch.einsum("hij,i->hij", w_tensor, ln_weight)
            torch.testing.assert_close(new_w, expected)

        # Independent guard distinguishing the d_model axis from d_head: scaling a single
        # d_model row by ln_weight[i] must multiply that whole row, leaving others intact.
        i = 5
        torch.testing.assert_close(new_wq[:, i, :], wq_tensor[:, i, :] * ln_weight[i])

        # Unit LN weight is the identity on W.
        ident_wq, _, _ = ProcessWeights.fold_layer_norm_weights(
            wq_tensor, wk_tensor, wv_tensor, torch.ones_like(ln_weight)
        )
        torch.testing.assert_close(ident_wq, wq_tensor)

    def test_center_attention_weights_consistency(self, sample_tensors):
        """Centering subtracts the per-(head, d_head) mean over d_model, making it zero."""
        wq_tensor, wk_tensor, wv_tensor = sample_tensors["weights"]

        centered_wq, centered_wk, centered_wv = ProcessWeights.center_attention_weights(
            wq_tensor, wk_tensor, wv_tensor
        )

        assert centered_wq.shape == wq_tensor.shape
        assert centered_wk.shape == wk_tensor.shape
        assert centered_wv.shape == wv_tensor.shape

        for w_tensor, centered in (
            (wq_tensor, centered_wq),
            (wk_tensor, centered_wk),
            (wv_tensor, centered_wv),
        ):
            # Defining effect: mean over d_model (axis 1) is zero for every (head, d_head).
            # This is computed without einops.reduce, so a wrong reduction axis in the
            # impl leaves a nonzero mean here and fails. A bug collapsing d_head instead
            # would also fail this.
            torch.testing.assert_close(
                centered.mean(dim=1),
                torch.zeros(w_tensor.shape[0], w_tensor.shape[2]),
                atol=1e-6,
                rtol=0,
            )
            # The removed component is constant across d_model (a pure column mean), so
            # the residual w - centered equals the broadcast per-(head, d_head) mean.
            expected_mean = w_tensor.mean(dim=1, keepdim=True)
            torch.testing.assert_close(w_tensor - centered, expected_mean.expand_as(w_tensor))

        # Centering is idempotent: re-centering an already-centered weight is a no-op.
        recentered_wq, _, _ = ProcessWeights.center_attention_weights(
            centered_wq, centered_wk, centered_wv
        )
        torch.testing.assert_close(recentered_wq, centered_wq)

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

    @pytest.mark.skip(
        reason="Weight processing format consistency failing due to architectural differences"
    )
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
