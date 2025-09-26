#!/usr/bin/env python3
"""
Unit tests for reversible weight converter.

This module contains comprehensive unit tests for the HF ⇄ TransformerLens
weight conversion system with round-trip guarantees.
"""

import unittest
from unittest.mock import Mock, patch
import warnings

import torch
import numpy as np

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.conversion_utils.reversible_weight_converter import (
    ReversibleWeightConverter,
    EmbeddingConverter,
    AttentionConverter,
    MLPConverter,
    NormalizationConverter,
    UnembeddingConverter,
    ConversionError,
    RoundTripError
)


class TestEmbeddingConverter(unittest.TestCase):
    """Test cases for EmbeddingConverter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = EmbeddingConverter()
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_vocab=50257,
            act_fn="gelu",
            normalization_type="LN"
        )

    def test_gpt2_embedding_conversion(self):
        """Test GPT-2 style embedding conversion."""
        # Create mock HF weights
        hf_weights = {
            "transformer.wte.weight": torch.randn(50257, 768),
            "transformer.wpe.weight": torch.randn(1024, 768)
        }

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, self.config)

        # Check expected keys
        self.assertIn("embed.W_E", tlens_weights)
        self.assertIn("pos_embed.W_pos", tlens_weights)

        # Check shapes
        self.assertEqual(tlens_weights["embed.W_E"].shape, (50257, 768))
        self.assertEqual(tlens_weights["pos_embed.W_pos"].shape, (1024, 768))

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, self.config, model_type="gpt2")

        # Check round-trip
        self.assertTrue(torch.equal(hf_weights["transformer.wte.weight"], recovered_hf["transformer.wte.weight"]))
        self.assertTrue(torch.equal(hf_weights["transformer.wpe.weight"], recovered_hf["transformer.wpe.weight"]))

    def test_llama_embedding_conversion(self):
        """Test LLaMA style embedding conversion."""
        # Create mock HF weights
        hf_weights = {
            "model.embed_tokens.weight": torch.randn(32000, 4096)
        }

        config = HookedTransformerConfig(
            d_model=4096,
            n_heads=32,
            n_layers=32,
            n_ctx=2048,
            d_vocab=32000,
            act_fn="silu",
            normalization_type="RMS"
        )

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, config, model_type="llama")

        # Check expected keys
        self.assertIn("embed.W_E", tlens_weights)
        self.assertNotIn("pos_embed.W_pos", tlens_weights)  # LLaMA doesn't use position embeddings

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, config, model_type="llama")

        # Check round-trip
        self.assertTrue(torch.equal(hf_weights["model.embed_tokens.weight"], recovered_hf["model.embed_tokens.weight"]))

    def test_round_trip_validation(self):
        """Test round-trip validation method."""
        hf_weights = {
            "transformer.wte.weight": torch.randn(50257, 768),
            "transformer.wpe.weight": torch.randn(1024, 768)
        }

        # Should pass validation
        self.assertTrue(self.converter.validate_round_trip(hf_weights, self.config, model_type="gpt2"))

        # Test with corrupted weights
        hf_weights_corrupted = hf_weights.copy()
        hf_weights_corrupted["transformer.wte.weight"] = torch.randn(50257, 768) * 1000  # Different values

        with self.assertRaises(RoundTripError):
            self.converter.validate_round_trip(hf_weights_corrupted, self.config, model_type="gpt2")


class TestAttentionConverter(unittest.TestCase):
    """Test cases for AttentionConverter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = AttentionConverter()
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_vocab=50257,
            d_head=64,
            act_fn="gelu",
            normalization_type="LN"
        )

    def test_gpt2_attention_conversion(self):
        """Test GPT-2 style attention conversion."""
        layer_idx = 0

        # Create mock HF weights
        hf_weights = {
            f"transformer.h.{layer_idx}.attn.c_attn.weight": torch.randn(768, 2304),  # 3 * 768
            f"transformer.h.{layer_idx}.attn.c_attn.bias": torch.randn(2304),
            f"transformer.h.{layer_idx}.attn.c_proj.weight": torch.randn(768, 768),
            f"transformer.h.{layer_idx}.attn.c_proj.bias": torch.randn(768)
        }

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, self.config, layer_idx=layer_idx, model_type="gpt2")

        # Check expected keys
        expected_keys = [f"blocks.{layer_idx}.attn.W_Q", f"blocks.{layer_idx}.attn.W_K",
                        f"blocks.{layer_idx}.attn.W_V", f"blocks.{layer_idx}.attn.W_O",
                        f"blocks.{layer_idx}.attn.b_Q", f"blocks.{layer_idx}.attn.b_K",
                        f"blocks.{layer_idx}.attn.b_V", f"blocks.{layer_idx}.attn.b_O"]

        for key in expected_keys:
            self.assertIn(key, tlens_weights)

        # Check shapes
        self.assertEqual(tlens_weights[f"blocks.{layer_idx}.attn.W_Q"].shape, (12, 768, 64))
        self.assertEqual(tlens_weights[f"blocks.{layer_idx}.attn.b_Q"].shape, (12, 64))

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, self.config, layer_idx=layer_idx, model_type="gpt2")

        # Check round-trip (with tolerance due to reshaping operations)
        for key in hf_weights:
            self.assertTrue(
                torch.allclose(hf_weights[key], recovered_hf[key], atol=1e-6),
                f"Mismatch for key {key}"
            )

    def test_llama_attention_conversion(self):
        """Test LLaMA style attention conversion."""
        layer_idx = 0

        config = HookedTransformerConfig(
            d_model=4096,
            n_heads=32,
            n_layers=32,
            n_ctx=2048,
            d_vocab=32000,
            d_head=128,
            act_fn="silu",
            normalization_type="RMS"
        )

        # Create mock HF weights
        hf_weights = {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": torch.randn(4096, 4096),
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": torch.randn(4096, 4096),
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": torch.randn(4096, 4096),
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": torch.randn(4096, 4096)
        }

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, config, layer_idx=layer_idx, model_type="llama")

        # Check expected keys
        expected_keys = [f"blocks.{layer_idx}.attn.W_Q", f"blocks.{layer_idx}.attn.W_K",
                        f"blocks.{layer_idx}.attn.W_V", f"blocks.{layer_idx}.attn.W_O"]

        for key in expected_keys:
            self.assertIn(key, tlens_weights)

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, config, layer_idx=layer_idx, model_type="llama")

        # Check round-trip
        for key in hf_weights:
            self.assertTrue(
                torch.allclose(hf_weights[key], recovered_hf[key], atol=1e-6),
                f"Mismatch for key {key}"
            )


class TestMLPConverter(unittest.TestCase):
    """Test cases for MLPConverter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = MLPConverter()
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_vocab=50257,
            d_mlp=3072,
            act_fn="gelu",
            normalization_type="LN"
        )

    def test_gpt2_mlp_conversion(self):
        """Test GPT-2 style MLP conversion."""
        layer_idx = 0

        # Create mock HF weights
        hf_weights = {
            f"transformer.h.{layer_idx}.mlp.c_fc.weight": torch.randn(768, 3072),
            f"transformer.h.{layer_idx}.mlp.c_fc.bias": torch.randn(3072),
            f"transformer.h.{layer_idx}.mlp.c_proj.weight": torch.randn(3072, 768),
            f"transformer.h.{layer_idx}.mlp.c_proj.bias": torch.randn(768)
        }

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, self.config, layer_idx=layer_idx, model_type="gpt2")

        # Check expected keys
        expected_keys = [f"blocks.{layer_idx}.mlp.W_in", f"blocks.{layer_idx}.mlp.b_in",
                        f"blocks.{layer_idx}.mlp.W_out", f"blocks.{layer_idx}.mlp.b_out"]

        for key in expected_keys:
            self.assertIn(key, tlens_weights)

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, self.config, layer_idx=layer_idx, model_type="gpt2")

        # Check round-trip
        for key in hf_weights:
            self.assertTrue(
                torch.equal(hf_weights[key], recovered_hf[key]),
                f"Mismatch for key {key}"
            )

    def test_llama_mlp_conversion(self):
        """Test LLaMA style MLP conversion (SwiGLU)."""
        layer_idx = 0

        config = HookedTransformerConfig(
            d_model=4096,
            n_heads=32,
            n_layers=32,
            n_ctx=2048,
            d_vocab=32000,
            d_mlp=11008,
            act_fn="silu",
            normalization_type="RMS"
        )

        # Create mock HF weights
        hf_weights = {
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": torch.randn(11008, 4096),
            f"model.layers.{layer_idx}.mlp.up_proj.weight": torch.randn(11008, 4096),
            f"model.layers.{layer_idx}.mlp.down_proj.weight": torch.randn(4096, 11008)
        }

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, config, layer_idx=layer_idx, model_type="llama")

        # Check expected keys
        expected_keys = [f"blocks.{layer_idx}.mlp.W_gate", f"blocks.{layer_idx}.mlp.W_in",
                        f"blocks.{layer_idx}.mlp.W_out"]

        for key in expected_keys:
            self.assertIn(key, tlens_weights)

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, config, layer_idx=layer_idx, model_type="llama")

        # Check round-trip
        for key in hf_weights:
            self.assertTrue(
                torch.equal(hf_weights[key], recovered_hf[key]),
                f"Mismatch for key {key}"
            )


class TestNormalizationConverter(unittest.TestCase):
    """Test cases for NormalizationConverter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = NormalizationConverter()
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_vocab=50257,
            act_fn="gelu",
            normalization_type="LN"
        )

    def test_gpt2_layernorm_conversion(self):
        """Test GPT-2 style LayerNorm conversion."""
        layer_idx = 0

        # Test layer norms
        for norm_type in ["ln1", "ln2"]:
            hf_norm_name = "ln_1" if norm_type == "ln1" else "ln_2"
            hf_weights = {
                f"transformer.h.{layer_idx}.{hf_norm_name}.weight": torch.randn(768),
                f"transformer.h.{layer_idx}.{hf_norm_name}.bias": torch.randn(768)
            }

            # Convert HF → TLens
            tlens_weights = self.converter.hf_to_tlens(
                hf_weights, self.config, layer_idx=layer_idx, norm_type=norm_type, model_type="gpt2"
            )

            # Check expected keys
            expected_keys = [f"blocks.{layer_idx}.{norm_type}.w", f"blocks.{layer_idx}.{norm_type}.b"]
            for key in expected_keys:
                self.assertIn(key, tlens_weights)

            # Convert back TLens → HF
            recovered_hf = self.converter.tlens_to_hf(
                tlens_weights, self.config, layer_idx=layer_idx, norm_type=norm_type, model_type="gpt2"
            )

            # Check round-trip
            for key in hf_weights:
                self.assertTrue(torch.equal(hf_weights[key], recovered_hf[key]))

        # Test final norm
        hf_weights = {
            "transformer.ln_f.weight": torch.randn(768),
            "transformer.ln_f.bias": torch.randn(768)
        }

        tlens_weights = self.converter.hf_to_tlens(
            hf_weights, self.config, norm_type="final", model_type="gpt2"
        )

        self.assertIn("ln_final.w", tlens_weights)
        self.assertIn("ln_final.b", tlens_weights)

    def test_llama_rmsnorm_conversion(self):
        """Test LLaMA style RMSNorm conversion."""
        layer_idx = 0

        config = HookedTransformerConfig(
            d_model=4096,
            n_heads=32,
            n_layers=32,
            n_ctx=2048,
            d_vocab=32000,
            act_fn="silu",
            normalization_type="RMS"
        )

        # Test layer norms
        for norm_type in ["ln1", "ln2"]:
            if norm_type == "ln1":
                hf_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            else:
                hf_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"

            hf_weights = {hf_key: torch.randn(4096)}

            # Convert HF → TLens
            tlens_weights = self.converter.hf_to_tlens(
                hf_weights, config, layer_idx=layer_idx, norm_type=norm_type, model_type="llama"
            )

            # Check expected key (no bias for RMSNorm)
            expected_key = f"blocks.{layer_idx}.{norm_type}.w"
            self.assertIn(expected_key, tlens_weights)

            # Convert back TLens → HF
            recovered_hf = self.converter.tlens_to_hf(
                tlens_weights, config, layer_idx=layer_idx, norm_type=norm_type, model_type="llama"
            )

            # Check round-trip
            self.assertTrue(torch.equal(hf_weights[hf_key], recovered_hf[hf_key]))


class TestUnembeddingConverter(unittest.TestCase):
    """Test cases for UnembeddingConverter."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = UnembeddingConverter()
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_vocab=50257,
            act_fn="gelu",
            normalization_type="LN"
        )

    def test_gpt2_unembedding_conversion(self):
        """Test GPT-2 style unembedding conversion."""
        # Create mock HF weights
        hf_weights = {
            "lm_head.weight": torch.randn(50257, 768),
            "lm_head.bias": torch.randn(50257)
        }

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, self.config, model_type="gpt2")

        # Check expected keys
        self.assertIn("unembed.W_U", tlens_weights)
        self.assertIn("unembed.b_U", tlens_weights)

        # Check shapes (note transpose)
        self.assertEqual(tlens_weights["unembed.W_U"].shape, (768, 50257))
        self.assertEqual(tlens_weights["unembed.b_U"].shape, (50257,))

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, self.config, model_type="gpt2")

        # Check round-trip
        self.assertTrue(torch.equal(hf_weights["lm_head.weight"], recovered_hf["lm_head.weight"]))
        self.assertTrue(torch.equal(hf_weights["lm_head.bias"], recovered_hf["lm_head.bias"]))

    def test_llama_unembedding_conversion(self):
        """Test LLaMA style unembedding conversion (no bias)."""
        config = HookedTransformerConfig(
            d_model=4096,
            n_heads=32,
            n_layers=32,
            n_ctx=2048,
            d_vocab=32000,
            act_fn="silu",
            normalization_type="RMS"
        )

        # Create mock HF weights (no bias)
        hf_weights = {
            "model.lm_head.weight": torch.randn(32000, 4096)
        }

        # Convert HF → TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, config, model_type="llama")

        # Check expected keys
        self.assertIn("unembed.W_U", tlens_weights)
        self.assertIn("unembed.b_U", tlens_weights)  # Should create zero bias

        # Check zero bias
        self.assertTrue(torch.allclose(tlens_weights["unembed.b_U"], torch.zeros(32000)))

        # Convert back TLens → HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, config, model_type="llama")

        # Should not include bias in recovered weights (since it's zero)
        self.assertNotIn("model.lm_head.bias", recovered_hf)
        self.assertTrue(torch.equal(hf_weights["model.lm_head.weight"], recovered_hf["model.lm_head.weight"]))


class TestReversibleWeightConverter(unittest.TestCase):
    """Test cases for the main ReversibleWeightConverter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ReversibleWeightConverter()
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=2,  # Small for testing
            n_ctx=1024,
            d_vocab=50257,
            d_head=64,
            d_mlp=3072,
            act_fn="gelu",
            normalization_type="LN"
        )

    def create_mock_gpt2_state_dict(self):
        """Create a mock GPT-2 state dict for testing."""
        state_dict = {}

        # Embeddings
        state_dict["transformer.wte.weight"] = torch.randn(50257, 768)
        state_dict["transformer.wpe.weight"] = torch.randn(1024, 768)

        # Layers
        for layer_idx in range(2):
            # Attention
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"] = torch.randn(768, 2304)
            state_dict[f"transformer.h.{layer_idx}.attn.c_attn.bias"] = torch.randn(2304)
            state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"] = torch.randn(768, 768)
            state_dict[f"transformer.h.{layer_idx}.attn.c_proj.bias"] = torch.randn(768)

            # MLP
            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.weight"] = torch.randn(768, 3072)
            state_dict[f"transformer.h.{layer_idx}.mlp.c_fc.bias"] = torch.randn(3072)
            state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"] = torch.randn(3072, 768)
            state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.bias"] = torch.randn(768)

            # Norms
            state_dict[f"transformer.h.{layer_idx}.ln_1.weight"] = torch.randn(768)
            state_dict[f"transformer.h.{layer_idx}.ln_1.bias"] = torch.randn(768)
            state_dict[f"transformer.h.{layer_idx}.ln_2.weight"] = torch.randn(768)
            state_dict[f"transformer.h.{layer_idx}.ln_2.bias"] = torch.randn(768)

        # Final norm and unembedding
        state_dict["transformer.ln_f.weight"] = torch.randn(768)
        state_dict["transformer.ln_f.bias"] = torch.randn(768)
        state_dict["lm_head.weight"] = torch.randn(50257, 768)
        state_dict["lm_head.bias"] = torch.randn(50257)

        return state_dict

    def test_complete_hf_to_tlens_conversion(self):
        """Test complete HF → TLens conversion."""
        hf_state_dict = self.create_mock_gpt2_state_dict()

        # Convert
        tlens_state_dict = self.converter.hf_to_tlens(hf_state_dict, self.config, "gpt2")

        # Check expected TLens keys exist
        expected_tlens_keys = [
            "embed.W_E", "pos_embed.W_pos",
            "blocks.0.attn.W_Q", "blocks.0.attn.W_K", "blocks.0.attn.W_V", "blocks.0.attn.W_O",
            "blocks.0.mlp.W_in", "blocks.0.mlp.W_out",
            "blocks.0.ln1.w", "blocks.0.ln2.w",
            "ln_final.w", "unembed.W_U"
        ]

        for key in expected_tlens_keys:
            self.assertIn(key, tlens_state_dict, f"Missing TLens key: {key}")

    def test_complete_tlens_to_hf_conversion(self):
        """Test complete TLens → HF conversion."""
        # Start with HF, convert to TLens, then back to HF
        hf_state_dict = self.create_mock_gpt2_state_dict()
        tlens_state_dict = self.converter.hf_to_tlens(hf_state_dict, self.config, "gpt2")
        recovered_hf_state_dict = self.converter.tlens_to_hf(tlens_state_dict, self.config, "gpt2")

        # Check that all original keys are recovered
        for key in hf_state_dict:
            self.assertIn(key, recovered_hf_state_dict, f"Missing recovered key: {key}")

    def test_hf_to_tlens_round_trip_validation(self):
        """Test HF → TLens → HF round-trip validation."""
        hf_state_dict = self.create_mock_gpt2_state_dict()

        # Should pass validation
        result = self.converter.validate_round_trip_hf_to_tlens(hf_state_dict, self.config, "gpt2")
        self.assertTrue(result)

    def test_tlens_to_hf_round_trip_validation(self):
        """Test TLens → HF → TLens round-trip validation."""
        # Create a TLens state dict by converting from HF first
        hf_state_dict = self.create_mock_gpt2_state_dict()
        tlens_state_dict = self.converter.hf_to_tlens(hf_state_dict, self.config, "gpt2")

        # Should pass validation
        result = self.converter.validate_round_trip_tlens_to_hf(tlens_state_dict, self.config, "gpt2")
        self.assertTrue(result)

    def test_conversion_error_handling(self):
        """Test error handling for invalid conversions."""
        hf_state_dict = {}  # Empty state dict

        # Should handle missing keys gracefully
        tlens_state_dict = self.converter.hf_to_tlens(hf_state_dict, self.config, "gpt2")
        self.assertIsInstance(tlens_state_dict, dict)

        # Test with unsupported model type
        with self.assertRaises(ConversionError):
            self.converter.hf_to_tlens(hf_state_dict, self.config, "unsupported_model")

    def test_debug_conversion_mismatch(self):
        """Test debug functionality for conversion mismatches."""
        original_dict = {"key1": torch.tensor([1.0, 2.0, 3.0])}
        recovered_dict = {"key1": torch.tensor([1.0, 2.1, 3.0])}  # Small difference

        debug_info = self.converter.debug_conversion_mismatch(original_dict, recovered_dict)

        self.assertIn("value_mismatches", debug_info)
        self.assertEqual(len(debug_info["value_mismatches"]), 1)
        self.assertEqual(debug_info["value_mismatches"][0]["key"], "key1")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ReversibleWeightConverter()
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=1,
            n_ctx=1024,
            d_vocab=50257,
            act_fn="gelu"
        )

    def test_shape_mismatch_detection(self):
        """Test detection of shape mismatches."""
        # Create weights with wrong shapes
        hf_weights = {
            "transformer.wte.weight": torch.randn(100, 768),  # Wrong vocab size
        }

        # Convert should work
        tlens_weights = self.converter.embedding_converter.hf_to_tlens(hf_weights, self.config)

        # But shapes should be different
        self.assertEqual(tlens_weights["embed.W_E"].shape, (100, 768))

    def test_missing_keys_handling(self):
        """Test handling of missing keys."""
        # Empty HF weights
        hf_weights = {}

        # Should not crash
        tlens_weights = self.converter.embedding_converter.hf_to_tlens(hf_weights, self.config)
        self.assertIsInstance(tlens_weights, dict)

    def test_dtype_preservation(self):
        """Test that dtypes are preserved during conversion."""
        hf_weights = {
            "transformer.wte.weight": torch.randn(50257, 768, dtype=torch.float16),
        }

        tlens_weights = self.converter.embedding_converter.hf_to_tlens(hf_weights, self.config)
        recovered_hf = self.converter.embedding_converter.tlens_to_hf(tlens_weights, self.config)

        # Check dtype preservation
        self.assertEqual(hf_weights["transformer.wte.weight"].dtype,
                        recovered_hf["transformer.wte.weight"].dtype)


if __name__ == "__main__":
    # Set random seed for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    unittest.main(verbosity=2)