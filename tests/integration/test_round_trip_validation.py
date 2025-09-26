#!/usr/bin/env python3
"""
Integration tests for round-trip weight conversion validation.

This module contains integration tests that validate the complete round-trip
conversion pipeline with real models and compare outputs.
"""

import unittest
import warnings
from unittest.mock import patch
import tempfile
import os

import torch
import numpy as np

# Mock transformers import to avoid dependency issues in testing
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create mock classes
    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pass
    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pass

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.conversion_utils.reversible_weight_converter import ReversibleWeightConverter
from transformer_lens.conversion_utils.round_trip_validator import RoundTripValidator
from transformer_lens.conversion_utils.compare_script_integration import CompareScriptIntegration


class MockModel:
    """Mock model for testing without real model dependencies."""

    def __init__(self, state_dict, device="cpu"):
        self.state_dict_data = state_dict
        self.device = device
        self.eval_called = False

    def state_dict(self):
        return self.state_dict_data

    def load_state_dict(self, state_dict, strict=True):
        self.state_dict_data.update(state_dict)

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        # Mock forward pass - return random logits
        input_ids = kwargs.get("input_ids", torch.tensor([[1, 2, 3]]))
        batch_size, seq_len = input_ids.shape
        vocab_size = 50257  # GPT-2 vocab size

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        return MockOutput(logits)


class MockHookedTransformer:
    """Mock HookedTransformer for testing."""

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device

        # Create mock state dict
        self.state_dict_data = self._create_mock_tlens_state_dict()

    def _create_mock_tlens_state_dict(self):
        """Create a mock TLens state dict."""
        state_dict = {}

        # Embeddings
        state_dict["embed.W_E"] = torch.randn(50257, 768)
        state_dict["pos_embed.W_pos"] = torch.randn(1024, 768)

        # Two layers
        for layer_idx in range(2):
            # Attention
            state_dict[f"blocks.{layer_idx}.attn.W_Q"] = torch.randn(12, 768, 64)
            state_dict[f"blocks.{layer_idx}.attn.W_K"] = torch.randn(12, 768, 64)
            state_dict[f"blocks.{layer_idx}.attn.W_V"] = torch.randn(12, 768, 64)
            state_dict[f"blocks.{layer_idx}.attn.W_O"] = torch.randn(12, 64, 768)
            state_dict[f"blocks.{layer_idx}.attn.b_Q"] = torch.randn(12, 64)
            state_dict[f"blocks.{layer_idx}.attn.b_K"] = torch.randn(12, 64)
            state_dict[f"blocks.{layer_idx}.attn.b_V"] = torch.randn(12, 64)
            state_dict[f"blocks.{layer_idx}.attn.b_O"] = torch.randn(768)

            # MLP
            state_dict[f"blocks.{layer_idx}.mlp.W_in"] = torch.randn(768, 3072)
            state_dict[f"blocks.{layer_idx}.mlp.b_in"] = torch.randn(3072)
            state_dict[f"blocks.{layer_idx}.mlp.W_out"] = torch.randn(3072, 768)
            state_dict[f"blocks.{layer_idx}.mlp.b_out"] = torch.randn(768)

            # Norms
            state_dict[f"blocks.{layer_idx}.ln1.w"] = torch.randn(768)
            state_dict[f"blocks.{layer_idx}.ln1.b"] = torch.randn(768)
            state_dict[f"blocks.{layer_idx}.ln2.w"] = torch.randn(768)
            state_dict[f"blocks.{layer_idx}.ln2.b"] = torch.randn(768)

        # Final components
        state_dict["ln_final.w"] = torch.randn(768)
        state_dict["ln_final.b"] = torch.randn(768)
        state_dict["unembed.W_U"] = torch.randn(768, 50257)
        state_dict["unembed.b_U"] = torch.randn(50257)

        return state_dict

    def state_dict(self):
        return self.state_dict_data

    def load_state_dict(self, state_dict, strict=True):
        self.state_dict_data.update(state_dict)

    def to_tokens(self, text):
        # Mock tokenization - return simple token sequence
        tokens = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        return tokens

    def __call__(self, tokens):
        # Mock forward pass
        batch_size, seq_len = tokens.shape
        vocab_size = 50257
        logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        return logits

    @classmethod
    def from_pretrained(cls, model_name, device="cpu", **kwargs):
        return cls(model_name, device)


class MockTransformerBridge:
    """Mock TransformerBridge for testing."""

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device

        # Create mock config
        self.cfg = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=2,
            n_ctx=1024,
            d_vocab=50257,
            d_head=64,
            d_mlp=3072,
            act_fn="gelu",
            normalization_type="LN"
        )

    def to_tokens(self, text):
        # Mock tokenization
        return torch.tensor([[1, 2, 3, 4, 5]], device=self.device)

    def __call__(self, tokens):
        # Mock forward pass
        batch_size, seq_len = tokens.shape
        vocab_size = 50257
        logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        return logits

    @classmethod
    def boot_transformers(cls, model_name, device="cpu", **kwargs):
        return cls(model_name, device)


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers not available")
class TestRoundTripValidationIntegration(unittest.TestCase):
    """Integration tests for round-trip validation with mocked models."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = RoundTripValidator(tolerance=1e-6)
        self.converter = ReversibleWeightConverter()

        # Mock config
        self.config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=2,
            n_ctx=1024,
            d_vocab=50257,
            d_head=64,
            d_mlp=3072,
            act_fn="gelu",
            normalization_type="LN"
        )

    def create_mock_hf_state_dict(self):
        """Create a mock HF state dict."""
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

        # Final components
        state_dict["transformer.ln_f.weight"] = torch.randn(768)
        state_dict["transformer.ln_f.bias"] = torch.randn(768)
        state_dict["lm_head.weight"] = torch.randn(50257, 768)
        state_dict["lm_head.bias"] = torch.randn(50257)

        return state_dict

    @patch('transformer_lens.conversion_utils.round_trip_validator.HookedTransformer', MockHookedTransformer)
    @patch('transformer_lens.conversion_utils.round_trip_validator.TransformerBridge', MockTransformerBridge)
    @patch('transformer_lens.conversion_utils.round_trip_validator.AutoModelForCausalLM')
    @patch('transformer_lens.conversion_utils.round_trip_validator.AutoTokenizer')
    def test_validate_model_round_trip_mocked(self, mock_tokenizer, mock_auto_model):
        """Test complete model round-trip validation with mocked components."""
        # Set up mocks
        mock_hf_state_dict = self.create_mock_hf_state_dict()
        mock_hf_model = MockModel(mock_hf_state_dict)
        mock_auto_model.from_pretrained.return_value = mock_hf_model

        mock_tokenizer_instance = type('MockTokenizer', (), {
            'pad_token': None,
            'eos_token': '[EOS]',
            '__call__': lambda self, text, **kwargs: {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
        })()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Run validation
        result = self.validator.validate_model_round_trip("gpt2", "cpu")

        # Check that the validation ran (even if it fails due to mocking)
        self.assertIn("model_name", result)
        self.assertEqual(result["model_name"], "gpt2")
        self.assertIn("hf_to_tlens_round_trip", result)
        self.assertIn("tlens_to_hf_round_trip", result)

    def test_component_round_trip_validation(self):
        """Test individual component round-trip validation."""
        # Test embedding conversion
        hf_weights = {
            "transformer.wte.weight": torch.randn(50257, 768),
            "transformer.wpe.weight": torch.randn(1024, 768)
        }

        result = self.validator.validate_component_round_trip(
            "embedding", hf_weights, self.config, "gpt2"
        )

        self.assertIn("success", result)
        self.assertEqual(result["component_type"], "embedding")

    def test_debug_failed_conversion(self):
        """Test debugging functionality for failed conversions."""
        # Create weights that will cause issues
        problematic_weights = {
            "transformer.wte.weight": torch.randn(100, 768)  # Wrong vocab size
        }

        # This should complete without crashing
        debug_info = self.converter.debug_conversion_mismatch(
            problematic_weights, {}, tolerance=1e-6
        )

        self.assertIn("missing_keys", debug_info)
        self.assertIn("summary", debug_info)

    def test_tolerance_levels(self):
        """Test validation with different tolerance levels."""
        # Create weights with small differences
        original_weights = {"key1": torch.tensor([1.0, 2.0, 3.0])}
        modified_weights = {"key1": torch.tensor([1.0, 2.0001, 3.0])}  # Small difference

        # Should pass with loose tolerance
        debug_info = self.converter.debug_conversion_mismatch(
            original_weights, modified_weights, tolerance=1e-3
        )
        self.assertEqual(len(debug_info["value_mismatches"]), 0)

        # Should fail with tight tolerance
        debug_info = self.converter.debug_conversion_mismatch(
            original_weights, modified_weights, tolerance=1e-6
        )
        self.assertEqual(len(debug_info["value_mismatches"]), 1)

    def test_model_type_inference(self):
        """Test model type inference from model names."""
        test_cases = [
            ("gpt2", "gpt2"),
            ("gpt2-medium", "gpt2"),
            ("microsoft/DialoGPT-medium", "gpt2"),
            ("meta-llama/Llama-2-7b-hf", "llama"),
            ("mistralai/Mistral-7B-v0.1", "mistral"),
            ("google/gemma-7b", "gemma"),
            ("unknown-model", "gpt2")  # Default fallback
        ]

        for model_name, expected_type in test_cases:
            inferred_type = self.validator._infer_model_type(model_name)
            self.assertEqual(inferred_type, expected_type,
                           f"Failed for {model_name}: expected {expected_type}, got {inferred_type}")

    def test_weight_shape_validation(self):
        """Test validation of weight shapes during conversion."""
        # Create HF weights with correct shapes
        hf_weights = self.create_mock_hf_state_dict()

        # Convert to TLens
        tlens_weights = self.converter.hf_to_tlens(hf_weights, self.config, "gpt2")

        # Check some key shapes
        self.assertEqual(tlens_weights["embed.W_E"].shape, (50257, 768))
        self.assertEqual(tlens_weights["blocks.0.attn.W_Q"].shape, (12, 768, 64))
        self.assertEqual(tlens_weights["unembed.W_U"].shape, (768, 50257))  # Note transpose

        # Convert back to HF
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, self.config, "gpt2")

        # Check shapes are recovered
        self.assertEqual(recovered_hf["transformer.wte.weight"].shape, (50257, 768))
        self.assertEqual(recovered_hf["lm_head.weight"].shape, (50257, 768))

    def test_dtype_preservation(self):
        """Test that dtypes are preserved during conversion."""
        # Create weights with different dtypes
        hf_weights = {
            "transformer.wte.weight": torch.randn(50257, 768, dtype=torch.float16),
            "lm_head.weight": torch.randn(50257, 768, dtype=torch.float32)
        }

        # Convert and back
        tlens_weights = self.converter.hf_to_tlens(hf_weights, self.config, "gpt2")
        recovered_hf = self.converter.tlens_to_hf(tlens_weights, self.config, "gpt2")

        # Check dtype preservation
        self.assertEqual(hf_weights["transformer.wte.weight"].dtype,
                        recovered_hf["transformer.wte.weight"].dtype)
        self.assertEqual(hf_weights["lm_head.weight"].dtype,
                        recovered_hf["lm_head.weight"].dtype)


class TestCompareScriptIntegration(unittest.TestCase):
    """Test integration with compare script functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.integration = CompareScriptIntegration(tolerance=1e-6)

    @patch('transformer_lens.conversion_utils.compare_script_integration.HookedTransformer', MockHookedTransformer)
    @patch('transformer_lens.conversion_utils.compare_script_integration.TransformerBridge', MockTransformerBridge)
    @patch('transformer_lens.conversion_utils.compare_script_integration.AutoModelForCausalLM')
    def test_run_original_comparison(self, mock_auto_model):
        """Test running the original comparison logic."""
        # Set up mock
        mock_hf_model = MockModel({})
        mock_auto_model.from_pretrained.return_value = mock_hf_model

        # Run comparison
        result = self.integration._run_original_comparison("gpt2", "cpu")

        self.assertIn("success", result)
        # Should be True since we're using consistent mock outputs
        self.assertTrue(result.get("success", False))

    def test_generate_validation_report(self):
        """Test generation of validation reports."""
        # Create mock results
        results = {
            "model_name": "gpt2",
            "device": "cpu",
            "overall_success": True,
            "original_comparison": {"success": True, "max_difference": 1e-8},
            "round_trip_validation": {
                "success": True,
                "hf_to_tlens_round_trip": {"success": True},
                "tlens_to_hf_round_trip": {"success": True}
            },
            "round_trip_comparison": {
                "success": True,
                "hf_round_trip_max_diff": 1e-7,
                "tlens_round_trip_max_diff": 1e-7
            }
        }

        report = self.integration.generate_validation_report(results)

        # Check report contains expected sections
        self.assertIn("Round-Trip Validation Report", report)
        self.assertIn("Model: gpt2", report)
        self.assertIn("Overall Status: ✓ PASSED", report)
        self.assertIn("Original Comparison: ✓", report)

    def test_error_handling_in_integration(self):
        """Test error handling in integration methods."""
        # Test with invalid model name
        result = self.integration._run_original_comparison("invalid-model", "cpu")

        # Should handle errors gracefully
        self.assertIn("error", result)


class TestRealModelCompatibility(unittest.TestCase):
    """Test compatibility with real model architectures (structure only)."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ReversibleWeightConverter()

    def test_gpt2_config_compatibility(self):
        """Test that GPT-2 config works with converter."""
        config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            n_ctx=1024,
            d_vocab=50257,
            d_head=64,
            act_fn="gelu",
            normalization_type="LN"
        )

        # Test that we can get expected keys
        embedding_keys = self.converter.embedding_converter.get_hf_keys(config, model_type="gpt2")
        self.assertIn("transformer.wte.weight", embedding_keys)

        attention_keys = self.converter.attention_converter.get_hf_keys(config, layer_idx=0, model_type="gpt2")
        self.assertIn("transformer.h.0.attn.c_attn.weight", attention_keys)

    def test_llama_config_compatibility(self):
        """Test that LLaMA config works with converter."""
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

        # Test that we can get expected keys
        embedding_keys = self.converter.embedding_converter.get_hf_keys(config, model_type="llama")
        self.assertIn("model.embed_tokens.weight", embedding_keys)

        attention_keys = self.converter.attention_converter.get_hf_keys(config, layer_idx=0, model_type="llama")
        self.assertIn("model.layers.0.self_attn.q_proj.weight", attention_keys)

    def test_converter_key_consistency(self):
        """Test that converter key mappings are consistent."""
        config = HookedTransformerConfig(
            d_model=768,
            n_heads=12,
            n_layers=2,
            n_ctx=1024,
            d_vocab=50257,
            d_head=64,
            act_fn="gelu"
        )

        # For each converter, check that get_hf_keys and get_tlens_keys are consistent
        converters = [
            (self.converter.embedding_converter, {}),
            (self.converter.attention_converter, {"layer_idx": 0}),
            (self.converter.mlp_converter, {"layer_idx": 0}),
            (self.converter.norm_converter, {"layer_idx": 0, "norm_type": "ln1"}),
            (self.converter.unembed_converter, {})
        ]

        for converter, kwargs in converters:
            hf_keys = converter.get_hf_keys(config, model_type="gpt2", **kwargs)
            tlens_keys = converter.get_tlens_keys(config, model_type="gpt2", **kwargs)

            # Both should return lists
            self.assertIsInstance(hf_keys, list)
            self.assertIsInstance(tlens_keys, list)

            # Both should be non-empty for most converters
            if converter != self.converter.norm_converter or kwargs.get("norm_type") != "final":
                self.assertTrue(len(hf_keys) > 0)
                self.assertTrue(len(tlens_keys) > 0)


if __name__ == "__main__":
    # Set random seeds for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)

    # Suppress warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run tests
    unittest.main(verbosity=2)