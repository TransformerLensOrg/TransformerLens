"""Unit tests for tensor utilities.

This module tests the tensor utility functions, particularly the filter_dict_by_prefix function.
"""

import torch

from transformer_lens.utilities.tensors import filter_dict_by_prefix


class TestFilterDictByPrefix:
    """Test cases for the filter_dict_by_prefix function."""

    def test_filter_dict_basic_prefix(self):
        """Test filtering dictionary with a basic prefix."""
        test_dict = {
            "transformer.h.0.attn.W_Q": torch.randn(10, 10),
            "transformer.h.0.attn.W_K": torch.randn(10, 10),
            "transformer.h.0.mlp.W_in": torch.randn(10, 10),
            "transformer.h.1.attn.W_Q": torch.randn(10, 10),
        }

        result = filter_dict_by_prefix(test_dict, "transformer.h.0")

        assert len(result) == 3
        assert "attn.W_Q" in result
        assert "attn.W_K" in result
        assert "mlp.W_in" in result
        assert "transformer.h.0.attn.W_Q" not in result  # Original keys should be stripped

    def test_filter_dict_with_trailing_dot(self):
        """Test filtering dictionary when prefix has a trailing dot."""
        test_dict = {
            "blocks.0.attn.W_Q": torch.randn(5, 5),
            "blocks.0.mlp.W_in": torch.randn(5, 5),
            "blocks.1.attn.W_Q": torch.randn(5, 5),
        }

        result = filter_dict_by_prefix(test_dict, "blocks.0.")

        assert len(result) == 2
        assert "attn.W_Q" in result
        assert "mlp.W_in" in result

    def test_filter_dict_no_matches(self):
        """Test filtering dictionary when no keys match the prefix."""
        test_dict = {
            "model.layer1.weight": torch.randn(3, 3),
            "model.layer2.weight": torch.randn(3, 3),
        }

        result = filter_dict_by_prefix(test_dict, "nonexistent")

        assert len(result) == 0
        assert result == {}

    def test_filter_dict_empty_input(self):
        """Test filtering an empty dictionary."""
        test_dict = {}

        result = filter_dict_by_prefix(test_dict, "any.prefix")

        assert len(result) == 0
        assert result == {}

    def test_filter_dict_single_match(self):
        """Test filtering dictionary with a single matching key."""
        test_dict = {
            "embed.W_E": torch.randn(100, 50),
            "unembed.W_U": torch.randn(50, 100),
        }

        result = filter_dict_by_prefix(test_dict, "embed")

        assert len(result) == 1
        assert "W_E" in result
        assert isinstance(result["W_E"], torch.Tensor)

    def test_filter_dict_preserves_values(self):
        """Test that filtering preserves the tensor values."""
        tensor1 = torch.randn(4, 4)
        tensor2 = torch.randn(4, 4)
        test_dict = {
            "prefix.a": tensor1,
            "prefix.b": tensor2,
            "other.c": torch.randn(4, 4),
        }

        result = filter_dict_by_prefix(test_dict, "prefix")

        assert len(result) == 2
        assert torch.equal(result["a"], tensor1)
        assert torch.equal(result["b"], tensor2)

    def test_filter_dict_nested_prefix(self):
        """Test filtering with deeply nested prefixes."""
        test_dict = {
            "model.encoder.layer.0.attention.self.query": torch.randn(2, 2),
            "model.encoder.layer.0.attention.self.key": torch.randn(2, 2),
            "model.encoder.layer.0.output.dense": torch.randn(2, 2),
            "model.encoder.layer.1.attention.self.query": torch.randn(2, 2),
        }

        result = filter_dict_by_prefix(test_dict, "model.encoder.layer.0")

        assert len(result) == 3
        assert "attention.self.query" in result
        assert "attention.self.key" in result
        assert "output.dense" in result

    def test_filter_dict_non_tensor_values(self):
        """Test filtering dictionary with non-tensor values."""
        test_dict = {
            "config.param1": "value1",
            "config.param2": 42,
            "config.param3": [1, 2, 3],
            "other.param": "value",
        }

        result = filter_dict_by_prefix(test_dict, "config")

        assert len(result) == 3
        assert result["param1"] == "value1"
        assert result["param2"] == 42
        assert result["param3"] == [1, 2, 3]

    def test_filter_dict_all_match(self):
        """Test filtering when all keys match the prefix."""
        test_dict = {
            "shared.weight1": torch.randn(2, 2),
            "shared.weight2": torch.randn(2, 2),
            "shared.weight3": torch.randn(2, 2),
        }

        result = filter_dict_by_prefix(test_dict, "shared")

        assert len(result) == 3
        assert len(result) == len(test_dict)

    def test_filter_dict_prefix_without_dot(self):
        """Test that prefix without trailing dot automatically adds one."""
        test_dict = {
            "transformer.h.0.attn": torch.randn(3, 3),
            "transformer.h.0.mlp": torch.randn(3, 3),
            "transformer.h.1.attn": torch.randn(3, 3),
        }

        result_with_dot = filter_dict_by_prefix(test_dict, "transformer.h.0.")
        result_without_dot = filter_dict_by_prefix(test_dict, "transformer.h.0")

        assert result_with_dot.keys() == result_without_dot.keys()
        assert len(result_with_dot) == 2

    def test_filter_dict_type_preservation(self):
        """Test that the function returns a dictionary."""
        test_dict = {
            "key1.subkey": "value1",
            "key2.subkey": "value2",
        }

        result = filter_dict_by_prefix(test_dict, "key1")

        assert isinstance(result, dict)

    def test_filter_dict_with_real_state_dict_keys(self):
        """Test with realistic state dict keys from a transformer model."""
        test_dict = {
            "transformer.h.0.ln_1.weight": torch.randn(768),
            "transformer.h.0.ln_1.bias": torch.randn(768),
            "transformer.h.0.attn.c_attn.weight": torch.randn(768, 2304),
            "transformer.h.0.attn.c_attn.bias": torch.randn(2304),
            "transformer.h.0.attn.c_proj.weight": torch.randn(768, 768),
            "transformer.h.0.attn.c_proj.bias": torch.randn(768),
            "transformer.h.1.ln_1.weight": torch.randn(768),
        }

        result = filter_dict_by_prefix(test_dict, "transformer.h.0")

        assert len(result) == 6
        assert "ln_1.weight" in result
        assert "ln_1.bias" in result
        assert "attn.c_attn.weight" in result
        assert "attn.c_attn.bias" in result
        assert "attn.c_proj.weight" in result
        assert "attn.c_proj.bias" in result
        # Layer 1 should not be included
        assert "transformer.h.1.ln_1.weight" not in result
        assert "ln_1.weight" in result  # But layer 0's ln_1.weight should be

    def test_filter_dict_multiple_prefixes_sequentially(self):
        """Test filtering with different prefixes on the same dictionary."""
        test_dict = {
            "encoder.layer1.weight": torch.randn(5, 5),
            "encoder.layer2.weight": torch.randn(5, 5),
            "decoder.layer1.weight": torch.randn(5, 5),
            "decoder.layer2.weight": torch.randn(5, 5),
        }

        encoder_result = filter_dict_by_prefix(test_dict, "encoder")
        decoder_result = filter_dict_by_prefix(test_dict, "decoder")

        assert len(encoder_result) == 2
        assert len(decoder_result) == 2
        assert "layer1.weight" in encoder_result
        assert "layer1.weight" in decoder_result
        # Ensure original dict is not modified
        assert len(test_dict) == 4
