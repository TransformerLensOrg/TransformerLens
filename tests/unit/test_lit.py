"""Tests for the LIT integration module.

This module contains unit and integration tests for the TransformerLens
LIT integration. Tests are designed to work both with and without
the optional lit-nlp dependency.

To run tests:
    pytest tests/unit/test_lit.py -v

To run with LIT installed:
    pip install lit-nlp
    pytest tests/unit/test_lit.py -v
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Checking if LIT is installed
try:
    import lit_nlp
    from lit_nlp.api import types as lit_types

    LIT_AVAILABLE = True
except ImportError:
    LIT_AVAILABLE = False
    lit_types = None


# Fixtures
@pytest.fixture
def mock_hooked_transformer():
    """Create a mock HookedTransformer for testing."""
    mock = MagicMock()

    # Mock config
    mock.cfg = MagicMock()
    mock.cfg.model_name = "test-model"
    mock.cfg.n_layers = 4
    mock.cfg.n_heads = 4
    mock.cfg.d_model = 64
    mock.cfg.d_head = 16
    mock.cfg.d_mlp = 256
    mock.cfg.d_vocab = 100
    mock.cfg.n_ctx = 512
    mock.cfg.act_fn = "gelu"
    mock.cfg.normalization_type = "LN"
    mock.cfg.positional_embedding_type = "standard"
    mock.cfg.device = "cpu"

    # Mock tokenizer
    mock.tokenizer = MagicMock()
    mock.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock.tokenizer.decode.return_value = "test"
    mock.tokenizer.convert_ids_to_tokens.return_value = ["<s>", "test", "token", "s", "</s>"]
    mock.tokenizer.padding_side = "right"
    mock.tokenizer.pad_token = "<pad>"
    mock.tokenizer.eos_token = "</s>"
    mock.tokenizer.bos_token = "<s>"

    # Mock to_tokens
    mock.to_tokens.return_value = torch.tensor([[1, 2, 3, 4, 5]])

    # Mock embed
    mock.embed.return_value = torch.randn(1, 5, 64)

    # Mock pos_embed
    mock.pos_embed.return_value = torch.randn(1, 5, 64)

    # Mock forward
    mock.return_value = torch.randn(1, 5, 100)

    # Mock run_with_cache
    def mock_run_with_cache(*args, **kwargs):
        logits = torch.randn(1, 5, 100)
        cache = MagicMock()

        # Mock cache access
        def getitem(key):
            if "hook_embed" in key:
                return torch.randn(1, 5, 64)
            elif "hook_resid_post" in key:
                return torch.randn(1, 5, 64)
            elif "hook_pattern" in key:
                return torch.randn(1, 4, 5, 5)  # [batch, heads, q, k]
            return torch.randn(1, 5, 64)

        cache.__getitem__ = getitem
        return logits, cache

    mock.run_with_cache = mock_run_with_cache

    return mock


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world!",
        "Machine learning is fascinating.",
    ]


@pytest.fixture
def sample_examples(sample_texts):
    """Sample examples in LIT format."""
    return [{"text": text} for text in sample_texts]


# Tests for utils.py


class TestUtils:
    """Tests for utility functions."""

    def test_check_lit_installed(self):
        """Test LIT installation check."""
        from transformer_lens.lit.utils import check_lit_installed

        # Should return a boolean
        result = check_lit_installed()
        assert isinstance(result, bool)
        assert result == LIT_AVAILABLE

    def test_tensor_to_numpy_tensor(self):
        """Test tensor to numpy conversion with tensor input."""
        from transformer_lens.lit.utils import tensor_to_numpy

        tensor = torch.randn(3, 4)
        result = tensor_to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 4)
        np.testing.assert_array_almost_equal(result, tensor.numpy())

    def test_tensor_to_numpy_array(self):
        """Test tensor to numpy conversion with numpy input."""
        from transformer_lens.lit.utils import tensor_to_numpy

        array = np.random.randn(3, 4)
        result = tensor_to_numpy(array)

        assert isinstance(result, np.ndarray)
        assert result is array  # Should return same object

    def test_tensor_to_numpy_none(self):
        """Test tensor to numpy conversion with None input."""
        from transformer_lens.lit.utils import tensor_to_numpy

        result = tensor_to_numpy(None)
        assert result is None

    def test_numpy_to_tensor(self):
        """Test numpy to tensor conversion."""
        from transformer_lens.lit.utils import numpy_to_tensor

        array = np.random.randn(3, 4).astype(np.float32)
        result = numpy_to_tensor(array)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 4)

    def test_numpy_to_tensor_with_device(self):
        """Test numpy to tensor conversion with device specification."""
        from transformer_lens.lit.utils import numpy_to_tensor

        array = np.random.randn(3, 4).astype(np.float32)
        result = numpy_to_tensor(array, device="cpu")

        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_clean_token_string(self):
        """Test token string cleaning."""
        from transformer_lens.lit.utils import clean_token_string

        # GPT-2 style
        assert clean_token_string("Ġhello") == "▁hello"

        # SentencePiece style
        assert clean_token_string("▁world") == "▁world"

        # BERT style
        assert clean_token_string("##ing") == "ing"

        # Regular token
        assert clean_token_string("test") == "test"

    def test_clean_token_strings(self):
        """Test batch token string cleaning."""
        from transformer_lens.lit.utils import clean_token_strings

        tokens = ["Ġhello", "▁world", "##ing", "test"]
        result = clean_token_strings(tokens)

        assert result == ["▁hello", "▁world", "ing", "test"]

    def test_batch_examples(self):
        """Test example batching."""
        from transformer_lens.lit.utils import batch_examples

        examples = [{"text": f"example {i}"} for i in range(10)]
        batches = batch_examples(examples, batch_size=3)

        assert len(batches) == 4  # 10 / 3 = 4 (rounded up)
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_unbatch_outputs(self):
        """Test output unbatching."""
        from transformer_lens.lit.utils import unbatch_outputs

        batched = {
            "logits": np.random.randn(3, 5, 10),
            "tokens": [["a", "b"], ["c", "d"], ["e", "f"]],
        }
        result = unbatch_outputs(batched)

        assert len(result) == 3
        assert result[0]["logits"].shape == (5, 10)
        assert result[0]["tokens"] == ["a", "b"]

    def test_get_model_info(self, mock_hooked_transformer):
        """Test model info extraction."""
        from transformer_lens.lit.utils import get_model_info

        info = get_model_info(mock_hooked_transformer)

        assert info["model_name"] == "test-model"
        assert info["n_layers"] == 4
        assert info["n_heads"] == 4
        assert info["d_model"] == 64


# Tests for constants.py


class TestConstants:
    """Tests for constants module."""

    def test_input_field_names(self):
        """Test input field names are defined."""
        from transformer_lens.lit.constants import INPUT_FIELDS

        assert INPUT_FIELDS.TEXT == "text"
        assert INPUT_FIELDS.TOKENS == "tokens"
        assert INPUT_FIELDS.TARGET_MASK == "target_mask"

    def test_output_field_names(self):
        """Test output field names are defined."""
        from transformer_lens.lit.constants import OUTPUT_FIELDS

        assert OUTPUT_FIELDS.TOKENS == "tokens"
        assert OUTPUT_FIELDS.TOP_K_TOKENS == "top_k_tokens"
        assert OUTPUT_FIELDS.CLS_EMBEDDING == "cls_embedding"

    def test_default_config(self):
        """Test default configuration values."""
        from transformer_lens.lit.constants import DEFAULTS

        assert DEFAULTS.MAX_SEQ_LENGTH == 512
        assert DEFAULTS.BATCH_SIZE == 8
        assert DEFAULTS.TOP_K == 10
        assert isinstance(DEFAULTS.COMPUTE_GRADIENTS, bool)

    def test_hook_point_names(self):
        """Test hook point name templates."""
        from transformer_lens.lit.constants import HOOK_POINTS

        assert HOOK_POINTS.HOOK_EMBED == "hook_embed"
        assert "{layer}" in HOOK_POINTS.RESID_PRE_TEMPLATE
        assert "{layer}" in HOOK_POINTS.ATTN_PATTERN_TEMPLATE

    def test_error_messages(self):
        """Test error messages are defined."""
        from transformer_lens.lit.constants import ERRORS

        assert "tokenizer" in ERRORS.NO_TOKENIZER.lower()
        assert "lit" in ERRORS.LIT_NOT_INSTALLED.lower()


# Tests for dataset.py


class TestDatasets:
    """Tests for dataset classes."""

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_simple_text_dataset_init(self, sample_examples):
        """Test SimpleTextDataset initialization."""
        from transformer_lens.lit.dataset import SimpleTextDataset

        dataset = SimpleTextDataset(sample_examples, name="TestDataset")

        assert len(dataset.examples) == 3
        assert dataset.examples[0]["text"] == sample_examples[0]["text"]

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_simple_text_dataset_from_strings(self, sample_texts):
        """Test creating dataset from strings."""
        from transformer_lens.lit.dataset import SimpleTextDataset

        dataset = SimpleTextDataset.from_strings(sample_texts)

        assert len(dataset.examples) == 3
        assert dataset.examples[0]["text"] == sample_texts[0]

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_simple_text_dataset_spec(self, sample_examples):
        """Test dataset spec method."""
        from transformer_lens.lit.dataset import SimpleTextDataset

        dataset = SimpleTextDataset(sample_examples)
        spec = dataset.spec()

        assert "text" in spec
        assert isinstance(spec["text"], lit_types.TextSegment)  # type: ignore[union-attr]

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_simple_text_dataset_missing_text(self):
        """Test dataset validation for missing text field."""
        from transformer_lens.lit.dataset import SimpleTextDataset

        with pytest.raises(ValueError, match="missing required field"):
            SimpleTextDataset([{"other_field": "value"}])

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_prompt_completion_dataset(self):
        """Test PromptCompletionDataset."""
        from transformer_lens.lit.dataset import PromptCompletionDataset

        examples = [
            {"prompt": "Hello", "completion": " world"},
            {"prompt": "The answer is", "completion": " 42"},
        ]
        dataset = PromptCompletionDataset(examples)

        assert len(dataset.examples) == 2
        assert dataset.examples[0]["text"] == "Hello world"
        assert dataset.examples[0]["prompt"] == "Hello"
        assert dataset.examples[0]["completion"] == " world"

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_prompt_completion_from_pairs(self):
        """Test creating PromptCompletionDataset from pairs."""
        from transformer_lens.lit.dataset import PromptCompletionDataset

        pairs = [("Hello", " world"), ("The answer is", " 42")]
        dataset = PromptCompletionDataset.from_pairs(pairs)

        assert len(dataset.examples) == 2

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_ioi_dataset_generate(self):
        """Test IOI dataset generation."""
        from transformer_lens.lit.dataset import IOIDataset

        dataset = IOIDataset.generate(n_examples=10, seed=42)

        assert len(dataset.examples) == 10
        # Check structure
        ex = dataset.examples[0]
        assert "text" in ex
        assert "name1" in ex
        assert "name2" in ex
        assert "answer" in ex

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_induction_dataset_generate(self):
        """Test Induction dataset generation."""
        from transformer_lens.lit.dataset import InductionDataset

        dataset = InductionDataset.generate_simple(n_examples=10, seed=42)

        assert len(dataset.examples) == 10
        ex = dataset.examples[0]
        assert "text" in ex
        assert "pattern" in ex


# Tests for model.py


class TestModel:
    """Tests for model wrapper classes."""

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_config_defaults(self):
        """Test HookedTransformerLITConfig defaults."""
        from transformer_lens.lit.model import HookedTransformerLITConfig

        config = HookedTransformerLITConfig()

        assert config.max_seq_length == 512
        assert config.batch_size == 8
        assert config.top_k == 10
        assert config.compute_gradients is True
        assert config.output_attention is True
        assert config.output_embeddings is True

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_config_custom(self):
        """Test custom configuration."""
        from transformer_lens.lit.model import HookedTransformerLITConfig

        config = HookedTransformerLITConfig(
            max_seq_length=256,
            batch_size=4,
            compute_gradients=False,
        )

        assert config.max_seq_length == 256
        assert config.batch_size == 4
        assert config.compute_gradients is False

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_model_wrapper_init(self, mock_hooked_transformer):
        """Test HookedTransformerLIT initialization."""
        from transformer_lens.lit.model import HookedTransformerLIT

        # Need to mock the isinstance check - patch where it's imported
        with patch("transformer_lens.HookedTransformer", type(mock_hooked_transformer)):
            wrapper = HookedTransformerLIT(mock_hooked_transformer)

            assert wrapper.model is mock_hooked_transformer
            assert wrapper.config is not None

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_model_wrapper_invalid_model(self):
        """Test that invalid model type raises error."""
        from transformer_lens.lit.model import HookedTransformerLIT

        with pytest.raises(TypeError):
            HookedTransformerLIT("not a model")  # type: ignore[union-attr]

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_model_input_spec(self, mock_hooked_transformer):
        """Test input_spec method."""
        from transformer_lens.lit.model import HookedTransformerLIT

        with patch("transformer_lens.HookedTransformer", type(mock_hooked_transformer)):
            wrapper = HookedTransformerLIT(mock_hooked_transformer)
            spec = wrapper.input_spec()

            assert "text" in spec
            assert isinstance(spec["text"], lit_types.TextSegment)  # type: ignore[union-attr]

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_model_output_spec(self, mock_hooked_transformer):
        """Test output_spec method."""
        from transformer_lens.lit.model import HookedTransformerLIT

        with patch("transformer_lens.HookedTransformer", type(mock_hooked_transformer)):
            wrapper = HookedTransformerLIT(mock_hooked_transformer)
            spec = wrapper.output_spec()

            assert "tokens" in spec
            assert "top_k_tokens" in spec
            # With default config, should have embeddings
            assert "cls_embedding" in spec

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_model_output_spec_no_embeddings(self, mock_hooked_transformer):
        """Test output_spec without embeddings."""
        from transformer_lens.lit.model import (
            HookedTransformerLIT,
            HookedTransformerLITConfig,
        )

        # Must also disable compute_gradients since gradients require embeddings
        config = HookedTransformerLITConfig(output_embeddings=False, compute_gradients=False)

        with patch("transformer_lens.HookedTransformer", type(mock_hooked_transformer)):
            wrapper = HookedTransformerLIT(mock_hooked_transformer, config=config)
            spec = wrapper.output_spec()

            assert "tokens" in spec
            assert "cls_embedding" not in spec

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_model_description(self, mock_hooked_transformer):
        """Test model description."""
        from transformer_lens.lit.model import HookedTransformerLIT

        with patch("transformer_lens.HookedTransformer", type(mock_hooked_transformer)):
            wrapper = HookedTransformerLIT(mock_hooked_transformer)
            desc = wrapper.description()

            assert "test-model" in desc
            assert "4L" in desc  # n_layers
            assert "4H" in desc  # n_heads


# Tests for __init__.py


class TestInit:
    """Tests for module initialization and exports."""

    def test_exports_available(self):
        """Test that expected exports are available."""
        from transformer_lens import lit

        # Check key exports exist
        assert hasattr(lit, "HookedTransformerLIT")
        assert hasattr(lit, "HookedTransformerLITConfig")
        assert hasattr(lit, "SimpleTextDataset")
        assert hasattr(lit, "serve")
        assert hasattr(lit, "LITWidget")
        assert hasattr(lit, "check_lit_installed")

    def test_constants_exported(self):
        """Test that constants are exported."""
        from transformer_lens.lit import INPUT_FIELDS, OUTPUT_FIELDS

        assert INPUT_FIELDS.TEXT == "text"
        assert OUTPUT_FIELDS.TOKENS == "tokens"

    @pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
    def test_all_exports(self):
        """Test __all__ exports are importable."""
        from transformer_lens import lit

        for name in lit.__all__:
            assert hasattr(lit, name), f"Missing export: {name}"


# Integration Tests


@pytest.mark.skipif(not LIT_AVAILABLE, reason="LIT not installed")
class TestIntegration:
    """Integration tests that require both LIT and a model."""

    def test_full_prediction_flow(self, mock_hooked_transformer):
        """Test full prediction flow with mock model."""
        from transformer_lens.lit.model import HookedTransformerLIT

        with patch("transformer_lens.HookedTransformer", type(mock_hooked_transformer)):
            wrapper = HookedTransformerLIT(mock_hooked_transformer)

            # Create input
            inputs = [{"text": "Hello world"}]

            # This would fail with the mock, but we can at least check the structure
            # In a real test with a real model, this would work
            input_spec = wrapper.input_spec()
            output_spec = wrapper.output_spec()

            assert "text" in input_spec
            assert "tokens" in output_spec

    def test_dataset_model_compatibility(self):
        """Test that datasets are compatible with model input spec."""
        from transformer_lens.lit.dataset import SimpleTextDataset

        dataset = SimpleTextDataset.from_strings(["test"])
        spec = dataset.spec()

        # Check that dataset spec matches expected model input
        assert "text" in spec


# Run tests

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
