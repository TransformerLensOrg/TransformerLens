"""Integration tests for generation API compatibility.

This module tests generation API features including HuggingFace-style ModelOutput
support and TransformerBridge batch dimension compatibility.
"""

import warnings

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def gpt2_ht():
    """Load GPT-2 HookedTransformer once per module."""
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="module")
def gpt2_bridge():
    """Load GPT-2 TransformerBridge once per module."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


class TestHookedTransformerGenerationModelOutput:
    """Tests for HookedTransformer generation with ModelOutput returns."""

    def test_generate_with_output_logits_returns_modeloutput(self, gpt2_ht):
        """Test that output_logits=True returns a ModelOutput with sequences and logits."""
        prompt = "The quick brown"
        max_new_tokens = 5

        result = gpt2_ht.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
            output_logits=True,
        )

        # Check that we got a ModelOutput-like object
        assert hasattr(result, "sequences"), "Result should have sequences attribute"
        assert hasattr(result, "logits"), "Result should have logits attribute"

        # Check sequences shape and type
        assert isinstance(result.sequences, torch.Tensor), "sequences should be a tensor"
        assert result.sequences.ndim == 2, "sequences should be 2D [batch, pos]"

        # Check logits structure and shape
        assert isinstance(result.logits, tuple), "logits should be a tuple"
        assert (
            len(result.logits) == max_new_tokens
        ), f"logits tuple should have {max_new_tokens} elements"

        # Each logit tensor should be [batch, vocab]
        for i, logit in enumerate(result.logits):
            assert isinstance(logit, torch.Tensor), f"logits[{i}] should be a tensor"
            assert logit.ndim == 2, f"logits[{i}] should be 2D [batch, vocab]"
            assert (
                logit.shape[0] == result.sequences.shape[0]
            ), f"logits[{i}] batch size should match sequences"
            assert (
                logit.shape[1] == gpt2_ht.cfg.d_vocab
            ), f"logits[{i}] vocab size should match model config"

    def test_generate_without_output_logits_returns_normal(self, gpt2_ht):
        """Test that without output_logits flag, generation returns normal format."""
        prompt = "The quick brown"

        result = gpt2_ht.generate(
            prompt,
            max_new_tokens=5,
            do_sample=False,
            verbose=False,
        )

        # Should return a string (default return_type="input" with string input)
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > len(prompt), "Generated text should be longer than prompt"

    def test_generate_output_logits_with_return_type_tokens(self, gpt2_ht):
        """Test output_logits with return_type='tokens' returns ModelOutput with token sequences."""
        prompt = "Hello world"
        max_new_tokens = 3

        result = gpt2_ht.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            return_type="tokens",
            do_sample=False,
            verbose=False,
            output_logits=True,
        )

        # Check ModelOutput structure
        assert hasattr(result, "sequences"), "Result should have sequences"
        assert hasattr(result, "logits"), "Result should have logits"

        # Sequences should be tokens
        assert isinstance(result.sequences, torch.Tensor), "sequences should be a tensor"
        assert result.sequences.dtype in [
            torch.long,
            torch.int,
            torch.int64,
        ], "sequences should be integer tokens"

        # Check logits
        assert len(result.logits) == max_new_tokens, "logits should match max_new_tokens"

    def test_return_dict_in_generate_silently_ignored(self, gpt2_ht):
        """Test that return_dict_in_generate is silently ignored without warnings."""
        prompt = "Test"

        # Should not raise any warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = gpt2_ht.generate(
                prompt,
                max_new_tokens=2,
                verbose=False,
                return_dict_in_generate=True,  # Should be silently ignored
            )

            # Check no warnings were raised
            assert len(w) == 0, "return_dict_in_generate should be silently ignored"

        # Result should still be normal (string)
        assert isinstance(result, str), "Result should be a string"

    def test_unsupported_hf_flags_trigger_warning(self, gpt2_ht):
        """Test that unsupported HF generation kwargs trigger UserWarning."""
        prompt = "Test"

        with pytest.warns(UserWarning, match="unsupported generation kwargs"):
            result = gpt2_ht.generate(
                prompt,
                max_new_tokens=2,
                verbose=False,
                output_scores=True,  # Unsupported flag
                output_attentions=True,  # Unsupported flag
            )

        # Result should still work (string)
        assert isinstance(result, str), "Result should be a string despite warnings"

    def test_logits_consistency_with_forward_pass(self, gpt2_ht):
        """Test that logits from generate match those from forward pass."""
        prompt = "Hello"

        # Generate with output_logits
        result = gpt2_ht.generate(
            prompt,
            max_new_tokens=1,
            do_sample=False,
            verbose=False,
            output_logits=True,
        )

        # Get first generated token from sequences
        first_new_token = result.sequences[0, -1]

        # Get logits for that token
        first_logits = result.logits[0][0]

        # The argmax of logits should match the generated token (since do_sample=False)
        assert first_logits.argmax() == first_new_token, "Greedy token should match logits argmax"

    def test_output_logits_batch_generation(self, gpt2_ht):
        """Test output_logits works with batch inputs."""
        prompts = ["Hello", "World"]
        max_new_tokens = 3

        result = gpt2_ht.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
            output_logits=True,
        )

        # Check batch dimension
        assert result.sequences.shape[0] == len(
            prompts
        ), "Batch dimension should match number of prompts"

        # Check logits batch dimension
        for logit in result.logits:
            assert logit.shape[0] == len(prompts), "Logits batch dimension should match prompts"


class TestTransformerBridgeGenerationModelOutput:
    """Tests for TransformerBridge generation with ModelOutput returns."""

    def test_generate_with_output_logits_returns_modeloutput(self, gpt2_bridge):
        """Test that output_logits=True returns a ModelOutput with sequences and logits."""
        prompt = "The quick brown"
        max_new_tokens = 5

        result = gpt2_bridge.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
            output_logits=True,
        )

        # Check that we got a ModelOutput-like object
        assert hasattr(result, "sequences"), "Result should have sequences attribute"
        assert hasattr(result, "logits"), "Result should have logits attribute"

        # Check sequences shape and type
        assert isinstance(result.sequences, torch.Tensor), "sequences should be a tensor"
        assert result.sequences.ndim == 2, "sequences should be 2D [batch, pos]"

        # Check logits structure and shape
        assert isinstance(result.logits, tuple), "logits should be a tuple"
        assert (
            len(result.logits) == max_new_tokens
        ), f"logits tuple should have {max_new_tokens} elements"

        # Each logit tensor should be [batch, vocab]
        for i, logit in enumerate(result.logits):
            assert isinstance(logit, torch.Tensor), f"logits[{i}] should be a tensor"
            assert logit.ndim == 2, f"logits[{i}] should be 2D [batch, vocab]"
            assert (
                logit.shape[0] == result.sequences.shape[0]
            ), f"logits[{i}] batch size should match sequences"
            assert (
                logit.shape[1] == gpt2_bridge.cfg.d_vocab
            ), f"logits[{i}] vocab size should match model config"

    def test_generate_without_output_logits_returns_normal(self, gpt2_bridge):
        """Test that without output_logits flag, generation returns normal format."""
        prompt = "The quick brown"

        result = gpt2_bridge.generate(
            prompt,
            max_new_tokens=5,
            do_sample=False,
            verbose=False,
        )

        # Should return a string (default return_type="input" with string input)
        assert isinstance(result, str), "Result should be a string"
        assert len(result) > len(prompt), "Generated text should be longer than prompt"

    def test_generate_output_logits_batch(self, gpt2_bridge):
        """Test output_logits works with batch inputs."""
        prompts = ["Hello", "World"]
        max_new_tokens = 3

        result = gpt2_bridge.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            verbose=False,
            output_logits=True,
        )

        # Check ModelOutput structure
        assert hasattr(result, "sequences"), "Result should have sequences"
        assert hasattr(result, "logits"), "Result should have logits"

        # Check batch dimension
        assert result.sequences.shape[0] == len(
            prompts
        ), "Batch dimension should match number of prompts"

        # Check logits batch dimension
        for logit in result.logits:
            assert logit.shape[0] == len(prompts), "Logits batch dimension should match prompts"


class TestTransformerBridgeHFGenerate:
    """Tests for TransformerBridge.hf_generate() with full HF API support."""

    def test_hf_generate_with_output_scores(self, gpt2_bridge):
        """Test that output_scores is forwarded to HF model."""
        prompt = "Test"

        # output_scores should be forwarded without error
        result = gpt2_bridge.hf_generate(
            prompt,
            max_new_tokens=3,
            do_sample=False,
            output_scores=True,
        )

        # Should return a string (default behavior with string input)
        assert isinstance(result, str), "Result should be a string"

    def test_hf_generate_sets_return_dict_in_generate(self, gpt2_bridge):
        """Test that hf_dict_flags automatically set return_dict_in_generate=True."""
        prompt = "Hello"

        # When we pass output_logits, return_dict_in_generate should be auto-set
        # We can't directly inspect the HF call, but we can verify it doesn't error
        result = gpt2_bridge.hf_generate(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            output_logits=True,
        )

        # Should work without error
        assert isinstance(result, str), "Result should be generated successfully"

    def test_hf_generate_multiple_flags_simultaneously(self, gpt2_bridge):
        """Test that multiple HF-style flags can be passed simultaneously."""
        prompt = "Test"

        result = gpt2_bridge.hf_generate(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            output_logits=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        # Should work and return a result
        assert isinstance(result, str), "Result should be generated with multiple flags"

    def test_hf_generate_return_type_tokens(self, gpt2_bridge):
        """Test return_type='tokens' works with HF flags."""
        prompt = "Hello"

        result = gpt2_bridge.hf_generate(
            prompt,
            max_new_tokens=2,
            return_type="tokens",
            do_sample=False,
            output_logits=True,
        )

        # With return_type='tokens', we should get either tokens tensor or ModelOutput
        # The implementation returns the raw HF output for tokens
        assert result is not None, "Result should not be None"

    def test_hf_generate_flags_coerced_to_bool(self, gpt2_bridge):
        """Test that HF flags are properly coerced to boolean values."""
        prompt = "Test"

        # Pass non-boolean values that should be coerced to bool
        result = gpt2_bridge.hf_generate(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            output_logits=1,  # Should be coerced to True
            output_scores=0,  # 0 is not None, so flag is set (coerces to False)
        )

        # Should work without error
        assert isinstance(result, str) or result is not None, "Result should be generated"

    def test_hf_generate_batch_generation(self, gpt2_bridge):
        """Test batch generation works with HF-style flags."""
        prompts = ["Hello", "World"]

        result = gpt2_bridge.hf_generate(
            prompts,
            max_new_tokens=2,
            do_sample=False,
            output_logits=True,
        )

        # Should return list of strings for batch input
        assert isinstance(result, list), "Batch input should return list"
        assert len(result) == len(prompts), "Output list should match input length"


class TestGenerationBackwardCompatibility:
    """Tests to ensure backward compatibility with existing generation usage."""

    def test_hooked_transformer_basic_generation_unchanged(self, gpt2_ht):
        """Test that basic generation without new flags works as before."""
        prompt = "Hello world"

        result = gpt2_ht.generate(
            prompt,
            max_new_tokens=5,
            do_sample=False,
            verbose=False,
        )

        assert isinstance(result, str), "Basic generation should return string"
        assert len(result) > len(prompt), "Generated text should be longer"

    def test_bridge_basic_generation_unchanged(self, gpt2_bridge):
        """Test that basic bridge generation without new flags works as before."""
        prompt = "Hello world"

        result = gpt2_bridge.generate(
            prompt,
            max_new_tokens=5,
            do_sample=False,
            verbose=False,
        )

        assert isinstance(result, str), "Basic generation should return string"
        assert len(result) > len(prompt), "Generated text should be longer"

    def test_hooked_transformer_return_types_unchanged(self, gpt2_ht):
        """Test that all return_type options still work."""
        prompt = "Test"

        # Test return_type='str'
        result_str = gpt2_ht.generate(
            prompt, max_new_tokens=2, return_type="str", verbose=False, do_sample=False
        )
        assert isinstance(result_str, str), "return_type='str' should return string"

        # Test return_type='tokens'
        result_tokens = gpt2_ht.generate(
            prompt, max_new_tokens=2, return_type="tokens", verbose=False, do_sample=False
        )
        assert isinstance(result_tokens, torch.Tensor), "return_type='tokens' should return tensor"

        # Test return_type='embeds'
        result_embeds = gpt2_ht.generate(
            prompt, max_new_tokens=2, return_type="embeds", verbose=False, do_sample=False
        )
        assert isinstance(result_embeds, torch.Tensor), "return_type='embeds' should return tensor"
        assert result_embeds.ndim == 3, "Embeddings should be 3D"


class TestBlockBridgeBatchCompatibility:
    """Tests for BlockBridge tuple return format and batch dimension preservation."""

    def test_block_bridge_batched_generation_compatibility(self, gpt2_bridge):
        """Test BlockBridge maintains tuple format and batch dimensions during generation.

        This test exercises two critical aspects of improved HF compatibility:
        1. BlockBridge.forward() always returns tuples (not bare tensors)
        2. Batch dimensions are preserved through multi-block generation pipeline
        """
        # Test 1: Direct block forward returns tuple with preserved batch dimension
        batch_size = 2
        seq_len = 8
        hidden_dim = gpt2_bridge.cfg.d_model
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        # Get first transformer block (BlockBridge component)
        first_block = gpt2_bridge.original_model.transformer.h[0]

        # Call forward - this is what HF's GPT2Model does in its loop
        block_output = first_block(hidden_states)

        # BlockBridge must return tuple
        assert isinstance(
            block_output, tuple
        ), f"BlockBridge must return tuple for HF compatibility, got {type(block_output)}"

        # Verify first element is a tensor
        assert isinstance(
            block_output[0], torch.Tensor
        ), "First element of BlockBridge output must be a tensor"

        # Batch dimension must be preserved
        # Without tuple wrapping, outputs[0] idx op would turn [batch, seq, dim] -> [seq, dim]
        assert block_output[0].shape == (
            batch_size,
            seq_len,
            hidden_dim,
        ), f"Expected shape [{batch_size}, {seq_len}, {hidden_dim}], got {block_output[0].shape}"

        assert (
            block_output[0].shape[0] == batch_size
        ), f"Batch dimension lost! Expected {batch_size}, got {block_output[0].shape[0]}"

        # Test 2: Batched generation works end-to-end through multiple blocks
        prompts = ["Hello world", "Goodbye world"]

        # Tokenize with left padding
        tokens = gpt2_bridge.to_tokens(prompts, prepend_bos=False, padding_side="left")

        # Generate tokens - this exercises the full HF generation loop with multiple blocks
        output = gpt2_bridge.generate(
            tokens,
            max_new_tokens=4,
            do_sample=False,  # Deterministic for testing
            use_past_kv_cache=True,
            verbose=False,
        )

        # Verify output preserves batch dimension
        assert output.shape[0] == len(
            prompts
        ), f"Batch size must be preserved through generation. Expected {len(prompts)}, got {output.shape[0]}"

        # Verify we actually generated new tokens
        assert (
            output.shape[1] > tokens.shape[1]
        ), "Generation should produce longer sequences than input"

        # Verify batch items remain independent (not collapsed into single item)
        assert not torch.equal(
            output[0], output[1]
        ), "Batch items should be independent - different prompts should produce different outputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
