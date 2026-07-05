"""Unit tests for Bridge.generate() BOS handling and return_input_tokens.

Tests cover:
- prepend_bos parameter being respected (not ignored)
- return_input_tokens flag returning input tokens
- return_input_tokens + return_cache combo
- generate_stream respecting prepend_bos
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def gpt2_bridge():
    """Load a small GPT-2 bridge for testing."""
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


class TestGeneratePrependBos:
    """Test that generate() respects the prepend_bos parameter."""

    def test_prepend_bos_true_adds_bos(self, gpt2_bridge):
        """prepend_bos=True should add BOS token to the input."""
        bridge = gpt2_bridge
        prompt = "Hello"

        _, input_tokens = bridge.generate(
            prompt,
            max_new_tokens=1,
            prepend_bos=True,
            return_input_tokens=True,
            verbose=False,
        )

        assert input_tokens[0, 0].item() == bridge.tokenizer.bos_token_id
        assert input_tokens.shape[1] >= 2  # At least BOS + one token

    def test_prepend_bos_false_no_bos(self, gpt2_bridge):
        """prepend_bos=False should not add BOS token to the input."""
        bridge = gpt2_bridge
        prompt = "Hello"

        _, input_tokens = bridge.generate(
            prompt,
            max_new_tokens=1,
            prepend_bos=False,
            return_input_tokens=True,
            verbose=False,
        )

        # First token should NOT be BOS
        assert input_tokens[0, 0].item() != bridge.tokenizer.bos_token_id

    def test_prepend_bos_difference_is_one_token(self, gpt2_bridge):
        """The difference between prepend_bos=True and False should be exactly 1 token."""
        bridge = gpt2_bridge
        prompt = "Hello"

        _, tokens_with_bos = bridge.generate(
            prompt,
            max_new_tokens=1,
            prepend_bos=True,
            return_input_tokens=True,
            verbose=False,
        )

        _, tokens_no_bos = bridge.generate(
            prompt,
            max_new_tokens=1,
            prepend_bos=False,
            return_input_tokens=True,
            verbose=False,
        )

        assert tokens_with_bos.shape[1] - tokens_no_bos.shape[1] == 1

    def test_prepend_bos_none_uses_default(self, gpt2_bridge):
        """prepend_bos=None should use cfg.default_prepend_bos."""
        bridge = gpt2_bridge
        prompt = "Hello"

        _, tokens_default = bridge.generate(
            prompt,
            max_new_tokens=1,
            prepend_bos=None,
            return_input_tokens=True,
            verbose=False,
        )

        _, tokens_explicit = bridge.generate(
            prompt,
            max_new_tokens=1,
            prepend_bos=bridge.cfg.default_prepend_bos,
            return_input_tokens=True,
            verbose=False,
        )

        assert tokens_default.shape == tokens_explicit.shape
        assert torch.equal(tokens_default, tokens_explicit)

    def test_prepend_bos_ignored_for_tensor_input(self, gpt2_bridge):
        """prepend_bos should be ignored when input is already a token tensor."""
        bridge = gpt2_bridge
        tokens = bridge.to_tokens("Hello", prepend_bos=False)

        # Pass tensor directly - prepend_bos should have no effect
        _, input_tokens_true = bridge.generate(
            tokens,
            max_new_tokens=1,
            prepend_bos=True,
            return_input_tokens=True,
            verbose=False,
        )

        _, input_tokens_false = bridge.generate(
            tokens,
            max_new_tokens=1,
            prepend_bos=False,
            return_input_tokens=True,
            verbose=False,
        )

        # Both should be identical since input was already tokenized
        assert torch.equal(input_tokens_true, input_tokens_false)


class TestReturnInputTokens:
    """Test the return_input_tokens flag on generate()."""

    def test_return_input_tokens_returns_tuple(self, gpt2_bridge):
        """return_input_tokens=True should return (output, input_tokens) tuple."""
        bridge = gpt2_bridge

        result = bridge.generate(
            "Hello",
            max_new_tokens=2,
            return_input_tokens=True,
            verbose=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        output, input_tokens = result
        assert isinstance(input_tokens, torch.Tensor)
        assert input_tokens.dim() == 2  # [batch, seq_len]

    def test_return_input_tokens_false_returns_single(self, gpt2_bridge):
        """return_input_tokens=False should return just the output."""
        bridge = gpt2_bridge

        result = bridge.generate(
            "Hello",
            max_new_tokens=2,
            return_input_tokens=False,
            verbose=False,
        )

        # Should not be a tuple (or if it is, not from return_input_tokens)
        assert not isinstance(result, tuple) or not isinstance(result[1], torch.Tensor)

    def test_return_input_tokens_matches_to_tokens(self, gpt2_bridge):
        """Returned input_tokens should match what to_tokens() would produce."""
        bridge = gpt2_bridge
        prompt = "Hello world"

        _, input_tokens = bridge.generate(
            prompt,
            max_new_tokens=1,
            prepend_bos=True,
            return_input_tokens=True,
            verbose=False,
        )

        expected_tokens = bridge.to_tokens(prompt, prepend_bos=True)

        assert torch.equal(input_tokens, expected_tokens)

    def test_return_input_tokens_with_list_input(self, gpt2_bridge):
        """return_input_tokens should work with list input."""
        bridge = gpt2_bridge

        _, input_tokens = bridge.generate(
            ["Hello", "World"],
            max_new_tokens=1,
            return_input_tokens=True,
            verbose=False,
        )

        assert input_tokens.shape[0] == 2  # Batch size 2


class TestReturnInputTokensWithCache:
    """Test return_input_tokens combined with return_cache."""

    def test_return_cache_and_input_tokens(self, gpt2_bridge):
        """return_cache=True and return_input_tokens=True should return 3-tuple."""
        bridge = gpt2_bridge

        result = bridge.generate(
            "Hi",
            max_new_tokens=2,
            return_cache=True,
            return_input_tokens=True,
            verbose=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        output, cache, input_tokens = result
        assert hasattr(cache, "keys")  # ActivationCache is dict-like
        assert isinstance(input_tokens, torch.Tensor)

    def test_return_cache_only(self, gpt2_bridge):
        """return_cache=True alone should return 2-tuple (output, cache)."""
        bridge = gpt2_bridge

        result = bridge.generate(
            "Hi",
            max_new_tokens=2,
            return_cache=True,
            return_input_tokens=False,
            verbose=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        output, cache = result
        assert hasattr(cache, "keys")  # ActivationCache


class TestGenerateStreamPrependBos:
    """Test that generate_stream() respects the prepend_bos parameter."""

    def test_generate_stream_prepend_bos_true(self, gpt2_bridge):
        """generate_stream with prepend_bos=True should include BOS in first yield."""
        bridge = gpt2_bridge
        prompt = "Hello"

        # Get first yield which includes input tokens
        first_yield = None
        for tokens in bridge.generate_stream(
            prompt,
            max_new_tokens=3,
            prepend_bos=True,
            return_type="tokens",
            verbose=False,
        ):
            first_yield = tokens
            break

        assert first_yield is not None
        assert first_yield[0, 0].item() == bridge.tokenizer.bos_token_id

    def test_generate_stream_prepend_bos_false(self, gpt2_bridge):
        """generate_stream with prepend_bos=False should not include BOS."""
        bridge = gpt2_bridge
        prompt = "Hello"

        first_yield = None
        for tokens in bridge.generate_stream(
            prompt,
            max_new_tokens=3,
            prepend_bos=False,
            return_type="tokens",
            verbose=False,
        ):
            first_yield = tokens
            break

        assert first_yield is not None
        assert first_yield[0, 0].item() != bridge.tokenizer.bos_token_id
