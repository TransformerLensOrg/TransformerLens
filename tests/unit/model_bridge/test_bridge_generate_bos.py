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
def distilgpt2_bridge():
    """Distilgpt2 bridge with pad_token set — local so the tokenizer mutation stays contained."""
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    if bridge.tokenizer.pad_token is None:
        bridge.tokenizer.pad_token = bridge.tokenizer.eos_token
    return bridge


class TestGeneratePrependBos:
    """Test that generate() respects the prepend_bos parameter."""

    def test_prepend_bos_true_adds_bos(self, distilgpt2_bridge):
        """prepend_bos=True should add BOS token to the input."""
        bridge = distilgpt2_bridge
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

    def test_prepend_bos_false_no_bos(self, distilgpt2_bridge):
        """prepend_bos=False should not add BOS token to the input."""
        bridge = distilgpt2_bridge
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

    def test_prepend_bos_difference_is_one_token(self, distilgpt2_bridge):
        """The difference between prepend_bos=True and False should be exactly 1 token."""
        bridge = distilgpt2_bridge
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

    def test_prepend_bos_none_uses_default(self, distilgpt2_bridge):
        """prepend_bos=None should use cfg.default_prepend_bos."""
        bridge = distilgpt2_bridge
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

    def test_prepend_bos_ignored_for_tensor_input(self, distilgpt2_bridge):
        """prepend_bos should be ignored when input is already a token tensor."""
        bridge = distilgpt2_bridge
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

    def test_return_input_tokens_returns_tuple(self, distilgpt2_bridge):
        """return_input_tokens=True should return (output, input_tokens) tuple."""
        bridge = distilgpt2_bridge

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

    def test_return_input_tokens_false_returns_single(self, distilgpt2_bridge):
        """return_input_tokens=False should return just the output."""
        bridge = distilgpt2_bridge

        result = bridge.generate(
            "Hello",
            max_new_tokens=2,
            return_input_tokens=False,
            verbose=False,
        )

        # Should not be a tuple (or if it is, not from return_input_tokens)
        assert not isinstance(result, tuple) or not isinstance(result[1], torch.Tensor)

    def test_return_input_tokens_matches_to_tokens(self, distilgpt2_bridge):
        """Returned input_tokens should match what to_tokens() would produce."""
        bridge = distilgpt2_bridge
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

    def test_return_input_tokens_with_list_input(self, distilgpt2_bridge):
        """return_input_tokens should work with list input."""
        bridge = distilgpt2_bridge

        _, input_tokens = bridge.generate(
            ["Hello", "World"],
            max_new_tokens=1,
            return_input_tokens=True,
            verbose=False,
        )

        assert input_tokens.shape[0] == 2  # Batch size 2


class TestReturnInputTokensWithCache:
    """Test return_input_tokens combined with return_cache."""

    def test_return_cache_and_input_tokens(self, distilgpt2_bridge):
        """return_cache=True and return_input_tokens=True should return 3-tuple."""
        bridge = distilgpt2_bridge

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

    def test_return_cache_only(self, distilgpt2_bridge):
        """return_cache=True alone should return 2-tuple (output, cache)."""
        bridge = distilgpt2_bridge

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

    def test_generate_stream_prepend_bos_true(self, distilgpt2_bridge):
        """generate_stream with prepend_bos=True should include BOS in first yield."""
        bridge = distilgpt2_bridge
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

    def test_generate_stream_prepend_bos_false(self, distilgpt2_bridge):
        """generate_stream with prepend_bos=False should not include BOS."""
        bridge = distilgpt2_bridge
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


@pytest.mark.parametrize("stream", [False, True], ids=["generate", "generate_stream"])
class TestBatchedPromptMask:
    """A batched row must continue exactly as its prompt does alone.

    distilgpt2's BOS is also its pad token, so a left-padded row's tokens can't show
    where its padding ends.
    """

    PROMPTS = ["The capital of France is a very", "Hello"]
    KWARGS = {
        "max_new_tokens": 6,
        "do_sample": False,
        "stop_at_eos": False,
        "return_type": "tokens",
        "verbose": False,
    }

    def _run(self, bridge, stream, prompts, prepend_bos):
        if stream:
            chunks = bridge.generate_stream(
                prompts, prepend_bos=prepend_bos, max_tokens_per_yield=100, **self.KWARGS
            )
            return torch.cat(list(chunks), dim=-1)
        return bridge.generate(prompts, prepend_bos=prepend_bos, **self.KWARGS)

    def _assert_rows_match_single_prompts(self, bridge, stream, prompts, prepend_bos):
        batched = self._run(bridge, stream, prompts, prepend_bos)
        for row, prompt in enumerate(prompts):
            single = self._run(bridge, stream, prompt, prepend_bos)
            assert torch.equal(batched[row, -single.shape[-1] :], single[0])

    def test_prepend_bos_false_attends_no_pad(self, distilgpt2_bridge, stream):
        self._assert_rows_match_single_prompts(distilgpt2_bridge, stream, self.PROMPTS, False)

    def test_prepend_bos_overrides_cfg_default(self, distilgpt2_bridge, stream, monkeypatch):
        monkeypatch.setattr(distilgpt2_bridge.cfg, "default_prepend_bos", False)
        self._assert_rows_match_single_prompts(distilgpt2_bridge, stream, self.PROMPTS, True)

    def test_literal_bos_stays_attended(self, distilgpt2_bridge, stream):
        """Pre-formatted text carries its own BOS, which has the pad token's id here."""
        bos = distilgpt2_bridge.tokenizer.bos_token
        prompts = [bos + prompt for prompt in self.PROMPTS]
        self._assert_rows_match_single_prompts(distilgpt2_bridge, stream, prompts, False)


def test_empty_prompt_attends_only_its_last_pad(distilgpt2_bridge):
    prompts = ["", "Hello there friend"]
    tokens = distilgpt2_bridge.to_tokens(
        prompts, prepend_bos=False, padding_side="left", truncate=False
    )

    mask = distilgpt2_bridge._left_padded_prompt_mask(prompts, tokens, False)

    assert mask.tolist() == [[0, 0, 1], [1, 1, 1]]
