"""Unit tests for Bridge.generate() KV cache safety and correctness.

Tests cover:
- try/finally cleanup of _capture_hf_cache on exception
- Encoder-decoder rejection of use_past_kv_cache
- _last_hf_cache stores only cache, not full output
"""

from unittest.mock import patch

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


class TestGenerateExceptionSafety:
    """Test that generate() cleans up state even when forward() raises."""

    def test_capture_flag_cleared_on_exception(self, gpt2_bridge):
        """_capture_hf_cache is cleaned up even if forward() raises mid-generation."""
        bridge = gpt2_bridge

        # Verify clean state before
        assert not getattr(bridge, "_capture_hf_cache", False)

        original_forward = bridge.forward

        call_count = 0

        def failing_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Let step 0 succeed, crash on step 1
            if call_count > 1:
                raise RuntimeError("Simulated failure in forward()")
            return original_forward(*args, **kwargs)

        tokens = bridge.to_tokens("Hello world", prepend_bos=False)

        with patch.object(bridge, "forward", side_effect=failing_forward):
            with pytest.raises(RuntimeError, match="Simulated failure"):
                bridge.generate(
                    tokens,
                    max_new_tokens=5,
                    use_past_kv_cache=True,
                    verbose=False,
                )

        # Critical: flag must be cleaned up despite the exception
        assert bridge._capture_hf_cache is False
        assert not hasattr(bridge, "_last_hf_cache")

    def test_forward_clean_after_failed_generate(self, gpt2_bridge):
        """After a failed generate(), the next forward() must not stash caches."""
        bridge = gpt2_bridge
        original_forward = bridge.forward
        call_count = 0

        def failing_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("Simulated failure")
            return original_forward(*args, **kwargs)

        tokens = bridge.to_tokens("Hello world", prepend_bos=False)

        # Trigger a failed generate
        with patch.object(bridge, "forward", side_effect=failing_forward):
            with pytest.raises(RuntimeError):
                bridge.generate(tokens, max_new_tokens=5, use_past_kv_cache=True, verbose=False)

        # Now run a normal forward — should NOT stash any cache
        with torch.no_grad():
            bridge(tokens, return_type="logits")

        assert not hasattr(
            bridge, "_last_hf_cache"
        ), "_last_hf_cache leaked into normal forward() after failed generate()"


class TestGenerateGuards:
    """Test that generate() rejects invalid cache configurations."""

    def test_encoder_decoder_raises(self, gpt2_bridge):
        """Encoder-decoder models should raise ValueError for use_past_kv_cache=True."""
        bridge = gpt2_bridge

        original_config = bridge.original_model.config
        original_config.is_encoder_decoder = True

        try:
            tokens = bridge.to_tokens("Test", prepend_bos=False)
            with pytest.raises(ValueError, match="encoder-decoder"):
                bridge.generate(
                    tokens,
                    max_new_tokens=2,
                    use_past_kv_cache=True,
                    verbose=False,
                )
        finally:
            original_config.is_encoder_decoder = False


class TestForwardCacheExtraction:
    """Test that forward() extracts only past_key_values, not the full output."""

    def test_forward_stashes_only_cache(self, gpt2_bridge):
        """When _capture_hf_cache is True, only past_key_values is stored."""
        bridge = gpt2_bridge
        tokens = bridge.to_tokens("Hello world", prepend_bos=False)

        bridge._capture_hf_cache = True
        try:
            with torch.no_grad():
                bridge(tokens, return_type="logits", use_cache=True)

            # Should have _last_hf_cache, not _last_hf_output
            assert hasattr(bridge, "_last_hf_cache")
            assert not hasattr(bridge, "_last_hf_output")

            # The stashed object should be a cache, not a full model output
            cache = bridge._last_hf_cache
            assert cache is not None
            # DynamicCache or similar should support indexing
            assert len(cache) > 0
        finally:
            bridge._capture_hf_cache = False
            if hasattr(bridge, "_last_hf_cache"):
                del bridge._last_hf_cache


class TestGenerateKVCacheParity:
    """Test that generate with KV cache produces same results as without."""

    def test_greedy_parity(self, gpt2_bridge):
        """Greedy generation with and without KV cache should produce identical tokens."""
        bridge = gpt2_bridge
        prompt = "The quick brown fox"
        max_new = 8

        # Without cache
        result_no_cache = bridge.generate(
            prompt,
            max_new_tokens=max_new,
            do_sample=False,
            use_past_kv_cache=False,
            verbose=False,
            return_type="tokens",
        )

        # With cache
        result_with_cache = bridge.generate(
            prompt,
            max_new_tokens=max_new,
            do_sample=False,
            use_past_kv_cache=True,
            verbose=False,
            return_type="tokens",
        )

        assert torch.equal(result_no_cache, result_with_cache), (
            f"KV cache generation diverged from non-cached:\n"
            f"  no_cache:   {result_no_cache}\n"
            f"  with_cache: {result_with_cache}"
        )

    def test_batched_greedy_parity(self, gpt2_bridge):
        """Batched greedy generation should match between cached and non-cached."""
        bridge = gpt2_bridge
        prompts = ["Hello world", "Goodbye world"]
        max_new = 5

        tokens = bridge.to_tokens(prompts, prepend_bos=False, padding_side="left")

        result_no_cache = bridge.generate(
            tokens,
            max_new_tokens=max_new,
            do_sample=False,
            use_past_kv_cache=False,
            verbose=False,
        )

        result_with_cache = bridge.generate(
            tokens,
            max_new_tokens=max_new,
            do_sample=False,
            use_past_kv_cache=True,
            verbose=False,
        )

        assert torch.equal(
            result_no_cache, result_with_cache
        ), "Batched KV cache generation diverged from non-cached"
