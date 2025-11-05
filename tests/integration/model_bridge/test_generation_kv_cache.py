#!/usr/bin/env python3
"""Integration tests for generation with KV cache in TransformerBridge.

These tests ensure that generation with key-value caching works correctly
in TransformerBridge, matching the behavior of HookedTransformer.
"""

from transformer_lens.benchmarks import (
    benchmark_generation_with_kv_cache,
    benchmark_multiple_generation_calls,
)
from transformer_lens.model_bridge import TransformerBridge


class TestGenerationWithKVCache:
    """Test generation with KV cache in TransformerBridge."""

    def test_bridge_generation_with_kv_cache(self):
        """Test that TransformerBridge can generate text with KV caching enabled.

        This test ensures that the KV cache (DynamicCache) is properly passed through
        the attention layers during generation, and that the cache update logic works correctly.

        Regression test for: RuntimeError: Expected size for first two dimensions of batch2
        tensor to be: [12, 13] but got: [12, 1] when DynamicCache was being evaluated as
        False in boolean context.
        """
        # Create model with TransformerBridge
        model = TransformerBridge.boot_transformers("gpt2", device="cpu")
        model.enable_compatibility_mode()

        # Use benchmark function
        result = benchmark_generation_with_kv_cache(
            model, "(CNN) President Barack Obama", max_new_tokens=10
        )
        assert result.passed, result.message

    def test_bridge_multiple_generation_calls(self):
        """Test that TransformerBridge can generate multiple times without errors.

        This ensures the KV cache handling is robust across multiple generate() calls.
        """
        # Create model
        model = TransformerBridge.boot_transformers("gpt2", device="cpu")
        model.enable_compatibility_mode()

        prompts = [
            "The quick brown fox",
            "Hello world",
            "Machine learning is",
        ]

        # Use benchmark function
        result = benchmark_multiple_generation_calls(model, prompts, max_new_tokens=5)
        assert result.passed, result.message


if __name__ == "__main__":
    # Run tests when executed directly
    test = TestGenerationWithKVCache()
    test.test_bridge_generation_with_kv_cache()
    print("✅ KV cache generation test passed!")
    test.test_bridge_multiple_generation_calls()
    print("✅ Multiple generation calls test passed!")
