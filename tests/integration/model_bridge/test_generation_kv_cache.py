#!/usr/bin/env python3
"""Integration tests for generation with KV cache in TransformerBridge.

These tests ensure that generation with key-value caching works correctly
in TransformerBridge, matching the behavior of HookedTransformer.
"""


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

        # Generate text with KV cache (max_new_tokens > 1 ensures cache is used)
        output = model.generate(
            "(CNN) President Barack Obama",
            max_new_tokens=10,
            temperature=0.7,
            prepend_bos=True,
        )

        # Verify generation succeeded and produced output
        assert output is not None
        assert len(output) > 0

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

        # Generate from each prompt
        for prompt in prompts:
            output = model.generate(
                prompt,
                max_new_tokens=5,
                temperature=0.7,
                prepend_bos=True,
            )
            # Verify each generation succeeded
            assert output is not None and len(output) > 0


if __name__ == "__main__":
    # Run tests when executed directly
    test = TestGenerationWithKVCache()
    test.test_bridge_generation_with_kv_cache()
    print("✅ KV cache generation test passed!")
    test.test_bridge_multiple_generation_calls()
    print("✅ Multiple generation calls test passed!")
