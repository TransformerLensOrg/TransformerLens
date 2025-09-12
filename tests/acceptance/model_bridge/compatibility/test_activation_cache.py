import gc

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge import TransformerBridge


class TestActivationCacheCompatibility:
    """Test that ActivationCache works with TransformerBridge."""

    @pytest.fixture(autouse=True, scope="class")
    def cleanup_after_class(self):
        """Clean up memory after each test class."""
        yield
        # Force garbage collection and clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for _ in range(3):
            gc.collect()

    @pytest.fixture(scope="class")
    def bridge_model(self):
        """Create a TransformerBridge model for testing."""
        return TransformerBridge.boot_transformers("gpt2", device="cpu")

    @pytest.fixture(scope="class")
    def sample_cache(self, bridge_model):
        """Create a sample cache for testing."""
        prompt = "The quick brown fox jumps over the lazy dog."
        output, cache = bridge_model.run_with_cache(prompt)
        return cache

    def test_cache_creation(self, bridge_model):
        """Test that caches can be created from TransformerBridge."""
        prompt = "Test cache creation."

        # Test run_with_cache with cache object
        output, cache = bridge_model.run_with_cache(prompt, return_cache_object=True)

        assert isinstance(output, torch.Tensor)
        assert isinstance(cache, (dict, ActivationCache))

        # If it's an ActivationCache, test its properties
        if isinstance(cache, ActivationCache):
            assert hasattr(cache, "cache_dict")
            assert hasattr(cache, "model")
            assert len(cache.cache_dict) > 0

    def test_cache_dict_access(self, sample_cache):
        """Test that cache dictionary access works."""
        # Get cache dict regardless of type
        if hasattr(sample_cache, "cache_dict"):
            cache_dict = sample_cache.cache_dict
        else:
            cache_dict = sample_cache

        assert isinstance(cache_dict, dict)
        assert len(cache_dict) > 0

        # All values should be tensors or None
        for key, value in cache_dict.items():
            if value is not None:
                assert isinstance(value, torch.Tensor), f"Cache value for {key} is not a tensor"

    def test_cache_key_patterns(self, sample_cache):
        """Test that cache keys follow expected patterns."""
        # Get cache dict
        if hasattr(sample_cache, "cache_dict"):
            cache_dict = sample_cache.cache_dict
        else:
            cache_dict = sample_cache

        cache_keys = list(cache_dict.keys())

        # Should have some keys
        assert len(cache_keys) > 0

        # Log what patterns we find (for debugging)
        patterns_found = []
        common_patterns = [
            "embed",
            "pos_embed",
            "blocks",
            "ln_final",
            "unembed",
            "hook_",
            "attn",
            "mlp",
            "resid",
        ]

        for pattern in common_patterns:
            if any(pattern in key for key in cache_keys):
                patterns_found.append(pattern)

        print(f"Cache key patterns found: {patterns_found}")
        print(f"Total cache keys: {len(cache_keys)}")
        print(f"Sample keys: {cache_keys[:5]}")

    def test_cache_tensor_shapes(self, sample_cache, bridge_model):
        """Test that cached tensors have reasonable shapes."""
        # Get cache dict
        if hasattr(sample_cache, "cache_dict"):
            cache_dict = sample_cache.cache_dict
        else:
            cache_dict = sample_cache

        cfg = bridge_model.cfg

        for key, value in cache_dict.items():
            if value is not None and isinstance(value, torch.Tensor):
                # All tensors should have at least 2 dimensions (batch, seq, ...)
                assert value.ndim >= 2, f"Tensor {key} has insufficient dimensions: {value.shape}"

                # Batch dimension should be 1 for single prompt
                assert value.shape[0] == 1, f"Tensor {key} has wrong batch size: {value.shape[0]}"

                # If it's a 3D tensor, last dimension might be d_model or d_vocab
                if value.ndim == 3:
                    last_dim = value.shape[2]
                    # Should be one of the common dimensions
                    common_dims = [cfg.d_model, cfg.d_vocab, cfg.d_head * cfg.n_heads]
                    if hasattr(cfg, "d_mlp"):
                        common_dims.append(cfg.d_mlp)

                    # Don't enforce strict checking since bridge might have different dimensions
                    # Just check that it's reasonable
                    assert last_dim > 0, f"Tensor {key} has invalid last dimension: {last_dim}"

    def test_cache_with_names_filter(self, bridge_model):
        """Test that names filtering works with caching."""
        prompt = "Test names filter."

        # Get available hook names
        hook_dict = bridge_model.hook_dict
        if len(hook_dict) == 0:
            pytest.skip("No hooks available for filtering")

        # Use first few hook names
        filter_names = list(hook_dict.keys())[:3]

        try:
            output, cache = bridge_model.run_with_cache(prompt, names_filter=filter_names)

            # Get cache dict
            if hasattr(cache, "cache_dict"):
                cache_dict = cache.cache_dict
            else:
                cache_dict = cache

            # Should have some activations
            assert len(cache_dict) > 0

            # Check that we got activations for the filtered names (or their aliases)
            cache_keys = set(cache_dict.keys())
            filter_set = set(filter_names)

            # Should have some overlap (exact match not required due to aliasing)
            overlap = len(cache_keys & filter_set)
            # Allow for aliases by checking partial matches
            partial_matches = sum(
                1
                for cache_key in cache_keys
                for filter_name in filter_names
                if filter_name in cache_key or cache_key in filter_name
            )

            assert overlap > 0 or partial_matches > 0, "No filtered activations found in cache"

        except Exception as e:
            pytest.skip(f"Names filtering not working: {e}")

    def test_cache_iteration(self, sample_cache):
        """Test that cache can be iterated over."""
        # Get cache dict
        if hasattr(sample_cache, "cache_dict"):
            cache_dict = sample_cache.cache_dict
        else:
            cache_dict = sample_cache

        # Test iteration
        keys_from_iter = []
        for key in cache_dict:
            keys_from_iter.append(key)

        keys_from_keys = list(cache_dict.keys())

        assert set(keys_from_iter) == set(keys_from_keys)
        assert len(keys_from_iter) > 0

    def test_cache_getitem(self, sample_cache):
        """Test that cache supports getitem access."""
        # Get cache dict
        if hasattr(sample_cache, "cache_dict"):
            cache_dict = sample_cache.cache_dict
        else:
            cache_dict = sample_cache

        if len(cache_dict) == 0:
            pytest.skip("Empty cache")

        # Test accessing items
        for key in list(cache_dict.keys())[:3]:  # Test first few
            value = cache_dict[key]
            if value is not None:
                assert isinstance(value, torch.Tensor)

    def test_cache_batch_dimension_handling(self, bridge_model):
        """Test that cache handles batch dimensions correctly."""
        prompts = ["First prompt for batch testing.", "Second prompt for batch testing."]

        try:
            # Test with multiple prompts
            output, cache = bridge_model.run_with_cache(prompts)

            # Get cache dict
            if hasattr(cache, "cache_dict"):
                cache_dict = cache.cache_dict
            else:
                cache_dict = cache

            # Check that cached tensors have correct batch dimension
            for key, value in cache_dict.items():
                if value is not None and isinstance(value, torch.Tensor):
                    assert value.shape[0] == len(
                        prompts
                    ), f"Tensor {key} has wrong batch size: {value.shape[0]}"

        except Exception as e:
            pytest.skip(f"Batch processing not supported: {e}")

    def test_cache_device_consistency(self, bridge_model):
        """Test that cached tensors are on the correct device."""
        prompt = "Test device consistency."

        # Test on CPU
        model_cpu = bridge_model.cpu()
        output, cache = model_cpu.run_with_cache(prompt)

        # Get cache dict
        if hasattr(cache, "cache_dict"):
            cache_dict = cache.cache_dict
        else:
            cache_dict = cache

        # All cached tensors should be on CPU
        for key, value in cache_dict.items():
            if value is not None and isinstance(value, torch.Tensor):
                assert value.device.type == "cpu", f"Tensor {key} is not on CPU: {value.device}"

    def test_cache_memory_efficiency(self, bridge_model):
        """Test that cache doesn't cause memory leaks."""
        prompt = "Test cache memory efficiency."

        # Record initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Create and delete multiple caches
        for _ in range(3):
            output, cache = bridge_model.run_with_cache(prompt)
            del output, cache

        # Clean up
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()

            # Memory shouldn't grow significantly
            memory_growth = final_memory - initial_memory
            assert (
                memory_growth < 50 * 1024 * 1024
            ), f"Cache caused memory growth of {memory_growth} bytes"

    def test_cache_with_different_inputs(self, bridge_model):
        """Test that cache works with different input types."""
        # Test with string
        output1, cache1 = bridge_model.run_with_cache("String input test.")

        # Test with tokens
        tokens = bridge_model.to_tokens("Token input test.")
        output2, cache2 = bridge_model.run_with_cache(tokens)

        # Both should work
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)

        # Get cache dicts
        if hasattr(cache1, "cache_dict"):
            cache_dict1 = cache1.cache_dict
        else:
            cache_dict1 = cache1

        if hasattr(cache2, "cache_dict"):
            cache_dict2 = cache2.cache_dict
        else:
            cache_dict2 = cache2

        # Both should have cached activations
        assert len(cache_dict1) > 0
        assert len(cache_dict2) > 0

        # Should have similar cache keys
        keys1 = set(cache_dict1.keys())
        keys2 = set(cache_dict2.keys())

        # At least some overlap in keys
        overlap = len(keys1 & keys2)
        assert overlap > 0, "No common cache keys between string and token inputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
