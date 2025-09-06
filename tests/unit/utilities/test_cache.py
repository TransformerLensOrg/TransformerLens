"""Unit tests for cache utilities.

This module tests the cache utility functions, particularly the get_pos_offset function.
"""

import torch

from transformer_lens.cache.key_value_cache import (
    TransformerLensKeyValueCache as HookedTransformerKeyValueCache,
)
from transformer_lens.cache.key_value_cache_entry import (
    TransformerLensKeyValueCacheEntry as HookedTransformerKeyValueCacheEntry,
)
from transformer_lens.utilities.cache import get_pos_offset


class TestGetPosOffset:
    """Test cases for the get_pos_offset function."""

    def test_get_pos_offset_none_cache(self):
        """Test get_pos_offset with None cache."""
        result = get_pos_offset(None, batch_size=1)
        assert result == 0
        assert isinstance(result, int)

    def test_get_pos_offset_none_cache_different_batch_sizes(self):
        """Test get_pos_offset with None cache and different batch sizes."""
        for batch_size in [1, 2, 10, 100]:
            result = get_pos_offset(None, batch_size=batch_size)
            assert result == 0
            assert isinstance(result, int)

    def test_get_pos_offset_with_empty_cache(self):
        """Test get_pos_offset with an empty cache (0 positions)."""
        # Create a real cache entry with 0 positions
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((2, 0, 8, 64)), past_values=torch.empty((2, 0, 8, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((2, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=2)
        assert result == 0
        assert isinstance(result, int)

    def test_get_pos_offset_with_single_position_cache(self):
        """Test get_pos_offset with a cache containing 1 position."""
        # Create a real cache entry with 1 position
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((1, 1, 12, 64)), past_values=torch.empty((1, 1, 12, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((1, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=1)
        assert result == 1
        assert isinstance(result, int)

    def test_get_pos_offset_with_multiple_positions_cache(self):
        """Test get_pos_offset with a cache containing multiple positions."""
        # Create a real cache entry with multiple positions
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((4, 10, 16, 128)), past_values=torch.empty((4, 10, 16, 128))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((4, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=4)
        assert result == 10
        assert isinstance(result, int)

    def test_get_pos_offset_with_large_cache(self):
        """Test get_pos_offset with a large cache."""
        # Create a real cache entry with many positions
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((8, 1024, 32, 256)), past_values=torch.empty((8, 1024, 32, 256))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((8, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=8)
        assert result == 1024
        assert isinstance(result, int)

    def test_get_pos_offset_with_real_cache_entry(self):
        """Test get_pos_offset with a real HookedTransformerKeyValueCacheEntry."""
        # Create a real cache entry
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((2, 5, 8, 64)), past_values=torch.empty((2, 5, 8, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((2, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=2)
        assert result == 5
        assert isinstance(result, int)

    def test_get_pos_offset_with_real_cache_multiple_entries(self):
        """Test get_pos_offset with a real cache containing multiple entries."""
        # Create multiple real cache entries
        cache_entries = [
            HookedTransformerKeyValueCacheEntry(
                past_keys=torch.empty((2, 3, 8, 64)), past_values=torch.empty((2, 3, 8, 64))
            ),
            HookedTransformerKeyValueCacheEntry(
                past_keys=torch.empty((2, 3, 8, 64)), past_values=torch.empty((2, 3, 8, 64))
            ),
            HookedTransformerKeyValueCacheEntry(
                past_keys=torch.empty((2, 3, 8, 64)), past_values=torch.empty((2, 3, 8, 64))
            ),
        ]

        cache = HookedTransformerKeyValueCache(
            entries=cache_entries, previous_attention_mask=torch.empty((2, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=2)
        assert result == 3  # Should use the first entry
        assert isinstance(result, int)

    def test_get_pos_offset_batch_size_parameter_ignored(self):
        """Test that the batch_size parameter is ignored when cache is provided."""
        # Create a real cache entry
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((2, 7, 8, 64)), past_values=torch.empty((2, 7, 8, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((2, 0), dtype=torch.int)
        )

        # Test with different batch_size values - should all return the same result
        for batch_size in [1, 2, 10, 100]:
            result = get_pos_offset(cache, batch_size=batch_size)
            assert result == 7
            assert isinstance(result, int)

    def test_get_pos_offset_type_hints(self):
        """Test that the function accepts the correct types."""
        # Test with None
        result = get_pos_offset(None, batch_size=1)
        assert isinstance(result, int)

        # Test with real cache
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((1, 5, 8, 64)), past_values=torch.empty((1, 5, 8, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((1, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=1)
        assert isinstance(result, int)

    def test_get_pos_offset_edge_cases(self):
        """Test get_pos_offset with edge cases."""
        # Test with very large position count
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((1, 999999, 8, 64)), past_values=torch.empty((1, 999999, 8, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((1, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=1)
        assert result == 999999
        assert isinstance(result, int)

        # Test with zero batch size in cache
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((0, 5, 8, 64)), past_values=torch.empty((0, 5, 8, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((0, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=0)
        assert result == 5
        assert isinstance(result, int)

    def test_get_pos_offset_documentation_examples(self):
        """Test get_pos_offset with examples that might be in documentation."""
        # Example 1: No cache
        result = get_pos_offset(None, batch_size=1)
        assert result == 0

        # Example 2: Cache with some positions
        cache_entry = HookedTransformerKeyValueCacheEntry(
            past_keys=torch.empty((1, 10, 8, 64)), past_values=torch.empty((1, 10, 8, 64))
        )

        cache = HookedTransformerKeyValueCache(
            entries=[cache_entry], previous_attention_mask=torch.empty((1, 0), dtype=torch.int)
        )

        result = get_pos_offset(cache, batch_size=1)
        assert result == 10
