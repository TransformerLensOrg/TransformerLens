"""Cache utilities for TransformerLens.

This module contains utility functions for working with caches, particularly
key-value caches used in transformer models.
"""

from typing import Optional

from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache


def get_pos_offset(past_kv_cache: Optional[TransformerLensKeyValueCache], batch_size: int) -> int:
    """Get position offset for KV cache.

    Args:
        past_kv_cache: Optional KV cache
        batch_size: Batch size

    Returns:
        Position offset
    """
    if past_kv_cache is None:
        return 0
    return past_kv_cache.entries[0].past_keys.shape[1]
