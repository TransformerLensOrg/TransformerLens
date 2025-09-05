"""Past Key Value Caching (compatibility shim).

This module historically defined the key-value cache classes. They have been moved to
`transformer_lens.cache`. Importing from here re-exports the new implementations.
"""

from transformer_lens.cache.key_value_cache import KeyValueCache
from transformer_lens.cache.key_value_cache_entry import KeyValueCacheEntry

__all__ = [
    "KeyValueCacheEntry",
    "KeyValueCache",
]
