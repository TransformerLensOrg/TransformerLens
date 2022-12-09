"""A cache for storing past keys and values for the Transformer. This is important for generating text - we can cache a lot of past computation and avoid repeating ourselves!"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchtyping import TensorType as TT

from easy_transformer.EasyTransformerConfig import EasyTransformerConfig


@dataclass
class EasyTransformerKeyValueCacheEntry:
    """A cache entry for a single layer in the Transformer. Stores a [batch, pos_so_far, n_heads, d_head] tensor for both keys and values, and has an append method to add a single new key and value."""

    past_keys: TT["batch", "pos_so_far", "n_heads", "d_head"]
    past_values: TT["batch", "pos_so_far", "n_heads", "d_head"]

    @classmethod
    def init_cache_entry(
        cls,
        cfg: EasyTransformerConfig,
        device: torch.device,
        batch_size: int = 1,
    ):
        """Returns a cache entry with an empty cache."""
        return cls(
            past_keys=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
            past_values=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
        )

    def append(
        self,
        new_keys: TT["batch", "new_tokens", "n_heads", "d_head"],
        new_values: TT["batch", "new_tokens", "n_heads", "d_head"],
    ):
        """Appends keys and values to the cache. Returns the new keys and values."""
        updated_keys: TT[
            "batch", "pos_so_far + new_tokens", "n_heads", "d_head"
        ] = torch.cat([self.past_keys, new_keys], dim=1)
        updated_values: TT[
            "batch", "pos_so_far + new_tokens", "n_heads", "d_head"
        ] = torch.cat([self.past_values, new_values], dim=1)
        self.past_keys = updated_keys
        self.past_values = updated_values
        return updated_keys, updated_values


@dataclass
class EasyTransformerKeyValueCache:
    """
    A cache for storing past keys and values for the Transformer. This is important for generating text - we can cache a lot of past computation and avoid repeating ourselves!

    This cache is a list of EasyTransformerKeyValueCacheEntry objects, one for each layer in the Transformer. Each object stores a [batch, pos_so_far, n_heads, d_head] tensor for both keys and values, and each entry has an append method to add a single new key and value.

    Generation is assumed to be done by initializing with some prompt and then continuing iteratively one token at a time. So append only works for adding a single token's worth of keys and values, and but the cache can be initialized with many.
    """

    entries: List[EasyTransformerKeyValueCacheEntry]

    @classmethod
    def init_cache(
        cls, cfg: EasyTransformerConfig, device: torch.device, batch_size: int = 1
    ):
        """Returns a cache with the correct number of layers, each initialized with an empty cache entry."""
        return cls(
            entries=[
                EasyTransformerKeyValueCacheEntry.init_cache_entry(
                    cfg, device, batch_size
                )
                for _ in range(cfg.n_layers)
            ]
        )

    def __getitem__(self, idx):
        """Returns the cache entry at the given index."""
        return self.entries[idx]
