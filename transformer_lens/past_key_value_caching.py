"""Past Key Value Caching.

This module contains the HookedTransformerKeyValueCache and HookedTransformerKeyValueCacheEntry
classes, which are used to store past keys and values for the Transformer. This is important for
generating text - we can cache a lot of past computation and avoid repeating ourselves!
"""
from dataclasses import dataclass
from typing import List, Union

import torch
from jaxtyping import Float, Int

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities.devices import get_device_for_block_index


@dataclass
class HookedTransformerKeyValueCacheEntry:
    past_keys: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]
    past_values: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]
    frozen: bool = False

    @classmethod
    def init_cache_entry(
        cls,
        cfg: HookedTransformerConfig,
        device: Union[torch.device, str, None],
        batch_size: int = 1,
    ):
        n_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else cfg.n_heads
        return cls(
            past_keys=torch.empty(
                (batch_size, 0, n_heads, cfg.d_head), device=device, dtype=cfg.dtype
            ),
            past_values=torch.empty(
                (batch_size, 0, n_heads, cfg.d_head), device=device, dtype=cfg.dtype
            ),
        )

    def append(
        self,
        new_keys: Float[torch.Tensor, "batch new_tokens n_heads d_head"],
        new_values: Float[torch.Tensor, "batch new_tokens n_heads d_head"],
    ):
        updated_keys: Float[
            torch.Tensor, "batch pos_so_far_plus_new_tokens n_heads d_head"
        ] = torch.cat([self.past_keys, new_keys], dim=1)
        updated_values: Float[
            torch.Tensor, "batch pos_so_far_plus_new_tokens n_heads d_head"
        ] = torch.cat([self.past_values, new_values], dim=1)
        if not self.frozen:
            self.past_keys = updated_keys
            self.past_values = updated_values
        return updated_keys, updated_values


@dataclass
class HookedTransformerKeyValueCache:
    """
    A cache for storing past keys and values for the Transformer. This is important for generating text - we can cache a lot of past computation and avoid repeating ourselves!

    This cache is a list of HookedTransformerKeyValueCacheEntry objects, one for each layer in the Transformer. Each object stores a [batch, pos_so_far, n_heads, d_head] tensor for both keys and values, and each entry has an append method to add a single new key and value.

    The cache can be frozen so that it is not updated during the forward pass. This is useful when we want to run many inputs with the same prefix.
    """

    entries: List[HookedTransformerKeyValueCacheEntry]
    previous_attention_mask: Int[torch.Tensor, "batch pos_so_far"]
    frozen: bool = False

    @classmethod
    def init_cache(
        cls,
        cfg: HookedTransformerConfig,
        device: Union[torch.device, str, None],
        batch_size: int = 1,
    ):
        return cls(
            entries=[
                HookedTransformerKeyValueCacheEntry.init_cache_entry(
                    cfg,
                    get_device_for_block_index(i, cfg, device),
                    batch_size,
                )
                for i in range(cfg.n_layers)
            ],
            previous_attention_mask=torch.empty(
                # This may actually be an int64, but type promotion will handle it:
                # See: https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc
                # See: https://github.com/pytorch/pytorch/issues/35014
                (batch_size, 0),
                device=device,
                dtype=torch.int,
            ),
        )

    def freeze(self):
        self.frozen = True
        for entry in self.entries:
            entry.frozen = True

    def unfreeze(self):
        self.frozen = False
        for entry in self.entries:
            entry.frozen = False

    def append_attention_mask(self, attention_mask: Int[torch.Tensor, "batch new_tokens"]):
        attention_mask = attention_mask.to(self.previous_attention_mask.device)
        updated_attention_mask = torch.cat([self.previous_attention_mask, attention_mask], dim=-1)
        if not self.frozen:
            self.previous_attention_mask = updated_attention_mask
        return updated_attention_mask

    def __getitem__(self, idx):
        return self.entries[idx]
