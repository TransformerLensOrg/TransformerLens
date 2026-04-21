"""Key-Value cache for TransformerLens.

Defines the TransformerLensKeyValueCache which manages a list of per-layer
cache entries and attention masks.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union, cast

import torch
from jaxtyping import Int

from transformer_lens.config.TransformerLensConfig import TransformerLensConfig
from transformer_lens.utilities.multi_gpu import get_device_for_block_index

from .key_value_cache_entry import TransformerLensKeyValueCacheEntry

if TYPE_CHECKING:
    from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


@dataclass
class TransformerLensKeyValueCache:
    """
    A cache for storing past keys and values for the Transformer. This is important for generating text - we can cache a lot of past computation and avoid repeating ourselves!

    This cache is a list of TransformerLensKeyValueCacheEntry objects, one for each layer in the Transformer. Each object stores a [batch, pos_so_far, n_heads, d_head] tensor for both keys and values, and each entry has an append method to add a single new key and value.

    The cache can be frozen so that it is not updated during the forward pass. This is useful when we want to run many inputs with the same prefix.
    """

    entries: List[TransformerLensKeyValueCacheEntry]
    previous_attention_mask: Int[torch.Tensor, "batch pos_so_far"]
    frozen: bool = False

    @classmethod
    def init_cache(
        cls,
        cfg: Union[TransformerLensConfig, "HookedTransformerConfig"],
        device: Union[torch.device, str, None],
        batch_size: int = 1,
    ):
        # Determine device for each layer
        if hasattr(cfg, "n_devices"):
            # HookedTransformer case: use our multi-GPU logic
            device_for_layer = lambda i: get_device_for_block_index(
                i, cast("HookedTransformerConfig", cfg), device
            )
        else:
            # Fallback when no model is provided - use single device
            fallback_device = device if device is not None else cfg.device
            if fallback_device is None:
                fallback_device = torch.device("cpu")
            device_for_layer = lambda i: fallback_device

        return cls(
            entries=[
                TransformerLensKeyValueCacheEntry.init_cache_entry(
                    cfg,
                    device_for_layer(i),
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
