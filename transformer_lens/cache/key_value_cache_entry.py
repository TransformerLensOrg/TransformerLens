"""Key-Value cache entry for TransformerLens.

This module defines the TransformerLensKeyValueCacheEntry class which stores
past keys and values for a single transformer layer.
"""

from dataclasses import dataclass
from typing import Union

import torch
from jaxtyping import Float

from transformer_lens.config.TransformerLensConfig import TransformerLensConfig


@dataclass
class TransformerLensKeyValueCacheEntry:
    past_keys: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]
    past_values: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]
    frozen: bool = False

    @classmethod
    def init_cache_entry(
        cls,
        cfg: TransformerLensConfig,
        device: Union[torch.device, str, None],
        batch_size: int = 1,
    ):
        n_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else cfg.n_heads
        return cls(
            past_keys=torch.empty(
                (batch_size, 0, n_heads, cfg.d_head), device=device, dtype=torch.get_default_dtype()
            ),
            past_values=torch.empty(
                (batch_size, 0, n_heads, cfg.d_head), device=device, dtype=torch.get_default_dtype()
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
