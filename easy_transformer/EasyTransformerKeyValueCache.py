import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig


@dataclass
class EasyTransformerKeyValueCacheEntry:
    past_keys: torch.Tensor
    past_values: torch.Tensor

    @classmethod
    def init_cache_entry(
        cls,
        cfg: EasyTransformerConfig,
        device: torch.device,
        batch_size: int = 1,
    ):
        return cls(
            past_keys=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
            past_values=torch.empty(
                (batch_size, 0, cfg.n_heads, cfg.d_head), device=device
            ),
        )

    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        self.past_keys = torch.cat([self.past_keys, new_keys], dim=1)
        self.past_values = torch.cat([self.past_values, new_values], dim=1)
        return self.past_keys, self.past_values


@dataclass
class EasyTransformerKeyValueCache:
    entries: List[EasyTransformerKeyValueCacheEntry]

    @classmethod
    def init_cache(
        cls, cfg: EasyTransformerConfig, device: torch.device, batch_size: int = 1
    ):
        return cls(
            entries=[
                EasyTransformerKeyValueCacheEntry.init_cache_entry(
                    cfg, device, batch_size
                )
                for _ in range(cfg.n_layers)
            ]
        )

    def __getitem__(self, idx):
        return self.entries[idx]
