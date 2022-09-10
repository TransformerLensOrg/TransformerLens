# %%
from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn

# %%
@dataclass
class EasyTransformerConfig:
    model_name: str
    model_type: str
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    n_layers: int
    n_ctx: int
    eps: float
    d_vocab: int
    act_fn: str
    use_attn_result: bool = False
    use_attn_scale: bool = True
    use_local_attn: bool = False
    checkpoint: Optional[int] = None
    full_model_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.use_local_attn:
            assert (
                self.window_size is not None
            ), "window_size must be specified for local attention"
            assert (
                self.attn_types is not None
            ), "attn_types must be specified for local attention"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)


# %%
