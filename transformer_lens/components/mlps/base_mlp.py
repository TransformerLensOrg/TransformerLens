"""Base Transformer MLP Component.

This module contains the shared functionality between all MLPs
"""
from typing import Callable, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import gelu_fast, gelu_new, solu

class BaseMLP(nn.Module):
    act_fn: Callable[..., torch.Tensor]
    ln: nn.Module
    
    def __init__(self, config: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(config)
        assert self.cfg.d_mlp is not None
        
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=self.cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]
