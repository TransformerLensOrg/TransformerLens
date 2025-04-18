"""Hooked Transformer RMS Norm Component.

This module contains all the component :class:`RMSNorm`.
"""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class RMSNormScaled(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], length: Optional[int] = None):
        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.eps = self.cfg.eps
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length
        self.scale = self.length ** -0.5 

        self.w = nn.Parameter(torch.zeros(self.length, dtype=self.cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.hook_normalized(x)
        x = self.scale * x * self.w.to(x.dtype)
        x = self.hook_scale(x)
        return x
