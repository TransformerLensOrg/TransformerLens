"""Hooked Transformer RMS Norm Component.

This module contains all the component :class:`RMSNorm`.
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

# RMSNorm operates on the last dimension and supports both 2D and 3D inputs.
# The 2D case arises when callers (e.g. QK normalization) reshape before normalizing.
RMSNormInput = Union[
    Float[torch.Tensor, "batch pos length"],
    Float[torch.Tensor, "batch_pos length"],
]


class RMSNorm(nn.Module):
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

        self.w = nn.Parameter(torch.ones(self.length, dtype=self.cfg.dtype))

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(self, x: RMSNormInput) -> RMSNormInput:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)
        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        x = self.hook_normalized(x / scale).to(self.cfg.dtype)  # [batch, pos, length]

        if x.device != self.w.device:
            self.to(x.device)

        return x * self.w
