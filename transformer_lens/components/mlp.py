"""Hooked Transformer MLP Component.

This module contains all the component :class:`MLP`.
"""

from typing import Callable, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import gelu_fast, gelu_new, solu


def addmm(bias, weight, x):
    """Fused addmm.

    Must match the Huggingface Conv1D implementation exactly.
    https://github.com/huggingface/transformers/blob/9ba9369a2557e53a01378199a9839ec6e82d8bc7/src/transformers/pytorch_utils.py#L102-L106
    """
    nf = weight.shape[1]
    size_out = x.size()[:-1] + (nf,)
    x = torch.addmm(bias, x.view(-1, x.size(-1)), weight)
    x = x.view(size_out)
    return x


# MLP Layers
class MLP(nn.Module):
    act_fn: Callable[..., torch.Tensor]
    ln: nn.Module

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        assert self.cfg.d_mlp is not None  # TODO: should this not be optional?
        self.W_in = nn.Parameter(
            torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=self.cfg.dtype)
        )
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=self.cfg.dtype))
        self.W_out = nn.Parameter(
            torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype)
        )
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif self.cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        elif self.cfg.act_fn == "silu":
            self.act_fn = F.silu
        elif self.cfg.act_fn == "gelu_new":
            self.act_fn = gelu_new
        elif self.cfg.act_fn == "gelu_fast":
            self.act_fn = gelu_fast
        elif self.cfg.act_fn == "solu_ln":
            self.act_fn = solu
            # Hook taken between activation and layer norm
            self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)

        else:
            raise ValueError(f"Invalid activation function name: {self.cfg.act_fn}")

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # This is equivalent to (roughly) W_in @ x + b_in. It's important to
        # use a fused addmm to ensure it matches the Huggingface implementation
        # exactly.
        pre_act = addmm(self.b_in, self.W_in, x)  # [batch, pos, d_mlp]
        if self.cfg.act_fn is not None and not self.cfg.act_fn.endswith("_ln"):
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        return addmm(self.b_out, self.W_out, post_act)
