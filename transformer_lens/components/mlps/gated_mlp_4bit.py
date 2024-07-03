"""Hooked Transformer Gated MLP Component.

This module contains all the component :class:`GatedMLP`.
"""
from typing import Callable, Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit


class GatedMLP4Bit(CanBeUsedAsMLP):
    """
    The equation of a gated MLP:
    pre = x @ W_gate
    pre_linear = x @ W_in
    post = Gelu(pre) * (pre_linear) + b_in
    mlp_out = post @ W_out + b_out

    In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
    """

    act_fn: Callable[..., torch.Tensor]
    ln: nn.Module

    def __init__(self, config: Union[Dict, HookedTransformerConfig]):
        super().__init__(config)

        nq = int((self.cfg.d_model * self.cfg.d_mlp) / 2)
        self.W_in = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
        self.W_gate = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
        self.W_out = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)

        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]


    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(
            bnb.matmul_4bit(x, self.W_gate.t(), bias=None, quant_state=self.W_gate.quant_state)
        )

        if self.is_layer_norm_activation():
            pre_linear = self.hook_pre_linear(
                bnb.matmul_4bit(x, self.W_in.t(), bias=None, quant_state=self.W_in.quant_state)
            )

            post_act = self.hook_post(
                (self.act_fn(pre_act) * pre_linear) + self.b_in
            )  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))

        return bnb.matmul_4bit(
            post_act, self.W_out.t(), bias=None, quant_state=self.W_out.quant_state
        )
