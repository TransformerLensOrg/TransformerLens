"""Hooked Transformer Gated MLP Component.

This module contains all the component :class:`GatedMLP`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class GatedMLPUnbiased(CanBeUsedAsMLP):
    """
    The equation of a gated MLP:
    pre = x @ W_gate
    pre_linear = x @ W_in
    post = Gelu(pre) * (pre_linear) + b_in
    mlp_out = post @ W_out + b_out

    In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__(cfg)
        self.select_activation_function()
        self.W_in = nn.Linear(self.cfg.d_model, self.d_mlp, bias=False)
        self.W_out = nn.Linear(self.d_mlp, self.cfg.d_model, bias=False)
        self.W_gate = nn.Linear(self.cfg.d_model, self.d_mlp, bias=False)

        # hook on gate output but before act_fn
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        gated_x = self.hook_pre(
            self.W_gate(x)  # batch pos d_model, d_model d_mlp -> batch pos d_mlp
        )  # [batch, pos, d_mlp]

        pre_act = self.hook_pre_linear(
            self.W_in(x)  # batch pos d_model, d_model d_mlp -> batch pos d_mlp
        )

        post_act = self.hook_post((self.act_fn(pre_act) * gated_x))  # [batch, pos, d_mlp]

        return self.W_out(post_act)
