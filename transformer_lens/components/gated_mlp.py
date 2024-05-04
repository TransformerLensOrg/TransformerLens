"""Hooked Transformer Gated MLP Component.

This module contains all the component :class:`GatedMLP`.
"""
from typing import Callable, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from jaxtyping import Float
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components import LayerNorm, LayerNormPre
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utils import gelu_fast, gelu_new, solu

if is_bitsandbytes_available():
    import bitsandbytes as bnb
    from bitsandbytes.nn.modules import Params4bit


# TODO
# not sure whether to fold this into MLP or not
class GatedMLP(nn.Module):
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

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        assert self.cfg.d_mlp is not None  # keep mypy happy

        if self.cfg.load_in_4bit:
            nq = int((self.cfg.d_model * self.cfg.d_mlp) / 2)
            self.W_in = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
            self.W_gate = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
            self.W_out = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
        else:
            self.W_in = nn.Parameter(
                torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=self.cfg.dtype)
            )
            self.W_gate = nn.Parameter(
                torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=self.cfg.dtype)
            )
            self.W_out = nn.Parameter(
                torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype)
            )

        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=self.cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        # hook on gate output but before act_fn
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
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
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        if self.cfg.load_in_4bit:
            pre_act = self.hook_pre(
                bnb.matmul_4bit(x, self.W_gate.t(), bias=None, quant_state=self.W_gate.quant_state)
            )
        else:
            pre_act = self.hook_pre(
                einsum(
                    "batch pos d_model, d_model d_mlp -> batch pos d_mlp",
                    x,
                    self.W_gate,
                )
            )  # [batch, pos, d_mlp]

        if self.cfg.act_fn is not None and not self.cfg.act_fn.endswith("_ln"):
            if self.cfg.load_in_4bit:
                pre_linear = self.hook_pre_linear(
                    bnb.matmul_4bit(x, self.W_in.t(), bias=None, quant_state=self.W_in.quant_state)
                )
            else:
                pre_linear = self.hook_pre_linear(
                    einsum(
                        "batch pos d_model, d_model d_mlp -> batch pos d_mlp",
                        x,
                        self.W_in,
                    )
                )

            post_act = self.hook_post(
                (self.act_fn(pre_act) * pre_linear) + self.b_in
            )  # [batch, pos, d_mlp]
        else:
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))

        if self.cfg.load_in_4bit:
            return bnb.matmul_4bit(
                post_act, self.W_out.t(), bias=None, quant_state=self.W_out.quant_state
            )
        else:
            return (
                einsum(
                    "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
                    post_act,
                    self.W_out,
                )
                + self.b_out
            )
