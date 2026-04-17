"""Hooked Transformer Bert Block Component.

This module contains all the component :class:`BertBlock`.
"""

from typing import Optional

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import Attention, LayerNorm
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.factories.mlp_factory import MLPFactory
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import repeat_along_head_dimension


class PreNormBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.attn = Attention(cfg, attn_type="global")  # bidirectional
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLPFactory.create_mlp(cfg)  # matches ViT FFN if cfg is correct

        # ---- Hooks ----
        self.hook_q_input = HookPoint()
        self.hook_k_input = HookPoint()
        self.hook_v_input = HookPoint()

        self.hook_resid_pre = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_resid_mid = HookPoint()

        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(
        self,
        resid: Float[torch.Tensor, "batch pos d_model"],
    ) -> Float[torch.Tensor, "batch pos d_model"]:

        # ---- Pre-attention ----
        resid = self.hook_resid_pre(resid)

        attn_in = self.ln1(resid)

        # ---- QKV inputs (TL-compatible) ----
        query_input = attn_in
        key_input = attn_in
        value_input = attn_in

        if self.cfg.use_split_qkv_input:
            n_heads = self.cfg.n_heads
            query_input = self.hook_q_input(
                repeat_along_head_dimension(query_input, n_heads)
            )
            key_input = self.hook_k_input(
                repeat_along_head_dimension(key_input, n_heads)
            )
            value_input = self.hook_v_input(
                repeat_along_head_dimension(value_input, n_heads)
            )

        # ---- Attention ----
        attn_out = self.hook_attn_out(
            self.attn(query_input, key_input, value_input)
        )

        resid = self.hook_resid_mid(resid + attn_out)

        # ---- MLP ----
        mlp_in = self.hook_mlp_in(self.ln2(resid))
        mlp_out = self.hook_mlp_out(self.mlp(mlp_in))

        resid = self.hook_resid_post(resid + mlp_out)

        return resid
