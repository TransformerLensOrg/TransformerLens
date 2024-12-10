"""Hooked Transformer Bert MLM Head Component.

This module contains all the component :class:`BertMLMHead`.
"""
from typing import Dict, Union

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class BertMLMHead(nn.Module):
    """
    Transforms BERT embeddings into logits. The purpose of this module is to predict masked tokens in a sentence.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_model, dtype=self.cfg.dtype))
        self.b = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))
        self.act_fn = nn.GELU()
        self.ln = LayerNorm(self.cfg)

    def forward(self, resid: Float[torch.Tensor, "batch pos d_model"]) -> torch.Tensor:
        # Add singleton dimension for broadcasting
        resid = einops.rearrange(resid, "batch pos d_model_in -> batch pos 1 d_model_in")

        # Element-wise multiplication of W and resid
        resid = resid * self.W

        # Sum over d_model_in dimension and add bias
        resid = resid.sum(-1) + self.b

        resid = self.act_fn(resid)
        resid = self.ln(resid)
        return resid
