"""Hooked Transformer Bert MLM Head Component.

This module contains all the component :class:`BertMLMHead`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.util import addmm


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
        resid = addmm(self.b, self.W, resid)
        resid = self.act_fn(resid)
        resid = self.ln(resid)
        return resid
