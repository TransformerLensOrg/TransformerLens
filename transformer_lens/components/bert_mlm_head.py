"""Hooked Transformer Components.

This module contains all the components (e.g. :class:`Attention`, :class:`MLP`, :class:`LayerNorm`)
needed to create many different types of generative language models. They are used by
:class:`transformer_lens.HookedTransformer`.
"""
from fancy_einsum import einsum
from jaxtyping import Float
import torch
import torch.nn as nn
from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from typing import Dict, Union


class BertMLMHead(nn.Module):
    """
    Transforms BERT embeddings into logits. The purpose of this module is to predict masked tokens in a sentence.
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W = nn.Parameter(torch.empty(cfg.d_model, cfg.d_model, dtype=cfg.dtype))
        self.b = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))
        self.act_fn = nn.GELU()
        self.ln = LayerNorm(cfg)

    def forward(self, resid: Float[torch.Tensor, "batch pos d_model"]) -> torch.Tensor:
        resid = (
            einsum(
                "batch pos d_model_in, d_model_out d_model_in -> batch pos d_model_out",
                resid,
                self.W,
            )
            + self.b
        )
        resid = self.act_fn(resid)
        resid = self.ln(resid)
        return resid
