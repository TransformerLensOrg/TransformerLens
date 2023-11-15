"""Hooked Transformer Embed Component.

This module contains all the components (e.g. :class:`Attention`, :class:`MLP`, :class:`LayerNorm`)
needed to create many different types of generative language models. They are used by
:class:`transformer_lens.HookedTransformer`.
"""
from .layer_norm import LayerNorm
from jaxtyping import Float, Int
import torch
import torch.nn as nn
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from typing import Dict, Union

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=cfg.dtype)
        )
        # Some models (e.g. Bloom) need post embedding layer norm
        if cfg.post_embedding_ln:
            self.ln = LayerNorm(cfg)

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        if self.cfg.post_embedding_ln:
            return self.ln(self.W_E[tokens, :])
        return self.W_E[tokens, :]