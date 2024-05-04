"""Hooked Transformer Embed Component.

This module contains all the component :class:`BertMLMHead`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W_E: Float[torch.Tensor, "d_vocab d_model"] = nn.Parameter(
            torch.empty(self.cfg.d_vocab, self.cfg.d_model, dtype=self.cfg.dtype)
        )
        # Some models (e.g. Bloom) need post embedding layer norm
        if self.cfg.post_embedding_ln:
            self.ln = LayerNorm(self.cfg)

    def forward(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        if self.cfg.post_embedding_ln:
            return self.ln(self.W_E[tokens, :])
        return self.W_E[tokens, :]
