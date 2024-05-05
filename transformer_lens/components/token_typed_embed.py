"""Hooked Transformer Token Typed Embed Component.

This module contains all the component :class:`TokenTypeEmbed`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Int

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class TokenTypeEmbed(nn.Module):
    """
    The token-type embed is a binary ids indicating whether a token belongs to sequence A or B. For example, for two sentences: "[CLS] Sentence A [SEP] Sentence B [SEP]", token_type_ids would be [0, 0, ..., 0, 1, ..., 1, 1]. `0` represents tokens from Sentence A, `1` from Sentence B. If not provided, BERT assumes a single sequence input. Typically, shape is (batch_size, sequence_length).

    See the BERT paper for more information: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.W_token_type = nn.Parameter(torch.empty(2, self.cfg.d_model, dtype=self.cfg.dtype))

    def forward(self, token_type_ids: Int[torch.Tensor, "batch pos"]):
        return self.W_token_type[token_type_ids, :]
