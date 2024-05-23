"""Hooked Transformer Attention Component.

This module contains all the component :class:`Attention`.
"""
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers.utils import is_bitsandbytes_available

from transformer_lens.components import AbstractAttention
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

if is_bitsandbytes_available():
    from bitsandbytes.nn.modules import Params4bit


# Attention
class Attention(AbstractAttention):
    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        """Attention Block - params have shape [head_index, d_model, d_head] (or [head_index, d_head, d_model] for W_O) and multiply on the right. attn_scores refers to query key dot product immediately before attention softmax

        Convention: All attention pattern-style matrices have shape [batch, head_index, query_pos, key_pos]

        Args:
            cfg (Union[Dict, HookedTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistal models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        super().__init__(cfg, attn_type, layer_id)
        self.cfg = HookedTransformerConfig.unwrap(cfg)

        if self.cfg.load_in_4bit:
            # 4-bit quantization convention
            nq = int((self.cfg.d_model * self.cfg.d_model) / 2)
            self.W_K = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
            self.W_V = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
        else:
            self.W_K = nn.Parameter(
                torch.empty(
                    self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=self.cfg.dtype
                )
            )
            self.W_V = nn.Parameter(
                torch.empty(
                    self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=self.cfg.dtype
                )
            )
        self.b_K = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
        self.b_V = nn.Parameter(
            torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype)
        )
