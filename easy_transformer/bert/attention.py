from dataclasses import dataclass

import einops
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum
from torchtyping import TensorType as TT

from .EasyBERTConfig import EasyBERTConfig


@dataclass
class Output:
    final_output: TT["batch", "seq", "hidden"]
    attention_post_softmax: TT["batch", "head", "seq", "seq"]


class SelfAttention(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        # TODO someday make head_size distinct so that this module can be parallel
        self.w_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_o = nn.Linear(config.hidden_size, config.hidden_size)

    def attention_pattern(
        self, x: TT["batch", "seq", "hidden"]
    ) -> TT["batch", "head", "seq", "seq"]:
        q = self.w_q(x)
        k = self.w_k(x)
        q = einops.rearrange(
            q,
            "batch seq (head head_size) -> batch head seq head_size",
            head=self.config.n_heads,
        )
        k = einops.rearrange(
            k,
            "batch seq (head head_size) -> batch head seq head_size",
            head=self.config.n_heads,
        )
        result = einsum(
            "batch head seq_q head_size, batch head seq_k head_size -> batch head seq_q seq_k",
            q,
            k,
        )
        head_size = self.config.hidden_size // self.config.n_heads
        return result / (head_size**0.5)

    @dataclass
    class Output:
        self_attention_output: TT["batch", "seq", "intermediate"]
        attention_post_softmax: TT["batch", "head", "seq", "seq"]

    def forward(self, x: TT["batch", "seq", "hidden"], mask=None) -> Output:
        # here we do the computation per head
        attention = (
            self.attention_pattern(x)
            if mask is None
            else self.attention_pattern(x) + mask
        )
        attention = attention.softmax(dim=-1)
        v = self.w_v(x)
        v = einops.rearrange(
            v,
            "b seq (head head_size) -> b head seq head_size",
            head=self.config.n_heads,
        )
        combined_values = einsum(
            "b head seq_k head_size, b head seq_q seq_k -> b head seq_q head_size",
            v,
            attention,
        )
        # now collapse back to the original shape
        rearranged = einops.rearrange(
            combined_values, "b head seq head_size -> b seq (head head_size)"
        )
        return self.Output(
            self_attention_output=self.w_o(rearranged),
            attention_post_softmax=attention,
        )


class Attention(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.self_attention = SelfAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.hidden_size, eps=1e-12, elementwise_affine=True)

    def forward(self, x: TT["batch", "seq", "hidden"], mask=None) -> Output:
        original_x = x  # for a residual connection
        sao: SelfAttention.Output = self.self_attention(x, mask=mask)
        x = self.dropout(sao.self_attention_output)
        return Output(
            final_output=self.ln(x + original_x),
            attention_post_softmax=sao.attention_post_softmax,
        )
