from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType as TT

from . import attention
from .config import Config


@dataclass
class Output:
    hidden_state: TT["batch", "seq", "hidden"]
    attention_post_softmax: TT["batch", "head", "seq", "seq"]


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.w_1 = nn.Linear(config.hidden_size, config.mlp_size)  # aka 'up' layer
        self.w_2 = nn.Linear(config.mlp_size, config.hidden_size)  # aka 'down' layer
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.hidden_size, eps=1e-12, elementwise_affine=True)

    def forward(self, x: TT["batch", "seq", "hidden"]) -> TT["batch", "seq", "hidden"]:
        original_x = x  # for a residual connection
        x = self.w_1(x)
        x = F.gelu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return self.ln(x + original_x)


class EncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attention = attention.Attention(config)
        self.mlp = MLP(config)

    def forward(self, x: TT["batch", "seq", "hidden"], mask=None) -> Output:
        attention_output: attention.Output = self.attention(x, mask)
        hidden = self.mlp(attention_output.final_output)
        return Output(
            hidden_state=hidden,
            attention_post_softmax=attention_output.attention_post_softmax,
        )
