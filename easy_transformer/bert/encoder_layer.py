import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType as TT

from . import attention
from .config import Config


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.w_1 = nn.Linear(config.d_model, config.mlp_size)  # aka 'up' layer
        self.w_2 = nn.Linear(config.mlp_size, config.d_model)  # aka 'down' layer
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.d_model, eps=1e-12, elementwise_affine=True)

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

    def forward(self, x: TT["batch", "seq", "hidden"], mask=None):
        attention_output: attention.Output = self.attention(x, mask)
        return self.mlp(attention_output.final_output)
