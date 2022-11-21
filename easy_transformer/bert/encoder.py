import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .EasyBERTConfig import EasyBERTConfig


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        assert config.hidden_size % config.n_heads == 0

        # We assume d_v always equals d_k
        self.d_k = config.hidden_size // config.n_heads
        self.h = config.n_heads

        self.linear_layers = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.hidden_size) for _ in range(3)]
        )
        self.output_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [
            l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class SublayerConnection(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, sublayer):
        # TODO this is a bit duplicated
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.input_sublayer = SublayerConnection(config)
        self.output_sublayer = SublayerConnection(config)


class Encoder(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, x, mask=None):
        return self.layers(x)
