import torch.nn as nn

from .EasyBERTConfig import EasyBERTConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_model = config.hidden_size
        self.d_k = self.d_model // self.n_heads
        self.d_v = self.d_model // self.n_heads

        self.query = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.key = nn.Linear(self.d_model, self.d_k * self.n_heads)
        self.value = nn.Linear(self.d_model, self.d_v * self.n_heads)

        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, query, key, value, mask=None):
        # TODO
        pass


class EncoderLayer(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        # TODO
        pass


class Encoder(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )

    def forward(self, x):
        return self.layers(x)
