import torch.nn as nn

from .EasyBERTConfig import EasyBERTConfig


class Embeddings(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.d_vocab, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_len, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.ln = nn.LayerNorm(config.hidden_size)  # TODO use layer norm

    def forward(self, input_ids):
        return (
            self.word_embeddings(input_ids)
            + self.position_embeddings(input_ids)
            + self.token_type_embeddings(input_ids)
        )
