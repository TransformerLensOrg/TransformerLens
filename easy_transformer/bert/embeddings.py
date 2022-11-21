import torch.nn as nn

from .EasyBERTConfig import EasyBERTConfig


class Embeddings(nn.Module):
    """
    'bert.embeddings.position_ids', 'bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias',
    """

    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.d_vocab, config.hidden_size)

    def load_and_process_state_dict(self, embeddings):
        self.word_embeddings.load_state_dict(
            embeddings["bert.embeddings.word_embeddings"]
        )
        """
        self.position_embeddings.load_state_dict(
            embeddings.position_embeddings.state_dict()
        )
        self.token_type_embeddings.load_state_dict(
            embeddings.token_type_embeddings.state_dict()
        )
        self.LayerNorm.load_state_dict(embeddings.LayerNorm.state_dict())
"""

    def forward(self, input_ids):
        return self.word_embeddings(
            input_ids
        )  # TODO + self.position_embeddings(input_ids) + self.token_type_embeddings(input_ids)
