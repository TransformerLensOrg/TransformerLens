import einops
import torch as t
import torch.nn as nn
from torchtyping import TensorType as TT

from .EasyBERTConfig import EasyBERTConfig


class Embeddings(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.d_vocab, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_len, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            2, config.hidden_size
        )  # aka segment embedding
        self.ln = nn.LayerNorm(config.hidden_size)  # TODO use layer norm

    def forward(self, input_ids: TT["batch", "seq"], segment_ids: TT["batch", "seq"]):
        w_e = self.word_embeddings(input_ids)
        # TODO maybe compute positional embeddings once as per this:  https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/embedding/position.py#L6
        base_index_id = t.arange(input_ids.shape[1], device=input_ids.device)
        index_ids = einops.repeat(
            base_index_id, "seq -> batch seq", batch=input_ids.shape[0]
        )
        p_e = self.position_embeddings(index_ids)
        t_e = self.token_type_embeddings(segment_ids)
        # TODO what about dropout?
        return self.ln(w_e + p_e + t_e)
