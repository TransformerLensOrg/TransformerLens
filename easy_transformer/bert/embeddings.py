import einops
import torch as t
import torch.nn as nn
from torchtyping import TensorType as TT

from ..components import LayerNorm
from .config import Config


class Embeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_length, config.d_model)
        self.token_type_embeddings = nn.Embedding(
            2, config.d_model
        )  # aka segment embedding
        self.ln = LayerNorm(cfg=config)  # type: ignore
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: TT["batch", "seq"], segment_ids: TT["batch", "seq"]):
        w_e = self.word_embeddings(input_ids)
        # alternatively we could compute positional embeddings once as per this:  https://github.com/codertimo/BERT-pytorch/blob/d10dc4f9d5a6f2ca74380f62039526eb7277c671/bert_pytorch/model/embedding/position.py#L6
        base_index_id = t.arange(input_ids.shape[1], device=input_ids.device)
        index_ids = einops.repeat(
            base_index_id, "seq -> batch seq", batch=input_ids.shape[0]
        )
        p_e = self.position_embeddings(index_ids)
        t_e = self.token_type_embeddings(segment_ids)
        # useful reference: https://github.com/maknotavailable/pytorch-pretrained-BERT/blob/8d5d1aa631480e395cdeed85ebb6cc19e89e84ab/pytorch_pretrained_bert/modeling.py#L198
        return self.dropout(self.ln(w_e + p_e + t_e))
