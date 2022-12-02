from dataclasses import dataclass
from typing import Literal, Optional

import torch as t

ModelName = Literal["bert-base-uncased"]
TokenizerName = Literal["bert-base-uncased"]


@dataclass
class Config:
    model: ModelName

    layers: int
    heads: int
    hidden_size: int
    head_size: int  # TODO add documentation
    vocab_size: int
    mlp_size: int

    max_length: int

    dropout: float = 0.1

    device: Optional[t.device] = None

    tokenizer: Optional[TokenizerName] = None

    # TODO add _post_init
