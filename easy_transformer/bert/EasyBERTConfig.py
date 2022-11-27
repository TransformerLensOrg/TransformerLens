from dataclasses import dataclass
from typing import Optional

import torch as t


@dataclass
class EasyBERTConfig:
    # TODO make all of these naming schemes the same

    model_name: str  # TODO use types to encode the officialness of this model name

    # TODO check all these
    n_layers: int
    n_heads: int
    hidden_size: int  # TODO aka d_model?
    d_vocab: int

    max_len: int

    dropout: float = 0.1

    # TODO unify [t] vs [torch]
    device: Optional[t.device] = None

    # TODO again, [str]?
    tokenizer_name: Optional[str] = None

    # TODO add _post_init
