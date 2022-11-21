from dataclasses import dataclass


@dataclass
class EasyBERTConfig:
    """TODO"""

    model_name: str  # TODO use types to encode the officialness of this model name

    # TODO check all these
    n_layers: int
    n_heads: int
    hidden_size: int  # TODO aka d_model?
    d_vocab: int

    dropout: float = 0.1

    # TODO add _post_init
