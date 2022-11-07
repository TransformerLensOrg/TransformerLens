from dataclasses import dataclass


@dataclass
class EasyBERTConfig:
    """TODO"""

    # TODO check all these
    n_layers: int
    n_heads: int
    hidden_size: int  # TODO aka d_model?
    dropout: float = 0.1

    # TODO add _post_init
