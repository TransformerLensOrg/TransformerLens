from dataclasses import dataclass
from typing import Any, Dict

from transformer_lens.TransformerLensConfig import TransformerLensConfig


@dataclass
class HookedEncoderConfig(TransformerLensConfig):
    """
    Configuration class to store the configuration of a HookedEncoder model.
    """
    d_model: int
    d_vocab: int

    model_type: str = "hooked_encoder"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `HookedEncoderConfig` from a Python dictionary of parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__
