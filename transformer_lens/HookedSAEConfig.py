from __future__ import annotations

import pprint
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from transformer_lens import utils


@dataclass
class HookedSAEConfig:
    """
    Configuration class to store the configuration of a HookedSAE model.

    Args:
        d_sae (int): The size of the dictionary.
        d_in (int): The dimension of the input activations.
        hook_name (str): The hook name of the activation the SAE was trained on (eg. blocks.0.attn.hook_z)
        use_error_term (bool): Whether to use the error term in the loss function. Defaults to False.
        dtype (torch.dtype, *optional*): The SAE's dtype. Defaults to torch.float32.
        seed (int, *optional*): The seed to use for the SAE.
            Used to set sources of randomness (Python, PyTorch and
            NumPy) and to initialize weights. Defaults to None. We recommend setting a seed, so your experiments are reproducible.
        device(str): The device to use for the SAE. Defaults to 'cuda' if
            available, else 'cpu'.
    """

    d_sae: int
    d_in: int
    hook_name: str
    use_error_term: bool = False
    dtype: torch.dtype = torch.float32
    seed: Optional[int] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.seed is not None:
            self.set_seed_everywhere(self.seed)

        if self.device is None:
            self.device = utils.get_device()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedSAEConfig:
        """
        Instantiates a `HookedSAEConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedSAEConfig:\n" + pprint.pformat(self.to_dict())

    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
