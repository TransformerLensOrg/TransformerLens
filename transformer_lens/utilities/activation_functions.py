"""Activation Functions.

Utilities for interacting with all supported activation functions.
"""
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from transformer_lens.utils import gelu_fast, gelu_new, solu

# Convenient type for the format of each activation function
ActivationFunction = Callable[..., torch.Tensor]

# All currently supported activation functions. To add a new function, simply
# put the name of the function as the key, and the value as the actual callable.
SUPPORTED_ACTIVATIONS: Dict[str, ActivationFunction] = {
    "solu": solu,
    "solu_ln": solu,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "silu": F.silu,
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda tensor: F.gelu(tensor, approximate="tanh"),
}
