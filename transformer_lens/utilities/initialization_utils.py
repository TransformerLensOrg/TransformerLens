"""initilization_utils.

This module contains utility functions related to initialization functions
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Literal

# Type alias for valid nonlinearity values accepted by nn.init.calculate_gain
NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]


def calc_fan_in_and_fan_out(tensor):
    """
    Calculate the fan in and fan out of a tensor. We define it ourselves because Torch uses a
    different convention for weights (e.g. for an MLP they use d_out x d_in, and we use d_in x
    d_out, for attention they do (n_head d_head) x d_model, we do n_head x d_model x d_head).
    """
    shape = tensor.shape

    if len(shape) == 0:
        raise ValueError("Fan in and fan out can not be computed for scalars.")
    elif len(shape) == 1:
        fan_in = 1
        fan_out = shape[0]
    elif len(shape) == 2:  # Linear transform
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 3:  # Attention head weight, has shape n_head x d_model x d_head
        fan_in = shape[1]
        fan_out = shape[0] * shape[2]
    else:
        raise ValueError(f"Fan in and fan out can not be computed for shape {shape} tensors.")

    return fan_in, fan_out


def init_xavier_uniform_(param, gain=1.0):
    """
    Initializes the input tensor using the Xavier initialization method.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    max = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return nn.init.uniform_(param, -max, max)


def init_xavier_normal_(param, gain=1.0):
    """
    Initializes the input tensor using the Xavier initialization method.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return nn.init.normal_(param, mean=0.0, std=std)


def init_kaiming_uniform_(
    param: torch.Tensor,
    a: float = 0,
    nonlinearity: NonlinearityType = "relu",
    gain: float = 1.0,
    mode: str = "fan_in",
) -> torch.Tensor:
    """
    Initializes the input tensor using the Kaiming initialization method.

    Starting from a std 1 uniform distribution, we scale the weights by c / sqrt(fan_in), where c =
    sqrt(2) if the params were immediately preceded by a relu and 1 for everything else.

    As with torch, `a` is a hyperparameter for `nonlinearity`, if it takes one.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    max = gain * np.sqrt(3.0 / fan)
    return nn.init.uniform_(param, -max, max)


def init_kaiming_normal_(
    param: torch.Tensor,
    a: float = 0,
    nonlinearity: NonlinearityType = "relu",
    gain: float = 1.0,
    mode: str = "fan_in",
) -> torch.Tensor:
    """
    Initializes the input tensor using the Kaiming initialization method.

    Starting from a std 1 normal distribution, we scale the weights by c / sqrt(fan_in), where c =
    sqrt(2) if the params were immediately preceded by a relu and 1 for everything else.

    As with torch, `a` is a hyperparameter for `nonlinearity`, if it takes one.
    """
    fan_in, fan_out = calc_fan_in_and_fan_out(param)
    fan = fan_in if mode == "fan_in" else fan_out
    gain *= nn.init.calculate_gain(nonlinearity, a)
    std = gain * np.sqrt(1.0 / fan)
    return nn.init.normal_(param, mean=0.0, std=std)
