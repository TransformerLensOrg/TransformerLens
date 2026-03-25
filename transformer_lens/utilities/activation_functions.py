"""Activation Functions.

Utilities for interacting with all supported activation functions.
"""

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


def gelu_new(
    input: Float[torch.Tensor, "batch pos d_mlp"],
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    )


def gelu_fast(
    input: Float[torch.Tensor, "batch pos d_mlp"],
) -> Float[torch.Tensor, "batch pos d_mlp"]:
    return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


def gelu_pytorch_tanh(input: torch.Tensor) -> torch.Tensor:
    """Approximation of the gelu activation function, used in some older models."""
    return F.gelu(input, approximate="tanh")


def solu(input: Float[torch.Tensor, "batch pos d_mlp"]) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.

    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)


class XIELU(nn.Module):
    """Trainable xIELU activation function.

    See https://arxiv.org/abs/2411.13010

    Matches HuggingFace's XIELUActivation parameterization: alpha_p and alpha_n
    are stored in softplus-inverse space, and beta is a non-trainable buffer.
    """

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta_init: float = 0.5,
        eps: float = -1e-6,
    ):
        super().__init__()
        # Store in softplus-inverse space to match HF's XIELUActivation
        self.alpha_p = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(alpha_p_init, dtype=torch.float32)))
        )
        self.alpha_n = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(alpha_n_init - beta_init, dtype=torch.float32)))
        )
        self.beta: torch.Tensor
        self.eps: torch.Tensor
        self.register_buffer("beta", torch.tensor(beta_init, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def forward(
        self, input: Float[torch.Tensor, "batch pos d_mlp"]
    ) -> Float[torch.Tensor, "batch pos d_mlp"]:
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return torch.where(
            input > 0,
            alpha_p * input * input + self.beta * input,
            (torch.expm1(torch.min(input, self.eps)) - input) * alpha_n + self.beta * input,
        )


def xielu(input: Float[torch.Tensor, "batch pos d_mlp"]) -> Float[torch.Tensor, "batch pos d_mlp"]:
    """Fixed-parameter xIELU activation function as described by
    https://arxiv.org/abs/2411.13010

    Original code: https://github.com/rubber-duck-debug/xielu

    Uses default parameter values. For trainable parameters, use the XIELU class.
    """
    alpha_p: float = 0.8
    alpha_n: float = 0.8
    beta: float = 0.5
    eps = torch.tensor(-1e-6)

    return torch.where(
        input > 0,
        alpha_p * input * input + beta * input,
        (torch.expm1(torch.min(input, eps)) - input) * alpha_n + beta * input,
    )


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
    "gelu_pytorch_tanh": gelu_pytorch_tanh,
    "xielu": xielu,
}
