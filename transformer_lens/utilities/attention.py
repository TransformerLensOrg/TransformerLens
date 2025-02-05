"""Attention.

Utilities for attention components.
"""

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float


def simple_attn_linear(
    input: Float[torch.Tensor, "batch pos d_model"],
    w: Float[torch.Tensor, "head_index d_model d_head"],
    b: Float[torch.Tensor, "head_index d_head"],
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation."""
    w = einops.rearrange(w, "head_index d_model d_head -> (head_index d_head) d_model")
    b_ = einops.rearrange(b, "head_index d_head -> (head_index d_head)")
    return F.linear(input, w, b_).reshape(input.shape[0], input.shape[1], b.shape[0], b.shape[1])


def complex_attn_linear(
    input: Float[torch.Tensor, "batch pos head_index d_model"],
    w: Float[torch.Tensor, "head_index d_model d_head"],
    b: Float[torch.Tensor, "head_index d_head"],
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation.

    This is almost the same as simple_attn_linear, but the input tensor has an extra head_index dimension, used when calculating the input of each attention head separately.
    """

    # Add singleton dimensions for broadcasting
    input = einops.rearrange(
        input, "batch pos head_index d_model -> batch pos head_index d_model 1"
    )
    w = einops.rearrange(w, "head_index d_model d_head -> 1 1 head_index d_model d_head")

    # Element-wise multiplication and sum over the d_model dimension
    result = input * w
    result = result.sum(dim=-2)
    return result + b
