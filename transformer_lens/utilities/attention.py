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
    return (
        einops.einsum(
            input,
            w,
            "batch pos head_index d_model, head_index d_model d_head -> batch pos head_index d_head",
        )
        + b
    )
