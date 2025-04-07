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
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation."""

    if input.device != w.device:
        w = w.to(input.device)

    # Keep track of original shape for reshaping
    n_heads = w.shape[0]
    d_head = w.shape[2]
    
    # Rearrange weight matrix
    w = einops.rearrange(w, "head_index d_model d_head -> (head_index d_head) d_model")
    
    # Use torch.matmul and reshape to [batch, pos, n_heads, d_head]
    return torch.matmul(input, w.T).reshape(input.shape[0], input.shape[1], n_heads, d_head)


def complex_attn_linear(
    input: Float[torch.Tensor, "batch pos head_index d_model"],
    w: Float[torch.Tensor, "head_index d_model d_head"],
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation.

    This is almost the same as simple_attn_linear, but the input tensor has an extra head_index dimension, used when calculating the input of each attention head separately.
    """

    # Handle case where input has a different head dimension
    if input.shape[2] != w.shape[0]:
        # Average over head dimension
        input = input.mean(dim=2, keepdim=True).expand(-1, -1, w.shape[0], -1)

    # Add singleton dimensions for broadcasting
    input = einops.rearrange(
        input, "batch pos head_index d_model -> batch pos head_index d_model 1"
    )
    w = einops.rearrange(w, "head_index d_model d_head -> 1 1 head_index d_model d_head")

    # Element-wise multiplication and sum over the d_model dimension
    result = input * w
    result = result.sum(dim=-2)
    return result
