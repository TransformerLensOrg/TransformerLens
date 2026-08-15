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

    if input.device != w.device:
        w = w.to(input.device)
    if input.device != b.device:
        b = b.to(input.device)

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

    result = einops.einsum(
        input,
        w,
        "batch pos head_index d_model, head_index d_model d_head -> batch pos head_index d_head",
    )
    return result + b


def clamp_qkv(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, clip: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp Q/K/V to [-clip, clip] out-of-place (OLMo/OLMoE/MPT clip_qkv).

    Out-of-place rather than HF's clamp_ so tensors wrapped by full backward
    hooks stay legal to use.
    """
    return (
        q.clamp(min=-clip, max=clip),
        k.clamp(min=-clip, max=clip),
        v.clamp(min=-clip, max=clip),
    )
