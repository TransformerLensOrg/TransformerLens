"""Shared ALiBi (Attention with Linear Biases) utility functions.

Used by Bloom and Falcon ALiBi attention bridges to generate positional bias tensors.
"""

import math

import torch


def build_alibi_slopes(num_heads: int, device: torch.device) -> torch.Tensor:
    """Compute ALiBi per-head slope values.

    For power-of-2 head counts, slopes are geometric: 2^(-8/n), 2^(-16/n), ...
    For non-power-of-2, extra slopes are interleaved from a finer geometric series.
    Matches the HuggingFace implementation.

    Args:
        num_heads: Number of attention heads.
        device: Device for the output tensor.

    Returns:
        Slopes tensor of shape [num_heads].
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
    slopes = torch.pow(torch.tensor(base, device=device, dtype=torch.float32), powers)

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32
        )
        slopes = torch.cat(
            [
                slopes,
                torch.pow(
                    torch.tensor(extra_base, device=device, dtype=torch.float32), extra_powers
                ),
            ],
            dim=0,
        )

    return slopes


def build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor:
    """Build ALiBi positional bias tensor.

    Computes per-head linear biases from token positions, matching HuggingFace's
    ALiBi implementation used in Bloom and Falcon models.

    Args:
        attention_mask: Binary mask of shape [batch_size, seq_length].
        num_heads: Number of attention heads.
        dtype: Output dtype.

    Returns:
        ALiBi tensor of shape [batch_size, num_heads, 1, seq_length].
    """
    batch_size, seq_length = attention_mask.shape
    slopes = build_alibi_slopes(num_heads, attention_mask.device)

    # Position indices: 0-indexed cumulative positions masked by attention_mask
    positions = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    # [batch, 1, seq] * [heads, 1, 1] -> [batch, heads, 1, seq]
    alibi = slopes[None, :, None, None] * positions[:, None, :, :]
    return alibi.to(dtype)
