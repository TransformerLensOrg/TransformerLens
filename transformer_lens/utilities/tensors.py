"""tensors.

This module contains utility functions related to raw tensors
"""

from __future__ import annotations

from typing import Tuple, cast

import einops
import numpy as np
import torch
from jaxtyping import Float, Int


def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (torch.Tensor, torch.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")


def get_corner(tensor, n=3):
    # Prints the top left corner of the tensor
    return tensor[tuple(slice(n) for _ in range(tensor.ndim))]


def remove_batch_dim(tensor: Float[torch.Tensor, "1 ..."]) -> Float[torch.Tensor, "..."]:
    """
    Removes the first dimension of a tensor if it is size 1, otherwise returns the tensor unchanged
    """
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)
    else:
        return tensor


def transpose(tensor: Float[torch.Tensor, "... a b"]) -> Float[torch.Tensor, "... b a"]:
    """
    Utility to swap the last two dimensions of a tensor, regardless of the number of leading dimensions
    """
    return tensor.transpose(-1, -2)


def is_square(x: torch.Tensor) -> bool:
    """Checks if `x` is a square matrix."""
    return x.ndim == 2 and x.shape[0] == x.shape[1]


def is_lower_triangular(x: torch.Tensor) -> bool:
    """Checks if `x` is a lower triangular matrix."""
    if not is_square(x):
        return False
    return x.equal(x.tril())


def check_structure(t1: torch.Tensor, t2: torch.Tensor, *, verbose: bool = False) -> None:
    """Validate that the two square tensors have the same structure, i.e.,
    that the directionality of comparisons points in the same directions both
    row-wise and column-wise.

    This function is not used anywhere in the code right now, just for debugging tests.
    """
    assert t1.ndim == 2
    assert t1.shape == t2.shape
    n_rows, n_cols = cast(Tuple[int, int], t1.shape)

    if verbose:
        print("Checking rows")
    row_mismatch = []
    for row_i in range(n_rows - 1):
        t1_result = t1[row_i].ge(t1[row_i + 1])
        t2_result = t2[row_i].ge(t2[row_i + 1])
        if any(t1_result != t2_result):
            row_mismatch.append(row_i)
            if verbose:
                print(f"\trows {row_i}:{row_i + 1}")
                print(f"\tt1: {t1_result.tolist()}")
                print(f"\tt2: {t2_result.tolist()}")

    if verbose:
        print("Checking columns")
    col_mismatch = []
    for col_i in range(n_cols - 1):
        t1_result = t1[:, col_i].ge(t1[:, col_i + 1])
        t2_result = t2[:, col_i].ge(t2[:, col_i + 1])
        if any(t1_result != t2_result):
            col_mismatch.append(col_i)
            if verbose:
                print(f"\trows {col_i}:{col_i + 1}")
                print(f"\tt1: {t1_result.tolist()}")
                print(f"\tt2: {t2_result.tolist()}")
    if not row_mismatch and not col_mismatch:
        print("PASSED")
    elif row_mismatch:
        print(f"row mismatch: {row_mismatch}")
    elif col_mismatch:
        print(f"column mismatch: {col_mismatch}")


def get_offset_position_ids(
    past_kv_pos_offset: int,
    attention_mask: Int[torch.Tensor, "batch offset_pos"],
) -> Int[torch.Tensor, "batch pos"]:
    """
    Returns the indices of non-padded tokens, offset by the position of the first attended token.
    """
    # shift the position ids so that the id at the the first attended token position becomes zero.
    # The position ids of the prepending pad tokens are shifted to -1.
    shifted_position_ids = attention_mask.cumsum(dim=1) - 1  # [batch, tokens_length]

    # Set the position ids of all prepending pad tokens to an arbitrary number (zero here)
    # just to avoid indexing errors.
    position_ids = shifted_position_ids.masked_fill(shifted_position_ids < 0, 0)
    return position_ids[:, past_kv_pos_offset:]  # [pos, batch]


def get_cumsum_along_dim(tensor, dim, reverse=False):
    """
    Returns the cumulative sum of a tensor along a given dimension.
    """
    if reverse:
        tensor = tensor.flip(dims=(dim,))
    cumsum = tensor.cumsum(dim=dim)
    if reverse:
        cumsum = cumsum.flip(dims=(dim,))
    return cumsum


def repeat_along_head_dimension(
    tensor: Float[torch.Tensor, "batch pos d_model"],
    n_heads: int,
    clone_tensor=True,
    # `einops.repeat` uses a view in torch, so we generally clone the tensor to avoid using shared storage for each head entry
):
    repeated_tensor = einops.repeat(
        tensor,
        "batch pos d_model -> batch pos n_heads d_model",
        n_heads=n_heads,
    )
    if clone_tensor:
        return repeated_tensor.clone()
    else:
        return repeated_tensor


def filter_dict_by_prefix(dictionary: dict, prefix: str) -> dict:
    """Filter a dictionary to only include keys that start with the given prefix and strip the prefix.

    Args:
        dictionary: Dictionary to filter
        prefix: Key prefix to match (will be stripped from returned keys)

    Returns:
        Dictionary containing only entries where keys start with the prefix, with the prefix removed from keys.
        If the prefix ends with a dot, the dot is included in what gets stripped. If not, a dot separator
        is automatically added/expected.

    Example:
        >>> import torch
        >>> d = {"transformer.h.0.attn.W_Q": torch.tensor([1]), "transformer.h.0.mlp.W_in": torch.tensor([2]), "transformer.h.1.attn.W_K": torch.tensor([3])}
        >>> result = filter_dict_by_prefix(d, "transformer.h.0")
        >>> sorted(result.keys())
        ['attn.W_Q', 'mlp.W_in']
        >>> result["attn.W_Q"]
        tensor([1])
        >>> result["mlp.W_in"]
        tensor([2])
    """
    # Ensure prefix ends with a dot for proper stripping
    search_prefix = prefix if prefix.endswith(".") else prefix + "."

    return {
        k[len(search_prefix) :]: v for k, v in dictionary.items() if k.startswith(search_prefix)
    }
