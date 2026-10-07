"""Torch-free validation of the single-sequence Inspect padding-mask wire format."""

from typing import Any

import numpy as np


def normalize_attention_mask(mask: Any, n_tokens: int) -> list[int]:
    """Accept a binary flat wire mask or a single-row public padding mask."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().tolist()
    array = np.asarray(mask)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.shape[0] != n_tokens:
        raise ValueError("Inspect attention_mask must match the single input sequence length.")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("Inspect attention_mask must be a binary 0/1 padding mask.")
    if not np.any(array):
        raise ValueError("Inspect attention_mask must retain at least one token.")
    return [int(value) for value in array.tolist()]
