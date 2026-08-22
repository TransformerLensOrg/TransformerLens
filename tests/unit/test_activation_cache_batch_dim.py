"""Batch-dim handling for caches holding broadcast / position-indexed entries.

T5-style caches mix genuinely batched activations with entries whose leading dim
is not the batch: broadcast entries (leading 1) and position-indexed entries
(leading dim = seq len, e.g. relative position bias). remove_batch_dim and
apply_slice_to_batch_dim must not corrupt those.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens.ActivationCache import ActivationCache


def _mixed_batch1_cache() -> ActivationCache:
    return ActivationCache(
        {
            "blocks.0.hook_resid_pre": torch.randn(1, 5, 4),
            "blocks.0.hook_pattern": torch.randn(1, 2, 5, 5),
            # Position-indexed entry: leading dim is seq len, not batch.
            "blocks.0.attn.hook_rel_pos_bias": torch.randn(5, 5),
        },
        model=None,
        has_batch_dim=True,
    )


def test_remove_batch_dim_leaves_position_indexed_entries_alone() -> None:
    cache = _mixed_batch1_cache()
    bias_before = cache["blocks.0.attn.hook_rel_pos_bias"].clone()

    cache.remove_batch_dim()

    assert cache["blocks.0.hook_resid_pre"].shape == (5, 4)
    assert cache["blocks.0.hook_pattern"].shape == (2, 5, 5)
    assert torch.equal(cache["blocks.0.attn.hook_rel_pos_bias"], bias_before)


def test_remove_batch_dim_refuses_true_batch_gt_1_despite_broadcast_entry() -> None:
    cache = ActivationCache(
        {
            "blocks.0.hook_resid_pre": torch.randn(2, 5, 4),
            "blocks.0.hook_mlp_out": torch.randn(2, 5, 4),
            "hook_pos_indices": torch.randn(1, 5),
        },
        model=None,
        has_batch_dim=True,
    )
    with pytest.raises(AssertionError, match="batch size 2"):
        cache.remove_batch_dim()


def test_apply_slice_to_batch_dim_skips_broadcast_entries() -> None:
    cache = ActivationCache(
        {
            "blocks.0.hook_resid_pre": torch.randn(3, 5, 4),
            "blocks.0.hook_mlp_out": torch.randn(3, 5, 4),
            "hook_pos_indices": torch.randn(1, 5),
        },
        model=None,
        has_batch_dim=True,
    )
    sliced = cache.apply_slice_to_batch_dim((1, 3))

    assert sliced["blocks.0.hook_resid_pre"].shape == (2, 5, 4)
    assert sliced["blocks.0.hook_mlp_out"].shape == (2, 5, 4)
    assert sliced["hook_pos_indices"].shape == (1, 5)
