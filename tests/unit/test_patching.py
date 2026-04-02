"""Tests for patching functions that are only covered by notebook cells."""

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens.patching import get_act_patch_attn_head_all_pos_every


@pytest.fixture(scope="module")
def model():
    return HookedTransformer.from_pretrained("solu-1l", device="cpu")


@pytest.fixture(scope="module")
def clean_cache(model):
    prompt = "The cat sat"
    _, cache = model.run_with_cache(prompt)
    return cache


@pytest.fixture(scope="module")
def corrupted_tokens(model):
    return model.to_tokens("The dog ran")


def test_get_act_patch_attn_head_all_pos_every_shape(model, corrupted_tokens, clean_cache):
    """Verify the function returns a [5, n_layers, n_heads] tensor."""

    def metric(logits):
        return logits[:, -1, :].sum()

    result = get_act_patch_attn_head_all_pos_every(
        model, corrupted_tokens, clean_cache, metric
    )

    assert result.shape == (5, model.cfg.n_layers, model.cfg.n_heads)


def test_get_act_patch_attn_head_all_pos_every_values_vary(model, corrupted_tokens, clean_cache):
    """Patching different heads should produce different metric values."""

    def metric(logits):
        return logits[:, -1, :].sum()

    result = get_act_patch_attn_head_all_pos_every(
        model, corrupted_tokens, clean_cache, metric
    )

    # Not all values should be identical — different heads have different effects
    assert not torch.all(result == result[0, 0, 0]), "All patch results are identical"
    # Values should be finite
    assert torch.isfinite(result).all()
