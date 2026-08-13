"""Tests for patching functions that are only covered by notebook cells.

Runs on a tiny random native bridge: the assertions are structural
(shape/finiteness/variation), so learned weights are unnecessary.
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.patching import get_act_patch_attn_head_all_pos_every


@pytest.fixture(scope="module")
def model():
    cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=100,
        n_ctx=16,
        act_fn="gelu",
        seed=0,
    )
    bridge = TransformerBridge.boot_native(cfg)
    bridge.eval()
    return bridge


@pytest.fixture(scope="module")
def clean_cache(model):
    torch.manual_seed(0)
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 6))
    _, cache = model.run_with_cache(tokens)
    return cache


@pytest.fixture(scope="module")
def corrupted_tokens(model):
    torch.manual_seed(1)
    return torch.randint(0, model.cfg.d_vocab, (1, 6))


def test_get_act_patch_attn_head_all_pos_every_shape(model, corrupted_tokens, clean_cache):
    """Verify the function returns a [5, n_layers, n_heads] tensor."""

    def metric(logits):
        return logits[:, -1, :].sum()

    result = get_act_patch_attn_head_all_pos_every(model, corrupted_tokens, clean_cache, metric)

    assert result.shape == (5, model.cfg.n_layers, model.cfg.n_heads)


def test_get_act_patch_attn_head_all_pos_every_values_vary(model, corrupted_tokens, clean_cache):
    """Patching different heads should produce different metric values."""

    def metric(logits):
        return logits[:, -1, :].sum()

    result = get_act_patch_attn_head_all_pos_every(model, corrupted_tokens, clean_cache, metric)

    # Not all values should be identical — different heads have different effects
    assert not torch.all(result == result[0, 0, 0]), "All patch results are identical"
    # Values should be finite
    assert torch.isfinite(result).all()
