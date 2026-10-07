"""run_with_hooks(remove_batch_dim=True) hands each hook a tensor without the batch dimension."""

from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

HOOK = "blocks.0.hook_out"


@pytest.fixture(scope="module")
def bridge():
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_heads=2,
        n_layers=1,
        n_ctx=8,
        d_vocab=16,
        d_mlp=64,
        act_fn="gelu",
        normalization_type="LN",
        seed=0,
    )
    return TransformerBridge.boot_native(cfg)


def _tokens(bridge, batch: int) -> torch.Tensor:
    return torch.randint(0, bridge.cfg.d_vocab, (batch, 6))


@torch.no_grad()
def test_hook_that_only_observes_runs_at_batch_size_one(bridge):
    tokens = _tokens(bridge, 1)
    seen = []

    def observe(tensor, hook):
        seen.append(tuple(tensor.shape))

    hooked = bridge.run_with_hooks(tokens, fwd_hooks=[(HOOK, observe)], remove_batch_dim=True)

    assert seen == [(6, bridge.cfg.d_model)]
    torch.testing.assert_close(hooked, bridge(tokens))


@torch.no_grad()
def test_hook_that_edits_the_activation_in_place_and_returns_none_reaches_the_model(bridge):
    tokens = _tokens(bridge, 1)

    def zero_first_position(tensor, hook):
        tensor[..., 0, :] = 0

    expected = bridge.run_with_hooks(tokens, fwd_hooks=[(HOOK, zero_first_position)])
    hooked = bridge.run_with_hooks(
        tokens, fwd_hooks=[(HOOK, zero_first_position)], remove_batch_dim=True
    )

    torch.testing.assert_close(hooked, expected)
    assert not torch.allclose(hooked, bridge(tokens))


@torch.no_grad()
def test_hook_that_replaces_the_activation_gets_its_batch_dimension_back(bridge):
    tokens = _tokens(bridge, 1)

    def double(tensor, hook):
        return tensor * 2

    expected = bridge.run_with_hooks(tokens, fwd_hooks=[(HOOK, double)])
    hooked = bridge.run_with_hooks(tokens, fwd_hooks=[(HOOK, double)], remove_batch_dim=True)

    torch.testing.assert_close(hooked, expected)


@torch.no_grad()
def test_a_larger_batch_reaches_the_hook_unchanged(bridge):
    tokens = _tokens(bridge, 2)
    seen = []

    def observe(tensor, hook):
        seen.append(tuple(tensor.shape))

    bridge.run_with_hooks(tokens, fwd_hooks=[(HOOK, observe)], remove_batch_dim=True)

    assert seen == [(2, 6, bridge.cfg.d_model)]
