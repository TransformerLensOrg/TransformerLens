"""`get_caching_hooks(remove_batch_dim=True, pos_slice=...)` slices the position axis.

`_pos_slice_dim` names the axis in the *batched* layout (dim 1 for resid / per-head
tensors, -2 for attention maps). `get_caching_hooks.save_hook` used to drop the batch
dim first and slice second, so with both flags set dim 1 was d_model (resid) or n_heads
(per-head) instead of the position, and a 2-D `[batch, pos]` activation (token ids at
`embed.hook_in`) was not sliced at all because the `dim() >= 2` guard ran on the 1-D
remainder. `run_with_cache` slices first, then removes the batch dim; the two caching
APIs must agree.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

# Canonical names: get_caching_hooks keys the cache by HookPoint name. Mixes token ids
# [b, p], a resid stream [b, p, d], a head-split tensor [b, p, h, d_head] and an
# attention map [b, h, dest, src], so a slice on a same-length neighbouring axis, or a
# skipped slice on the 2-D case, is caught.
NAMES = [
    "embed.hook_in",
    "blocks.0.hook_out",
    "blocks.0.attn.o.hook_in",
    "blocks.0.attn.hook_pattern",
]


def _bridge() -> TransformerBridge:
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


def _run_caching_hooks(bridge, tokens, **kwargs) -> dict:
    cache, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=NAMES, **kwargs)
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens)
    return cache


@pytest.mark.parametrize("pos_slice", [-1, 2, (1, 4), [0, 2, 5]])
def test_remove_batch_dim_with_pos_slice_matches_run_with_cache(pos_slice):
    """Same args, same tensors: values (not just shapes) match run_with_cache."""
    bridge = _bridge()
    torch.manual_seed(0)
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 6))

    with torch.no_grad():
        _, ref = bridge.run_with_cache(
            tokens, names_filter=NAMES, remove_batch_dim=True, pos_slice=pos_slice
        )
    cache = _run_caching_hooks(bridge, tokens, remove_batch_dim=True, pos_slice=pos_slice)

    for name in NAMES:
        assert cache[name].shape == ref[name].shape, f"{name} shape differs from run_with_cache"
        torch.testing.assert_close(cache[name], ref[name])


def test_remove_batch_dim_with_int_pos_slice_shapes():
    """Pin the rank and axis order left behind: position trimmed to 1, batch gone."""
    bridge = _bridge()
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 6))
    n_heads, d_head, d_model = bridge.cfg.n_heads, bridge.cfg.d_head, bridge.cfg.d_model

    cache = _run_caching_hooks(bridge, tokens, remove_batch_dim=True, pos_slice=-1)

    assert cache["embed.hook_in"].shape == (1,)
    assert cache["blocks.0.hook_out"].shape == (1, d_model)
    assert cache["blocks.0.attn.o.hook_in"].shape == (1, n_heads, d_head)
    assert cache["blocks.0.attn.hook_pattern"].shape == (n_heads, 1, 6)


def test_pos_slice_without_remove_batch_dim_is_unchanged():
    """The already-correct batched path keeps its behaviour."""
    bridge = _bridge()
    tokens = torch.randint(0, bridge.cfg.d_vocab, (1, 6))

    with torch.no_grad():
        _, ref = bridge.run_with_cache(tokens, names_filter=NAMES, pos_slice=(1, 4))
    cache = _run_caching_hooks(bridge, tokens, pos_slice=(1, 4))

    for name in NAMES:
        assert cache[name].shape == ref[name].shape, name
        torch.testing.assert_close(cache[name], ref[name])
