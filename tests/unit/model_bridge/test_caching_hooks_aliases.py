"""get_caching_hooks must key the cache under the alias a names_filter matched.

A hook point is reachable under its canonical name and under HookedTransformer-style
aliases. ``run_with_cache`` caches under both; ``get_caching_hooks`` used to key only by
the canonical name, so ``cache[alias]`` raised KeyError even though the hook had fired.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

ALIAS = "blocks.0.hook_resid_pre"
CANONICAL = "blocks.0.hook_in"


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


@pytest.fixture()
def tokens(bridge):
    return torch.randint(0, bridge.cfg.d_vocab, (2, 6))


def _filters():
    return {
        "str": ALIAS,
        "list": [ALIAS],
        "callable": lambda name: name.endswith("hook_resid_pre"),
    }


@pytest.mark.parametrize("kind", ["str", "list", "callable"])
def test_alias_filter_caches_under_alias_and_canonical(bridge, tokens, kind):
    names_filter = _filters()[kind]
    with torch.no_grad():
        _, ref = bridge.run_with_cache(tokens, names_filter=names_filter)

    cache, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=names_filter)
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens)

    assert {CANONICAL, ALIAS} <= set(cache)
    assert torch.equal(cache[ALIAS], ref[ALIAS])
    assert torch.equal(cache[CANONICAL], cache[ALIAS])
    if kind != "callable":
        # run_with_cache maps str/list alias filters to the canonical name too.
        assert set(cache) == set(ref.cache_dict)


def test_canonical_filter_is_unchanged(bridge, tokens):
    cache, fwd_hooks, _ = bridge.get_caching_hooks(names_filter=CANONICAL)
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens)
    assert set(cache) == {CANONICAL}


def test_default_sweep_still_caches_canonical_names(bridge, tokens):
    cache, fwd_hooks, _ = bridge.get_caching_hooks()
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens)
    assert CANONICAL in cache


def test_alias_filter_applies_to_gradients(bridge, tokens):
    cache, fwd_hooks, bwd_hooks = bridge.get_caching_hooks(names_filter=ALIAS, incl_bwd=True)
    with bridge.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        bridge(tokens, return_type="loss").backward()
    assert {CANONICAL, ALIAS, CANONICAL + "_grad", ALIAS + "_grad"} <= set(cache)
    assert torch.equal(cache[ALIAS + "_grad"], cache[CANONICAL + "_grad"])


def test_add_caching_hooks_honours_alias_filter(bridge, tokens):
    cache = bridge.add_caching_hooks(names_filter=ALIAS)
    try:
        with torch.no_grad():
            bridge.forward(tokens)
    finally:
        bridge.reset_hooks()
    assert torch.equal(cache[ALIAS], cache[CANONICAL])
