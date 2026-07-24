"""HookedRootModule caching-hook API on TransformerBridge.

get_caching_hooks / add_caching_hooks / cache_all / cache_some, plus
run_with_cache(incl_bwd=...). The correctness anchor is that caches built via
these paths match run_with_cache, and that incl_bwd captures nonzero gradients
under "<name>_grad".
"""

import warnings

import pytest
import torch


@pytest.fixture()
def bridge(distilgpt2_bridge):
    return distilgpt2_bridge


def _tokens():
    return torch.randint(0, 100, (1, 6))


def test_get_caching_hooks_matches_run_with_cache(bridge):
    tokens = _tokens()
    with torch.no_grad():
        _, ref = bridge.run_with_cache(tokens)

    cache, fwd_hooks, bwd_hooks = bridge.get_caching_hooks(
        names_filter=lambda n: n.endswith("hook_out")
    )
    assert bwd_hooks == []
    with torch.no_grad(), bridge.hooks(fwd_hooks=fwd_hooks):
        bridge.forward(tokens)

    assert cache, "nothing cached"
    for name in cache:
        assert torch.allclose(cache[name], ref[name], atol=1e-5), name


def test_add_caching_hooks_persists_until_reset(bridge):
    tokens = _tokens()
    with torch.no_grad():
        _, ref = bridge.run_with_cache(tokens)

    cache = bridge.add_caching_hooks(names_filter="blocks.2.hook_out")
    try:
        with torch.no_grad():
            bridge.forward(tokens)
        assert torch.allclose(cache["blocks.2.hook_out"], ref["blocks.2.hook_out"], atol=1e-5)
    finally:
        bridge.reset_hooks()

    cache.clear()
    with torch.no_grad():
        bridge.forward(tokens)
    assert cache == {}, "hooks should be gone after reset_hooks"


def test_cache_all_and_cache_some_are_deprecated(bridge):
    tokens = _tokens()
    for call in (
        lambda c: bridge.cache_all(c),
        lambda c: bridge.cache_some(c, names=lambda n: n.endswith("hook_out")),
    ):
        cache: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(cache)
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        try:
            with torch.no_grad():
                bridge.forward(tokens)
            assert cache, "deprecated helper cached nothing"
        finally:
            bridge.reset_hooks()


def test_run_with_cache_incl_bwd_caches_gradients(bridge):
    tokens = _tokens()
    output, cache = bridge.run_with_cache(tokens, incl_bwd=True, return_type="loss")

    assert output.dim() == 0, "incl_bwd needs a scalar output (return_type='loss')"
    grad_key = "blocks.4.hook_out_grad"
    assert grad_key in cache
    grad = cache[grad_key]
    assert grad.shape == (1, 6, bridge.cfg.d_model)
    assert grad.abs().sum() > 0, "gradient should be nonzero"
    # forward activation is also present and detached
    assert "blocks.4.hook_out" in cache
    assert not cache["blocks.4.hook_out"].requires_grad


def test_incl_bwd_leaves_no_hooks_attached(bridge):
    tokens = _tokens()
    bridge.run_with_cache(tokens, incl_bwd=True, return_type="loss")
    leftover = [hp.name for hp in bridge.hook_points() if hp.has_hooks(dir="both")]
    assert leftover == [], leftover


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
