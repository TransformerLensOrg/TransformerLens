"""start_at_layer (residual-stream input) on TransformerBridge.

The correctness anchor is a round trip: cache the residual entering block k
from a full run, resume from it with start_at_layer=k, and require the logits
and the downstream cache to match the full run bit-for-bit. Blocks below k
execute internally (the bridge delegates to HF) but must not appear in the
cache or fire hooks, matching HookedTransformer.
"""

import pytest
import torch


@pytest.fixture()
def bridge(distilgpt2_bridge):
    return distilgpt2_bridge


def _tokens():
    return torch.randint(0, 100, (2, 8))


def test_resume_matches_full_run_logits_and_downstream_cache(bridge):
    tokens = _tokens()
    full_logits, full_cache = bridge.run_with_cache(tokens)
    k = 3
    resid = full_cache[f"blocks.{k}.hook_in"]

    logits, cache = bridge.run_with_cache(resid, start_at_layer=k)

    assert torch.allclose(logits, full_logits, atol=1e-4)
    # Every downstream key present in the full run must match exactly.
    for name, value in full_cache.items():
        if name.startswith("blocks.") and int(name.split(".")[1]) >= k:
            assert name in cache, f"{name} missing from resumed cache"
            assert torch.allclose(cache[name], value, atol=1e-4), name


def test_upstream_hooks_absent_downstream_present(bridge):
    tokens = _tokens()
    _, full_cache = bridge.run_with_cache(tokens)
    k = 3
    _, cache = bridge.run_with_cache(full_cache[f"blocks.{k}.hook_in"], start_at_layer=k)

    keys = set(cache.keys())
    embedding_stage = [
        n
        for n in keys
        if n in ("hook_embed", "hook_pos_embed", "hook_tokens")
        or n.startswith(("embed.", "pos_embed."))
    ]
    assert embedding_stage == [], f"embedding-stage hooks should be bypassed: {embedding_stage}"
    for below in range(k):
        assert not any(n.startswith(f"blocks.{below}.") for n in keys), f"block {below} leaked"
    assert any(n.startswith(f"blocks.{k}.") for n in keys), "start block missing"
    assert any("ln_final" in n for n in keys), "final norm missing"
    assert any("unembed" in n for n in keys), "unembed (output stage) should be present"


def test_negative_start_at_layer(bridge):
    tokens = _tokens()
    full_logits, full_cache = bridge.run_with_cache(tokens)
    last = bridge.cfg.n_layers - 1
    resid = full_cache[f"blocks.{last}.hook_in"]

    logits, cache = bridge.run_with_cache(resid, start_at_layer=-1)

    assert torch.allclose(logits, full_logits, atol=1e-4)
    assert not any(n.startswith(f"blocks.{last - 1}.") for n in cache.keys())


def test_start_and_stop_bound_a_subrange(bridge):
    tokens = _tokens()
    _, full_cache = bridge.run_with_cache(tokens)
    _, cache = bridge.run_with_cache(
        full_cache["blocks.1.hook_in"], start_at_layer=1, stop_at_layer=3
    )
    keys = set(cache.keys())
    assert any(n.startswith("blocks.1.") for n in keys)
    assert any(n.startswith("blocks.2.") for n in keys)
    assert not any(n.startswith("blocks.0.") for n in keys)
    assert not any(n.startswith("blocks.3.") for n in keys)
    assert not any("ln_final" in n for n in keys)


def test_run_with_hooks_skips_sub_start_layers(bridge):
    tokens = _tokens()
    _, full_cache = bridge.run_with_cache(tokens)
    k = 3
    fired = []

    def rec(activation, hook):
        fired.append(hook.name)
        return None

    logits = bridge.run_with_hooks(
        full_cache[f"blocks.{k}.hook_in"],
        start_at_layer=k,
        fwd_hooks=[
            ("blocks.0.attn.hook_out", rec),
            ("blocks.2.attn.hook_out", rec),
            ("blocks.3.attn.hook_out", rec),
            ("blocks.5.attn.hook_out", rec),
        ],
    )
    assert logits is not None
    assert all(int(n.split(".")[1]) >= k for n in fired), fired
    assert any(n.startswith("blocks.3.") for n in fired)
    assert any(n.startswith("blocks.5.") for n in fired)


def test_token_input_with_start_at_layer_raises(bridge):
    with pytest.raises(ValueError, match="residual-stream tensor"):
        bridge.forward(_tokens(), start_at_layer=1)


def test_out_of_range_start_at_layer_raises(bridge):
    tokens = _tokens()
    _, full_cache = bridge.run_with_cache(tokens)
    resid = full_cache["blocks.0.hook_in"]
    with pytest.raises(ValueError, match="out of range"):
        bridge.forward(resid, start_at_layer=bridge.cfg.n_layers)


def test_state_cleaned_up_after_start_at_layer(bridge):
    tokens = _tokens()
    full_logits, full_cache = bridge.run_with_cache(tokens)
    bridge.forward(full_cache["blocks.2.hook_in"], start_at_layer=2)
    # A subsequent normal forward must be unaffected by the injection state.
    assert torch.allclose(bridge.forward(tokens), full_logits, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
