"""Tests for gated hook validation (issue #1688).

Adding a hook to a gated-off hook point (hook_result, hook_mlp_in, hook_attn_in,
hook_{q,k,v}_input) should fail loudly, not silently accept the hook and never fire it.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


def _cfg(**overrides) -> TransformerBridgeConfig:
    base = dict(
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
    base.update(overrides)
    return TransformerBridgeConfig(**base)


def test_add_hook_rejects_gated_attn_result():
    """add_hook on hook_result with use_attn_result=False raises a clear ValueError."""
    bridge = TransformerBridge.boot_native(_cfg())
    with pytest.raises(ValueError, match="use_attn_result"):
        bridge.add_hook("blocks.0.attn.hook_result", lambda t, hook=None: t)


def test_add_hook_rejects_gated_split_qkv_input():
    """add_hook on hook_q_input with use_split_qkv_input=False raises a clear ValueError."""
    bridge = TransformerBridge.boot_native(_cfg())
    with pytest.raises(ValueError, match="use_split_qkv_input"):
        bridge.add_hook("blocks.0.attn.hook_q_input", lambda t, hook=None: t)


def test_add_hook_rejects_gated_mlp_in():
    """add_hook on hook_mlp_in with use_hook_mlp_in=False raises a clear ValueError."""
    bridge = TransformerBridge.boot_native(_cfg())
    with pytest.raises(ValueError, match="use_hook_mlp_in"):
        bridge.add_hook("blocks.0.hook_mlp_in", lambda t, hook=None: t)


def test_add_hook_rejects_gated_attn_in():
    """add_hook on hook_attn_in with use_attn_in=False raises a clear ValueError."""
    bridge = TransformerBridge.boot_native(_cfg())
    with pytest.raises(ValueError, match="use_attn_in"):
        bridge.add_hook("blocks.0.hook_attn_in", lambda t, hook=None: t)


def test_add_hook_succeeds_after_enabling_setter():
    """Regression guard: enabling the flag via the setter still lets the hook fire."""
    bridge = TransformerBridge.boot_native(_cfg())
    bridge.set_use_hook_mlp_in(True)

    fired = []
    bridge.add_hook("blocks.0.hook_mlp_in", lambda t, hook=None: fired.append(1) or t)

    tokens = torch.randint(0, 16, (1, 8))
    bridge(tokens, return_type="logits")

    assert len(fired) > 0, "Hook did not fire after enabling use_hook_mlp_in via the setter"


def test_run_with_cache_warns_on_fully_gated_names_filter():
    """run_with_cache with a filter matching only gated-off names warns instead of
    silently returning an empty cache."""
    bridge = TransformerBridge.boot_native(_cfg())
    tokens = torch.randint(0, 16, (1, 8))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, cache = bridge.run_with_cache(tokens, names_filter=["blocks.0.hook_mlp_in"])

    assert len(cache) == 0
    assert any("gated-off" in str(w.message) for w in caught), (
        "Expected a warning naming the gated-off hook, got: " f"{[str(w.message) for w in caught]}"
    )


def test_add_hook_callable_filter_warns_and_skips_gated_points():
    """A callable filter matching only gated-off points attaches nothing and warns,
    rather than leaving dead hooks that never fire."""
    bridge = TransformerBridge.boot_native(_cfg())
    tokens = torch.randint(0, 16, (1, 8))

    fired = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge.add_hook(
            lambda name: name.endswith("hook_mlp_in"),
            lambda t, hook=None: fired.append(1) or t,
        )
    bridge(tokens, return_type="logits")

    # Not merely "did not fire" — a gated point never fires even when a dead
    # hook is attached, so assert nothing was attached in the first place.
    assert not bridge.hook_dict["blocks.0.hook_mlp_in"].fwd_hooks
    assert not fired, "Gated-off hook fired; the filter should have skipped it"
    assert any("gated-off" in str(w.message) for w in caught), (
        "Expected a warning naming the skipped gated-off hook, got: "
        f"{[str(w.message) for w in caught]}"
    )


def test_add_hook_callable_filter_attaches_once_flag_enabled():
    """The same filter attaches and fires — silently — once the setter enables the flag."""
    bridge = TransformerBridge.boot_native(_cfg())
    bridge.set_use_hook_mlp_in(True)
    tokens = torch.randint(0, 16, (1, 8))

    fired = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge.add_hook(
            lambda name: name.endswith("hook_mlp_in"),
            lambda t, hook=None: fired.append(1) or t,
        )
    bridge(tokens, return_type="logits")

    assert fired, "Hook did not fire after enabling use_hook_mlp_in via the setter"
    assert not [w for w in caught if "gated-off" in str(w.message)]


def test_add_hook_callable_filter_leaves_ungated_points_alone():
    """Control: a filter over an ungated point attaches and fires without warning."""
    bridge = TransformerBridge.boot_native(_cfg())
    tokens = torch.randint(0, 16, (1, 8))

    fired = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge.add_hook(
            lambda name: name == "blocks.0.hook_resid_post",
            lambda t, hook=None: fired.append(1) or t,
        )
    bridge(tokens, return_type="logits")

    assert fired, "Ungated hook should still attach and fire"
    assert not [w for w in caught if "gated-off" in str(w.message)]
