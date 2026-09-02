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


def test_run_with_hooks_callable_filter_warns_and_attaches_nothing():
    """The filter branch must warn and pre-skip, not silently attach dead hooks."""
    bridge = TransformerBridge.boot_native(_cfg())
    tokens = torch.tensor([[1, 2, 3]])
    fired: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge.run_with_hooks(
            tokens,
            fwd_hooks=[
                (lambda n: n.endswith("hook_mlp_in"), lambda t, hook=None: fired.append(1) or t)
            ],
        )
    assert not bridge.hook_dict["blocks.0.hook_mlp_in"].fwd_hooks
    assert not fired
    assert any("gated-off" in str(w.message) for w in caught)


def test_hooks_ctx_rejects_explicit_gated_name():
    """The hooks() context manager raises on an explicitly named gated point."""
    bridge = TransformerBridge.boot_native(_cfg())
    with pytest.raises(ValueError, match="use_hook_mlp_in"):
        with bridge.hooks(fwd_hooks=[("blocks.0.hook_mlp_in", lambda t, hook=None: t)]):
            pass


def test_hooks_ctx_callable_filter_warns_and_attaches_nothing():
    """The hooks() filter branch warns and pre-skips gated matches."""
    bridge = TransformerBridge.boot_native(_cfg())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with bridge.hooks(
            fwd_hooks=[(lambda n: n.endswith("hook_mlp_in"), lambda t, hook=None: t)]
        ):
            assert not bridge.hook_dict["blocks.0.hook_mlp_in"].fwd_hooks
    assert any("gated-off" in str(w.message) for w in caught)


def test_check_hooks_to_add_is_the_gate():
    """The documented extension point itself performs the gating check."""
    bridge = TransformerBridge.boot_native(_cfg())
    hp = bridge.hook_dict["blocks.0.hook_mlp_in"]
    with pytest.raises(ValueError, match="use_hook_mlp_in"):
        bridge.check_hooks_to_add(hp, "blocks.0.hook_mlp_in", lambda t, hook=None: t)


def test_add_caching_hooks_warns_for_gated_names():
    """Explicitly requesting a gated name for caching warns instead of silently
    returning a cache that will never fill."""
    bridge = TransformerBridge.boot_native(_cfg())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cache = bridge.add_caching_hooks(names_filter=["blocks.0.hook_mlp_in"])
    bridge.reset_hooks()
    assert "blocks.0.hook_mlp_in" not in cache
    assert any("gated-off" in str(w.message) for w in caught)


def test_gated_paths_work_once_enabled():
    """Control: with the flag on, all four previously silent paths attach and fire."""
    bridge = TransformerBridge.boot_native(_cfg())
    bridge.set_use_hook_mlp_in(True)
    tokens = torch.tensor([[1, 2, 3]])
    fired: list = []
    bridge.run_with_hooks(
        tokens,
        fwd_hooks=[
            (lambda n: n.endswith("hook_mlp_in"), lambda t, hook=None: fired.append("rwh") or t)
        ],
    )
    with bridge.hooks(
        fwd_hooks=[("blocks.0.hook_mlp_in", lambda t, hook=None: fired.append("ctx") or t)]
    ):
        bridge(tokens, return_type="logits")
    cache = bridge.add_caching_hooks(names_filter=["blocks.0.hook_mlp_in"])
    bridge(tokens, return_type="logits")
    bridge.reset_hooks()
    assert fired == ["rwh", "ctx"]
    assert "blocks.0.hook_mlp_in" in cache


def _tiny_bert_bridge():
    """BERT-style bridge whose blocks.N.hook_mlp_in is an alias OVERRIDE onto an
    always-firing point (mlp.in.hook_in) — alias != canonical, unlike boot_native."""
    import copy

    from transformers import BertConfig, BertForMaskedLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return build_bridge_from_module(
        BertForMaskedLM(cfg).eval(),
        "BertForMaskedLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    )


def test_alias_override_onto_firing_point_is_not_gated():
    """Gating keys on the POINT, not the spelling: on BERT, hook_mlp_in resolves
    to mlp.in.hook_in, which always fires — refusing it would reject a working
    hook. All four attach paths must accept it, and it must actually fire."""
    bridge = _tiny_bert_bridge()
    ids = torch.tensor([[1, 2, 3]])
    fired: list = []

    bridge.add_hook("blocks.0.hook_mlp_in", lambda t, hook=None: fired.append("add") or t)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge(ids)
        with bridge.hooks(
            fwd_hooks=[("blocks.0.hook_mlp_in", lambda t, hook=None: fired.append("ctx") or t)]
        ):
            bridge(ids)
        bridge.run_with_hooks(
            ids, fwd_hooks=[("blocks.0.hook_mlp_in", lambda t, hook=None: fired.append("rwh") or t)]
        )
    bridge.reset_hooks()
    assert fired == ["add", "add", "ctx", "add", "rwh"], fired
    assert not any("gated-off" in str(w.message) for w in caught)

    _, cache = bridge.run_with_cache(ids, names_filter=["blocks.0.hook_mlp_in"])
    assert "blocks.0.hook_mlp_in" in cache


def test_default_caching_sweep_composes_through_hooks():
    """get_caching_hooks() with no filter must not emit gated names — its
    documented use is feeding hooks()/run_with_hooks, which raise on them."""
    bridge = TransformerBridge.boot_native(_cfg())
    tokens = torch.tensor([[1, 2, 3]])
    cache, fwd_hooks, _ = bridge.get_caching_hooks()
    with bridge.hooks(fwd_hooks=fwd_hooks):
        bridge(tokens, return_type="logits")
    assert cache, "default sweep cached nothing"
    assert not any(name.endswith("hook_mlp_in") for name, _ in fwd_hooks)


def test_default_add_caching_hooks_does_not_warn():
    """The default filter matches everything; that must not read as an explicit
    request for gated names (run_with_cache's rule)."""
    bridge = TransformerBridge.boot_native(_cfg())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge.add_caching_hooks()
    bridge.reset_hooks()
    assert not any("gated-off" in str(w.message) for w in caught)
