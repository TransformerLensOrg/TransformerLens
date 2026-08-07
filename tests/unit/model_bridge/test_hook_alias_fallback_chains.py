"""Fallback-chain (list-valued) hook alias targets resolve by priority."""

from transformer_lens.model_bridge.bridge_core import BridgeCore


class _FakeComponent:
    def __init__(self, hook_aliases=None, submodules=None):
        self.hook_aliases = hook_aliases or {}
        self.submodules = submodules or {}


def _resolve(mapping, hook_names):
    core = BridgeCore.__new__(BridgeCore)
    component_aliases = core._collect_component_aliases(mapping)
    return dict(
        BridgeCore._compute_hook_aliases_cached(
            tuple(sorted(hook_names)), tuple(sorted(component_aliases.items()))
        )
    )


def test_string_targets_unchanged():
    mapping = {"blocks": _FakeComponent({"hook_out_alias": "mlp.hook_out"})}
    aliases = _resolve(mapping, ["blocks.0.mlp.hook_out"])
    assert aliases == {"blocks.0.hook_out_alias": "blocks.0.mlp.hook_out"}


def test_list_target_prefers_first_live_candidate():
    mapping = {"blocks": _FakeComponent({"hook_mlp_out": ["ln2_post.hook_out", "mlp.hook_out"]})}
    # Both candidates exist (post-norm layer): the first must win.
    aliases = _resolve(mapping, ["blocks.0.ln2_post.hook_out", "blocks.0.mlp.hook_out"])
    assert aliases["blocks.0.hook_mlp_out"] == "blocks.0.ln2_post.hook_out"


def test_list_target_falls_back_when_first_absent():
    mapping = {"blocks": _FakeComponent({"hook_mlp_out": ["ln2_post.hook_out", "mlp.hook_out"]})}
    # Linear-attention layer: no ln2_post hook — the fallback resolves.
    aliases = _resolve(mapping, ["blocks.1.mlp.hook_out"])
    assert aliases["blocks.1.hook_mlp_out"] == "blocks.1.mlp.hook_out"


def test_per_block_priority_is_independent():
    mapping = {"blocks": _FakeComponent({"hook_mlp_out": ["ln2_post.hook_out", "mlp.hook_out"]})}
    aliases = _resolve(
        mapping,
        [
            "blocks.0.ln2_post.hook_out",
            "blocks.0.mlp.hook_out",
            "blocks.1.mlp.hook_out",
        ],
    )
    assert aliases["blocks.0.hook_mlp_out"] == "blocks.0.ln2_post.hook_out"
    assert aliases["blocks.1.hook_mlp_out"] == "blocks.1.mlp.hook_out"
