"""Traversal safety for the two alias collectors on the hook_dict path.

Both walk an arbitrary object graph (component templates and, per layer, bound
block subtrees). A component reachable from itself must be cut rather than
recursed forever, and a component legitimately shared under two names must
still contribute aliases at both — a globally-visited set would silently drop
the second path.
"""

from __future__ import annotations

import re
import sys

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import LinearBridge, MLPBridge

MODEL_NAME = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)


@pytest.fixture
def block_template(bridge: TransformerBridge):
    return bridge.adapter.component_mapping["blocks"]


def _shared_component() -> MLPBridge:
    """A component that actually declares hook_aliases (LinearBridge declares none,
    so sharing one would make a diamond test vacuous)."""
    return MLPBridge(
        name="c_fc",
        submodules={"in": LinearBridge(name="c_fc"), "out": LinearBridge(name="c_proj")},
    )


class TestTemplateCollectorTraversal:
    """_collect_component_aliases runs first on every hook_dict access."""

    def test_self_cycle_is_cut(self, bridge: TransformerBridge, block_template) -> None:
        mlp = block_template.submodules["mlp"]
        mlp.submodules["self_cycle"] = mlp
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(300)
        try:
            bridge._collect_component_aliases(bridge.adapter.component_mapping)
        finally:
            sys.setrecursionlimit(original_limit)
            del mlp.submodules["self_cycle"]

    def test_mutual_cycle_is_cut(self, bridge: TransformerBridge, block_template) -> None:
        mlp = block_template.submodules["mlp"]
        attn = block_template.submodules["attn"]
        mlp.submodules["to_attn"] = attn
        attn.submodules["to_mlp"] = mlp
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(300)
        try:
            bridge._collect_component_aliases(bridge.adapter.component_mapping)
        finally:
            sys.setrecursionlimit(original_limit)
            del mlp.submodules["to_attn"]
            del attn.submodules["to_mlp"]

    def test_shared_component_aliases_at_every_path(
        self, bridge: TransformerBridge, block_template
    ) -> None:
        """Diamond, not cycle: the guard must be path-scoped."""
        mlp = block_template.submodules["mlp"]
        shared = _shared_component()
        mlp.submodules["alpha"] = shared
        mlp.submodules["beta"] = shared
        try:
            aliases = bridge._collect_component_aliases(bridge.adapter.component_mapping)
        finally:
            del mlp.submodules["alpha"]
            del mlp.submodules["beta"]
        assert "blocks.mlp.alpha.hook_pre" in aliases
        assert "blocks.mlp.beta.hook_pre" in aliases

    def test_list_valued_target_does_not_reach_the_cached_consumer(
        self, bridge: TransformerBridge, block_template
    ) -> None:
        """_compute_hook_aliases_cached reverse-matches with str.endswith and is
        lru_cached, so a list target is neither matchable nor hashable."""
        attn = block_template.submodules["attn"]
        saved = attn.hook_aliases
        attn.hook_aliases = {**dict(saved), "hook_fallback": ["hook_out", "hook_in"]}
        try:
            bridge._hook_registry_initialized = False
            bridge._block_alias_cache = None
            bridge._initialize_hook_registry()
            bridge._collect_hook_aliases_from_registry()  # must not raise TypeError
        finally:
            attn.hook_aliases = saved
            bridge._hook_registry_initialized = False
            bridge._block_alias_cache = None
            bridge._initialize_hook_registry()


class TestBoundBlockWalkTraversal:
    """_collect_block_instance_aliases walks each bound block's subtree."""

    def test_self_cycle_in_a_bound_block_is_cut(self, bridge: TransformerBridge) -> None:
        mlp = bridge.blocks[0].mlp
        mlp.submodules["self_cycle"] = mlp
        bridge._block_alias_cache = None
        try:
            bridge._collect_block_instance_aliases()
        finally:
            del mlp.submodules["self_cycle"]
            bridge._block_alias_cache = None

    def test_shared_submodule_aliases_at_every_path(self, bridge: TransformerBridge) -> None:
        """A component reachable under two names must be walked at BOTH — a
        globally-visited guard silently skips the second path. The unresolved-alias
        report is the evidence; the resolved map is empty either way (vacuous).
        """
        mlp = bridge.blocks[0].mlp
        shared = _shared_component()
        mlp.submodules["alpha"] = shared
        mlp.submodules["beta"] = shared
        bridge._block_alias_cache = None
        try:
            with pytest.warns(UserWarning, match="did not resolve") as record:
                bridge._collect_block_instance_aliases()
        finally:
            del mlp.submodules["alpha"]
            del mlp.submodules["beta"]
            bridge._block_alias_cache = None

        reported = " ".join(str(w.message) for w in record)
        count = int(re.search(r"^(\d+) block hook alias", reported).group(1))
        # _shared_component declares 2 aliases; both paths walked => 2 per path.
        assert count == 4, f"expected 2 aliases x 2 paths, got {count}: {reported}"
