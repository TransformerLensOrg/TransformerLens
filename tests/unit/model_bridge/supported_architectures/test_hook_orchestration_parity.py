"""HookedRootModule-parity for BridgeCore hook orchestration.

`reset_hooks` direction/permanence/level/clear-contexts selectivity, the
`mod_dict` accessor, and the `check_hooks_to_add` extension point -- the
surface migrated hook code relies on when moving off HookedTransformer.

Uses the network-free pretrain mock bridge (real block tree, HT-style
hook aliases like `blocks.0.hook_mlp_out`) rather than a thin stub, so the
"backward hook on a non-io hook point" case -- which the old registry walk
missed -- is actually exercised.
"""
from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge.supported_architectures.pretrain import (
    build_pretrain_bridge,
)

from ._pretrain_mocks import TinyPretrainModel, make_cfg


def _build():
    cfg = make_cfg(n_layers=2)
    model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=2, d_ff=32, vocab_size=64)
    return build_pretrain_bridge(model, cfg)


def _noop(value, hook=None):
    return value


class TestResetHooks:
    def test_removes_forward_and_backward_from_every_point(self):
        """Default direction='both' clears bwd hooks even on non-io points
        (block-level hook_mlp_in etc.), which the old fwd-only walk left behind."""
        bridge = _build()
        for hp in bridge.hook_points():
            hp.add_hook(_noop, dir="fwd")
            hp.add_hook(_noop, dir="bwd")

        bridge.reset_hooks()

        leftover = [
            hp.name
            for hp in bridge.hook_points()
            if hp.has_hooks(dir="both", including_permanent=False)
        ]
        assert leftover == []

    def test_direction_is_selective(self):
        bridge = _build()
        hp = bridge._hook_registry["blocks.0.hook_out"]
        hp.add_hook(_noop, dir="fwd")
        hp.add_hook(_noop, dir="bwd")

        bridge.reset_hooks(direction="fwd")
        assert not hp.has_hooks(dir="fwd")
        assert hp.has_hooks(dir="bwd")

        bridge.reset_hooks(direction="bwd")
        assert not hp.has_hooks(dir="bwd")

    def test_permanent_hooks_survive_default_reset(self):
        bridge = _build()
        hp = bridge._hook_registry["blocks.0.hook_out"]
        bridge.add_perma_hook("blocks.0.hook_out", _noop)

        bridge.reset_hooks()

        assert hp.has_hooks(dir="fwd", including_permanent=True)

    def test_including_permanent_removes_permanent_hooks(self):
        bridge = _build()
        hp = bridge._hook_registry["blocks.0.hook_out"]
        bridge.add_perma_hook("blocks.0.hook_out", _noop)

        bridge.reset_hooks(including_permanent=True)

        assert not hp.has_hooks(dir="fwd", including_permanent=True)

    def test_level_is_selective(self):
        bridge = _build()
        hp = bridge._hook_registry["blocks.0.hook_out"]
        hp.add_hook(_noop, dir="fwd", level=0)
        hp.add_hook(_noop, dir="fwd", level=1)

        bridge.reset_hooks(level=0)

        assert not hp.has_hooks(dir="fwd", level=0)
        assert hp.has_hooks(dir="fwd", level=1)

    def test_clear_contexts_flag(self):
        bridge = _build()
        hp = bridge._hook_registry["blocks.0.hook_out"]

        hp.ctx["scratch"] = 1
        bridge.reset_hooks(clear_contexts=False)
        assert hp.ctx == {"scratch": 1}

        bridge.reset_hooks(clear_contexts=True)
        assert hp.ctx == {}


class TestModDict:
    def test_canonical_and_alias_names_resolve_to_same_hook_point(self):
        bridge = _build()
        mod_dict = bridge.mod_dict
        assert mod_dict["blocks.0.hook_mlp_out"] is mod_dict["blocks.0.mlp.hook_out"]

    def test_includes_non_hook_modules(self):
        """mod_dict is a superset of hook_dict -- it also exposes container
        modules (blocks, block 0) that hook_dict omits."""
        bridge = _build()
        mod_dict = bridge.mod_dict
        assert "blocks" in mod_dict
        assert "blocks.0" in mod_dict
        for name, hook_point in bridge.hook_dict.items():
            assert mod_dict[name] is hook_point


class TestCheckHooksToAdd:
    def test_invoked_before_add(self, monkeypatch):
        bridge = _build()
        seen: list[str] = []

        def recording(
            self, hook_point, hook_point_name, hook, dir="fwd", is_permanent=False, prepend=False
        ):
            seen.append(hook_point_name)

        monkeypatch.setattr(type(bridge), "check_hooks_to_add", recording)
        bridge.add_hook("blocks.0.hook_out", _noop)
        assert seen == ["blocks.0.hook_out"]

    def test_can_veto_hook_addition(self, monkeypatch):
        """A raising override blocks the add -- the hook must not land."""
        bridge = _build()
        hp = bridge._hook_registry["blocks.0.hook_out"]

        def rejecting(
            self, hook_point, hook_point_name, hook, dir="fwd", is_permanent=False, prepend=False
        ):
            raise AssertionError(f"hook {hook_point_name} rejected")

        monkeypatch.setattr(type(bridge), "check_hooks_to_add", rejecting)
        with pytest.raises(AssertionError, match="rejected"):
            bridge.add_hook("blocks.0.hook_out", _noop)
        assert not hp.has_hooks(dir="fwd")
