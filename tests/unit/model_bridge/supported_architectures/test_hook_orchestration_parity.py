"""BridgeCore hook-orchestration behaviour.

`reset_hooks` direction/permanence/level/clear-contexts selectivity, the
`mod_dict` accessor, and the `check_hooks_to_add` extension point -- the
surface migrated hook code relies on.

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


def _record(events, label):
    def hook(value, hook=None):
        events.append(label)
        return value

    return hook


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

    def test_invoked_by_run_with_hooks(self, monkeypatch):
        """The extension point must fire for run_with_hooks too, not just
        add_hook -- run_with_hooks is the common hooking path (matches HT)."""
        bridge = _build()
        seen: list[str] = []

        def recording(
            self, hook_point, hook_point_name, hook, dir="fwd", is_permanent=False, prepend=False
        ):
            seen.append(hook_point_name)

        monkeypatch.setattr(type(bridge), "check_hooks_to_add", recording)
        tokens = torch.randint(0, 64, (1, 4))
        bridge.run_with_hooks(tokens, fwd_hooks=[("blocks.0.hook_out", _noop)])
        assert "blocks.0.hook_out" in seen

    def test_invoked_by_hooks_context_manager(self, monkeypatch):
        bridge = _build()
        seen: list[str] = []

        def recording(
            self, hook_point, hook_point_name, hook, dir="fwd", is_permanent=False, prepend=False
        ):
            seen.append(hook_point_name)

        monkeypatch.setattr(type(bridge), "check_hooks_to_add", recording)
        with bridge.hooks(fwd_hooks=[("blocks.0.hook_out", _noop)]):
            pass
        assert "blocks.0.hook_out" in seen


class TestTemporaryHookScopes:
    @pytest.mark.parametrize("helper", ["hooks", "run_with_hooks", "run_with_cache"])
    def test_preexisting_forward_hook_survives_helper_cleanup(self, helper):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]
        events: list[str] = []
        hook_point.add_hook(_record(events, "existing"))

        if helper == "hooks":
            with bridge.hooks(fwd_hooks=[(hook_name, _record(events, "temporary"))]):
                bridge(tokens)
        elif helper == "run_with_hooks":
            bridge.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, _record(events, "temporary"))],
            )
        else:
            bridge.run_with_cache(tokens, names_filter=hook_name)

        events_after_helper = events.copy()
        assert len(hook_point.fwd_hooks) == 1

        bridge(tokens)

        assert events == events_after_helper + ["existing"]

    def test_nested_context_removes_only_inner_hooks(self):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]
        events: list[str] = []

        with bridge.hooks(fwd_hooks=[(hook_name, _record(events, "outer"))]):
            with bridge.hooks(fwd_hooks=[(hook_name, _record(events, "inner"))]):
                bridge(tokens)
            assert len(hook_point.fwd_hooks) == 1
            bridge(tokens)

        assert events == ["outer", "inner", "outer"]
        assert not hook_point.has_hooks()

    def test_nested_run_with_cache_preserves_outer_hook(self):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]
        events: list[str] = []

        with bridge.hooks(fwd_hooks=[(hook_name, _record(events, "outer"))]):
            bridge.run_with_cache(tokens, names_filter=hook_name)
            assert len(hook_point.fwd_hooks) == 1
            bridge(tokens)

        assert events == ["outer", "outer"]
        assert not hook_point.has_hooks()

    def test_run_with_cache_exception_preserves_preexisting_hook(self):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]

        def raising_hook(value, hook=None):
            raise RuntimeError("existing hook failed")

        hook_point.add_hook(raising_hook)

        with pytest.raises(RuntimeError, match="existing hook failed"):
            bridge.run_with_cache(tokens, names_filter=hook_name)

        assert len(hook_point.fwd_hooks) == 1

    def test_preexisting_backward_hook_survives_context_cleanup(self):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]
        events: list[str] = []

        def existing_hook(gradient, hook=None):
            events.append("existing")

        def temporary_hook(gradient, hook=None):
            events.append("temporary")

        hook_point.add_hook(existing_hook, dir="bwd")
        with bridge.hooks(bwd_hooks=[(hook_name, temporary_hook)]):
            bridge(tokens).sum().backward()

        events_after_context = events.copy()
        assert len(hook_point.bwd_hooks) == 1

        bridge.zero_grad()
        bridge(tokens).sum().backward()

        assert events == events_after_context + ["existing"]

    def test_run_with_cache_incl_bwd_preserves_preexisting_hooks(self):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]
        hook_point.add_hook(_noop, dir="fwd")
        hook_point.add_hook(_noop, dir="bwd")

        bridge.run_with_cache(
            tokens,
            names_filter=hook_name,
            incl_bwd=True,
            return_type="loss",
        )

        assert len(hook_point.fwd_hooks) == 1
        assert len(hook_point.bwd_hooks) == 1

    def test_permanent_hook_survives_temporary_scope_cleanup(self):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]
        bridge.add_perma_hook(hook_name, _noop)

        bridge.run_with_hooks(tokens, fwd_hooks=[(hook_name, _noop)])

        assert len(hook_point.fwd_hooks) == 1
        assert hook_point.fwd_hooks[0].is_permanent

    @pytest.mark.parametrize("helper", ["hooks", "run_with_hooks"])
    def test_reset_hooks_end_false_retains_temporary_hooks(self, helper):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        hook_name = "blocks.0.hook_out"
        hook_point = bridge._hook_registry[hook_name]

        if helper == "hooks":
            with bridge.hooks(fwd_hooks=[(hook_name, _noop)], reset_hooks_end=False):
                bridge(tokens)
        else:
            bridge.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, _noop)],
                reset_hooks_end=False,
            )

        assert bridge.context_level == 0
        assert len(hook_point.fwd_hooks) == 1

    def test_callable_filter_adds_one_hook_for_canonical_and_alias_names(self):
        bridge = _build()
        tokens = torch.randint(0, 64, (1, 4))
        canonical_name = "blocks.0.mlp.hook_out"
        alias_name = "blocks.0.hook_mlp_out"
        hook_point = bridge.hook_dict[canonical_name]
        events: list[str] = []

        with bridge.hooks(
            fwd_hooks=[
                (
                    lambda name: name in {canonical_name, alias_name},
                    _record(events, "temporary"),
                )
            ]
        ):
            bridge(tokens)

        assert events == ["temporary"]
        assert not hook_point.has_hooks()
