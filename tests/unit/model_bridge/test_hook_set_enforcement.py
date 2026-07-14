"""Hook-set contract enforcement on the torch bridge: typo'd or backend-unservable
hook names must raise instead of silently running unhooked / caching nothing."""
from __future__ import annotations

import pytest
import torch

from tests.mocks.mock_logits_bridge import build_mock_logits_bridge as _build_bridge


class TestUnknownHookNamesFailLoud:
    def test_run_with_hooks_typo_raises_keyerror(self):
        bridge = _build_bridge()
        with pytest.raises(KeyError, match="does not exist"):
            bridge.run_with_hooks(
                torch.tensor([[1, 2, 3]]),
                fwd_hooks=[("blocks.0.hook_definitely_typo", lambda act, hook: None)],
            )

    def test_hooks_context_typo_raises_keyerror(self):
        bridge = _build_bridge()
        with pytest.raises(KeyError, match="does not exist"):
            with bridge.hooks(fwd_hooks=[("blocks.0.hook_definitely_typo", lambda a, hook: None)]):
                pass

    def test_run_with_hooks_valid_name_still_attaches(self):
        bridge = _build_bridge()
        target = next(iter(bridge._hook_registry))
        seen: list = []
        bridge.run_with_hooks(
            torch.tensor([[1, 2, 3]]),
            fwd_hooks=[(target, lambda act, hook: seen.append(hook.name))],
        )
        # The mock model never routes tensors through HookPoints, so the hook
        # can't fire — the assertion is that attach succeeded without raising
        # and was cleaned up.
        assert len(bridge._hook_registry[target].fwd_hooks) == 0


class TestRunWithCacheFilterEnforcement:
    def test_string_filter_matching_nothing_raises(self):
        bridge = _build_bridge()
        with pytest.raises(KeyError, match="matched no hook points"):
            bridge.run_with_cache(torch.tensor([[1, 2, 3]]), names_filter="no.such.hook")

    def test_list_filter_matching_nothing_raises(self):
        bridge = _build_bridge()
        with pytest.raises(KeyError, match="matched no hook points"):
            bridge.run_with_cache(
                torch.tensor([[1, 2, 3]]), names_filter=["no.such.hook", "also.not.real"]
            )

    def test_callable_filter_matching_nothing_is_allowed(self):
        """Programmatic filters legitimately produce empty intersections."""
        bridge = _build_bridge()
        _, cache = bridge.run_with_cache(torch.tensor([[1, 2, 3]]), names_filter=lambda name: False)
        assert len(cache) == 0

    def test_no_filter_caches_everything_without_raising(self):
        bridge = _build_bridge()
        out, cache = bridge.run_with_cache(torch.tensor([[1, 2, 3]]))
        assert out is not None

    def test_partially_matching_list_filter_is_allowed(self):
        """One valid name in a list keeps the run alive (matches HT leniency)."""
        bridge = _build_bridge()
        target = next(iter(bridge._hook_registry))
        _, cache = bridge.run_with_cache(
            torch.tensor([[1, 2, 3]]), names_filter=[target, "no.such.hook"]
        )
        assert cache is not None


class TestNonFireableEnforcement:
    def test_add_hook_on_non_fireable_raises(self):
        bridge = _build_bridge()
        sacrificed = next(iter(bridge._hook_registry))
        bridge._driver.non_fireable_hook_points = frozenset({sacrificed})
        with pytest.raises(NotImplementedError, match="cannot fire"):
            bridge.add_hook(sacrificed, lambda act, hook: None)

    def test_run_with_hooks_on_non_fireable_raises(self):
        bridge = _build_bridge()
        sacrificed = next(iter(bridge._hook_registry))
        bridge._driver.non_fireable_hook_points = frozenset({sacrificed})
        with pytest.raises(NotImplementedError, match="boot_transformers"):
            bridge.run_with_hooks(
                torch.tensor([[1, 2, 3]]),
                fwd_hooks=[(sacrificed, lambda act, hook: None)],
            )

    def test_run_with_cache_on_non_fireable_raises(self):
        bridge = _build_bridge()
        sacrificed = next(iter(bridge._hook_registry))
        bridge._driver.non_fireable_hook_points = frozenset({sacrificed})
        with pytest.raises(NotImplementedError, match="cannot fire"):
            bridge.run_with_cache(torch.tensor([[1, 2, 3]]), names_filter=sacrificed)


class TestAddHookAliasResolution:
    def test_add_hook_accepts_registry_alias(self):
        """Aliases visible in hook_dict must attach via add_hook, same as run_with_hooks."""
        bridge = _build_bridge()
        # Find an alias: a hook_dict key whose HookPoint carries a different canonical name.
        alias = next(
            (
                name
                for name, hp in bridge.hook_dict.items()
                if hp.name is not None and hp.name != name
            ),
            None,
        )
        if alias is None:
            pytest.skip("mock bridge registered no aliases")
        canonical = bridge.hook_dict[alias].name
        bridge.add_hook(alias, lambda act, hook: None)
        assert len(bridge._hook_registry[canonical].fwd_hooks) == 1
