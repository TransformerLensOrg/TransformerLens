"""Hook-set contract enforcement on the torch bridge: typo'd or backend-unservable
hook names must raise instead of silently running unhooked / caching nothing."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tests.mocks.architecture_adapter import MockArchitectureAdapter
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.driver_protocol import ForwardResult
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    NormalizationBridge,
)
from transformer_lens.model_bridge.sources._driver_base import DriverBase


def _cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=4,
        d_head=2,
        n_layers=1,
        n_ctx=8,
        n_heads=2,
        d_vocab=16,
        d_mlp=8,
        architecture="Mock",
    )


class _LogitsDriver(DriverBase):
    """Returns real logits so post-hook forward paths complete."""

    def __init__(self, model, cfg, tokenizer=None):
        super().__init__(cfg, tokenizer)
        # forward()'s encoder-decoder probe reads bridge.original_model.
        self.underlying_model = model

    def forward(
        self,
        input_ids=None,
        *,
        capture=(),
        intervene=None,
        max_new_tokens=1,
        return_logits=True,
        **kw,
    ):
        return ForwardResult(logits=torch.randn(1, 3, 16))


def _build_bridge() -> TransformerBridge:
    model = nn.Module()
    model.final_norm = nn.LayerNorm(10)
    model.encoder = nn.Module()
    model.encoder.layers = nn.ModuleList([nn.Module() for _ in range(1)])
    model.encoder.layers[0].norm1 = nn.LayerNorm(10)
    model.encoder.layers[0].self_attn = nn.Module()

    adapter = MockArchitectureAdapter()
    adapter.component_mapping = {
        "ln_final": NormalizationBridge(name="final_norm", config={}),
        "blocks": BlockBridge(
            name="encoder.layers",
            submodules={
                "ln1": NormalizationBridge(name="norm1", config={}),
                "attn": AttentionBridge(name="self_attn", config=SimpleNamespace(n_heads=1)),
            },
        ),
    }
    driver = _LogitsDriver(model, _cfg(), tokenizer=None)
    return TransformerBridge(model, adapter, tokenizer=MagicMock(), driver=driver)


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
