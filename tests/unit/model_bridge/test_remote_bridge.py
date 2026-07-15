"""RemoteBridge contract tests: same hook namespace as TransformerBridge, no
torch-only surface, no nn.Module parentage."""
from __future__ import annotations

import pytest
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.model_bridge.driver_protocol import ForwardResult
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
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


def _stub_adapter() -> ArchitectureAdapter:
    adapter = ArchitectureAdapter(_cfg())
    adapter.component_mapping = {}  # RemoteBridge doesn't walk it
    return adapter


def _stub_driver(supported_hooks: frozenset[str] = frozenset({"blocks.0.hook_resid_pre"})):
    """Minimal Driver-conformant stub usable inside a RemoteBridge."""

    class StubDriver(DriverBase):
        supported_hook_points = supported_hooks
        _supported_features = frozenset()

        def __init__(self):
            super().__init__(_cfg(), tokenizer=None)
            self.forward_calls: list = []

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
            self.forward_calls.append({"input_ids": input_ids, "kwargs": kw})
            return ForwardResult(logits=None, captured={})

    return StubDriver()


class TestRemoteBridgeContract:
    """Type-shape promises distinguishing RemoteBridge from TransformerBridge."""

    def test_is_bridge_core(self):
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        assert isinstance(bridge, BridgeCore)

    def test_is_not_nn_module(self):
        """The load-bearing property that strips parameters/state_dict/etc."""
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        assert not isinstance(bridge, nn.Module)

    def test_lacks_torch_specific_surface(self):
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        for missing in (
            "parameters",
            "state_dict",
            "load_state_dict",
            "generate",
            "enable_compatibility_mode",
            "W_Q",
            "W_K",
            "W_V",
            "W_O",
            "W_in",
            "W_out",
            "_set_processed_weight_attributes",
        ):
            assert not hasattr(bridge, missing), (
                f"RemoteBridge unexpectedly has '{missing}' — "
                "the type split is meant to strip torch-only surface."
            )


class TestRemoteBridgeConstruction:
    """Driver must declare the hook namespace — no model to walk."""

    def test_requires_driver_supported_hook_points(self):
        empty_driver = _stub_driver(supported_hooks=frozenset())
        with pytest.raises(ValueError, match="supported_hook_points to be non-empty"):
            RemoteBridge(_stub_adapter(), tokenizer=None, driver=empty_driver)

    def test_cfg_device_is_none(self):
        """RemoteBridge has no local device — explicit None so tensor.to() is a no-op."""
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        assert bridge.cfg.device is None

    def test_builds_hook_registry_from_driver_declarations(self):
        hook_names = frozenset({"blocks.0.hook_resid_pre", "blocks.0.hook_resid_post"})
        bridge = RemoteBridge(
            _stub_adapter(), tokenizer=None, driver=_stub_driver(supported_hooks=hook_names)
        )
        assert frozenset(bridge._hook_registry) == hook_names
        for name in hook_names:
            assert bridge._hook_registry[name].name == name


class TestRemoteBridgeHookLifecycle:
    """RemoteBridge has no nn.Module children walk — registry must be canonical."""

    def test_reset_hooks_clears_registry_hooks(self):
        """Without the registry path, reset_hooks is a silent no-op on RemoteBridge."""
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        hp = bridge._hook_registry["blocks.0.hook_resid_pre"]
        hp.add_hook(lambda act, hook: None)
        assert len(hp.fwd_hooks) == 1
        bridge.reset_hooks()
        assert len(hp.fwd_hooks) == 0

    def test_list_hooks_works(self):
        """HookIntrospectionMixin.list_hooks is framework-agnostic (uses hook_dict)."""
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        bridge.add_hook("blocks.0.hook_resid_pre", lambda act, hook: None)
        listing = bridge.list_hooks()
        assert "blocks.0.hook_resid_pre" in listing
        assert len(listing["blocks.0.hook_resid_pre"]) == 1

    def test_add_hook_by_name_uses_registry(self):
        """RemoteBridge has no component tree; attribute walk would AttributeError."""
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        bridge.add_hook("blocks.0.hook_resid_pre", lambda act, hook: None)
        assert len(bridge._hook_registry["blocks.0.hook_resid_pre"].fwd_hooks) == 1

    def test_input_device_returns_none_for_meta(self):
        """Meta-device params (load_weights=False path) must not pull inputs onto meta."""
        import torch

        class MetaDriver(DriverBase):
            supported_hook_points = frozenset({"x"})
            _supported_features = frozenset({"parameters"})

            def __init__(self):
                super().__init__(_cfg(), tokenizer=None)

            def parameters(self):
                yield torch.empty(2, 2, device="meta")

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
                return ForwardResult()

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=MetaDriver())
        assert bridge._input_device() is None

    def test_aliases_resolve_via_registry(self):
        """Bridge-level aliases (hook_embed → embed.hook_out) work on RemoteBridge
        when the driver declares the canonical target — registry lookup, not
        attribute walk."""
        driver = _stub_driver(supported_hooks=frozenset({"embed.hook_out"}))
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=driver)
        # hook_embed is in BridgeCore.hook_aliases as ["embed_ln.hook_out", "embed.hook_out"]
        assert "hook_embed" in bridge.hook_dict
        assert bridge.hook_dict["hook_embed"] is bridge._hook_registry["embed.hook_out"]


class TestRemoteBridgeForward:
    """The driver-routed forward path."""

    def test_forward_calls_driver(self):
        driver = _stub_driver()
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=driver)
        import torch

        bridge.forward(torch.tensor([[1, 2, 3]]))
        assert len(driver.forward_calls) == 1

    def test_forward_return_type_loss(self):
        """return_type='loss' must compute loss, not silently return logits."""
        import torch

        class LogitsDriver(DriverBase):
            supported_hook_points = frozenset({"x"})
            _supported_features = frozenset()

            def __init__(self):
                super().__init__(_cfg(), tokenizer=None)

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
                return ForwardResult(logits=torch.randn(1, 3, 16), captured={})

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=LogitsDriver())
        loss = bridge.forward(torch.tensor([[1, 2, 3]]), return_type="loss")
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

    def test_forward_return_type_both(self):
        import torch

        class LogitsDriver(DriverBase):
            supported_hook_points = frozenset({"x"})
            _supported_features = frozenset()

            def __init__(self):
                super().__init__(_cfg(), tokenizer=None)

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
                return ForwardResult(logits=torch.randn(1, 3, 16), captured={})

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=LogitsDriver())
        out = bridge.forward(torch.tensor([[1, 2, 3]]), return_type="both")
        assert isinstance(out, tuple) and len(out) == 2
        logits, loss = out
        assert isinstance(logits, torch.Tensor) and logits.shape == (1, 3, 16)
        assert isinstance(loss, torch.Tensor) and loss.dim() == 0

    def test_forward_invalid_return_type_raises(self):
        """Unknown return_type used to silently return logits — must raise."""
        import torch

        class LogitsDriver(DriverBase):
            supported_hook_points = frozenset({"x"})
            _supported_features = frozenset()

            def __init__(self):
                super().__init__(_cfg(), tokenizer=None)

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
                return ForwardResult(logits=torch.randn(1, 3, 16), captured={})

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=LogitsDriver())
        with pytest.raises(ValueError, match="Invalid return_type"):
            bridge.forward(torch.tensor([[1, 2, 3]]), return_type="nonsense")

    def test_forward_rejects_list_of_strings(self):
        """list[str] used to be passed through as raw input_ids and crash in numpy."""
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(TypeError, match="list of strings"):
            bridge.forward(["hello", "world"])

    def test_forward_rejects_stop_at_layer(self):
        """stop_at_layer used to be silently swallowed by **kwargs."""
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(NotImplementedError, match="stop_at_layer"):
            bridge.forward(torch.tensor([[1, 2]]), stop_at_layer=1)

    def test_run_with_hooks_rejects_stop_at_layer(self):
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(NotImplementedError, match="stop_at_layer"):
            bridge.run_with_hooks(torch.tensor([[1, 2]]), stop_at_layer=1)

    def test_run_with_cache_rejects_stop_at_layer(self):
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(NotImplementedError, match="stop_at_layer"):
            bridge.run_with_cache(torch.tensor([[1, 2]]), stop_at_layer=1)

    def test_run_with_hooks_rejects_names_filter_kwarg(self):
        """The parameter existed in the signature but was never read."""
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(TypeError, match="names_filter"):
            bridge.run_with_hooks(torch.tensor([[1, 2]]), names_filter="blocks.0.hook_resid_pre")

    def test_forward_replays_captures(self):
        import torch

        recorded: list = []

        class CapturingDriver(DriverBase):
            supported_hook_points = frozenset({"blocks.0.hook_resid_pre"})
            _supported_features = frozenset()

            def __init__(self):
                super().__init__(_cfg(), tokenizer=None)

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
                return ForwardResult(
                    logits=None,
                    captured={"blocks.0.hook_resid_pre": torch.zeros(2, 3)},
                )

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=CapturingDriver())
        bridge._hook_registry["blocks.0.hook_resid_pre"].add_hook(
            lambda act, hook: recorded.append(act)
        )
        bridge.forward(torch.tensor([[1, 2]]))
        assert len(recorded) == 1
        assert tuple(recorded[0].shape) == (2, 3)


class TestRemoteBridgeHookSetEnforcement:
    """Driver-declared hook sets are a contract, not metadata — requests
    outside them must fail loud instead of yielding empty caches."""

    def test_add_hook_resolves_ht_alias(self):
        """add_hook accepts the same HT-style aliases run_with_hooks resolves."""
        driver = _stub_driver(supported_hooks=frozenset({"embed.hook_out"}))
        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=driver)
        bridge.add_hook("hook_embed", lambda act, hook: None)
        assert len(bridge._hook_registry["embed.hook_out"].fwd_hooks) == 1

    def test_run_with_hooks_unknown_hook_name_raises(self):
        """Unknown string names used to be silently skipped — unhooked forward."""
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(KeyError, match="does not exist"):
            with pytest.warns(UserWarning, match="read-only"):
                bridge.run_with_hooks(
                    torch.tensor([[1, 2]]),
                    fwd_hooks=[("blocks.0.attn.hook_z_typo", lambda act, hook: None)],
                )

    def test_run_with_cache_unmatched_names_filter_raises(self):
        """A filter outside the driver whitelist used to return (logits, {})."""
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(KeyError, match="matched no hook points"):
            bridge.run_with_cache(torch.tensor([[1, 2]]), names_filter="blocks.0.attn.hook_z")

    def test_run_with_cache_unmatched_list_filter_raises(self):
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        with pytest.raises(KeyError, match="matched no hook points"):
            bridge.run_with_cache(
                torch.tensor([[1, 2]]), names_filter=["nope.hook_a", "nope.hook_b"]
            )

    def test_run_with_cache_matching_filter_still_works(self):
        import torch

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=_stub_driver())
        _, cache = bridge.run_with_cache(
            torch.tensor([[1, 2]]), names_filter="blocks.0.hook_resid_pre"
        )
        assert cache is not None  # no raise; driver returns no captures here

    def test_non_fireable_hook_raises_backend_message(self):
        """Hooks the backend declares non-fireable must name the remedy."""
        import torch

        class PartialDriver(DriverBase):
            supported_hook_points = frozenset({"blocks.0.hook_resid_pre"})
            non_fireable_hook_points = frozenset({"blocks.0.attn.hook_pattern"})

            def __init__(self):
                super().__init__(_cfg(), tokenizer=None)

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
                return ForwardResult()

        bridge = RemoteBridge(_stub_adapter(), tokenizer=None, driver=PartialDriver())
        with pytest.raises(NotImplementedError, match="cannot fire"):
            bridge.add_hook("blocks.0.attn.hook_pattern", lambda act, hook: None)
        with pytest.raises(NotImplementedError, match="boot_transformers"):
            bridge.run_with_cache(torch.tensor([[1, 2]]), names_filter="blocks.0.attn.hook_pattern")
        with pytest.raises(NotImplementedError, match="cannot fire"):
            with pytest.warns(UserWarning, match="read-only"):
                bridge.run_with_hooks(
                    torch.tensor([[1, 2]]),
                    fwd_hooks=[("blocks.0.attn.hook_pattern", lambda act, hook: None)],
                )
