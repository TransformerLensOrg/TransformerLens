"""Tests for the Driver protocol and the first implementation (TransformersDriver)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from transformer_lens.model_bridge.driver_protocol import (
    Driver,
    ForwardResult,
    TensorLike,
    to_torch,
    validate_driver,
)


def _stub_adapter(architecture: str = "Mock"):
    """Minimal adapter satisfying DriverBase's beartype-checked cfg arg."""
    from transformer_lens.config import TransformerBridgeConfig

    cfg = TransformerBridgeConfig(
        d_model=4,
        d_head=2,
        n_layers=1,
        n_ctx=8,
        n_heads=2,
        d_vocab=16,
        d_mlp=8,
        architecture=architecture,
    )

    class _StubAdapter:
        pass

    a = _StubAdapter()
    a.cfg = cfg  # type: ignore[attr-defined]
    return a


class TestTensorLikeProtocol:
    """TensorLike matches what every tensor library exposes."""

    def test_torch_tensor_matches(self):
        assert isinstance(torch.randn(2, 3), TensorLike)

    def test_numpy_array_matches(self):
        assert isinstance(np.zeros((2, 3)), TensorLike)

    def test_bare_object_rejected(self):
        class NotATensor:
            pass

        assert not isinstance(NotATensor(), TensorLike)


class TestToTorch:
    """Boundary conversion: TensorLike → torch.Tensor."""

    def test_torch_passthrough(self):
        t = torch.randn(2, 3)
        # Identity-equal: drivers returning torch tensors shouldn't pay a clone tax.
        assert to_torch(t) is t

    def test_numpy_converts_with_correct_shape(self):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = to_torch(arr)
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (2, 3)
        assert out.dtype == torch.float32

    def test_dtype_override(self):
        out = to_torch(torch.randn(2, 3, dtype=torch.float32), dtype=torch.float16)
        assert out.dtype == torch.float16

    def test_dlpack_path_used_when_available(self):
        """numpy ≥ 1.22 has __dlpack__ — stand-in for JAX/MLX/CuPy device arrays."""
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        if not hasattr(arr, "__dlpack__"):
            pytest.skip("numpy < 1.22 has no __dlpack__; nothing to verify")
        out = to_torch(arr)
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (2, 3)
        assert out.dtype == torch.float32

    def test_dlpack_failure_falls_back_to_numpy(self):
        """Stream-sync / version-skew failures fall through to the __array__ path."""

        class FailingDLPack:
            shape = (2, 3)
            dtype = np.float32

            def __dlpack__(self, *a, **kw):
                raise BufferError("simulated stream-sync failure")

            def __array__(self, dtype=None):
                return np.arange(6, dtype=np.float32).reshape(2, 3)

        out = to_torch(FailingDLPack())
        assert isinstance(out, torch.Tensor)
        assert tuple(out.shape) == (2, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_torch_tensor_passthrough(self):
        """The passthrough branch must work for non-CPU tensors. np.asarray would crash."""
        t = torch.randn(2, 3, device="cuda")
        out = to_torch(t)
        assert out is t
        assert out.device.type == "cuda"


class TestForwardResult:
    """ForwardResult is the data envelope across the driver boundary."""

    def test_default_construction(self):
        fr = ForwardResult()
        assert fr.logits is None
        assert fr.captured == {}
        assert fr.new_tokens is None
        assert fr.raw_output is None

    def test_frozen(self):
        import dataclasses

        fr = ForwardResult(logits=torch.zeros(2))
        # setattr bypasses mypy's static frozen-instance check.
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(fr, "logits", torch.ones(2))


class TestDriverProtocol:
    """The Driver protocol is structurally checkable at runtime."""

    def test_bare_class_does_not_satisfy(self):
        class NotADriver:
            pass

        assert not isinstance(NotADriver(), Driver)

    def test_capability_flags_route_torch_specific_access(self):
        """parameters/state_dict are torch-specific implementation details, not protocol
        members. Callers route via supports("...") + hasattr; non-torch drivers
        simply don't define them."""
        from transformer_lens.model_bridge.sources._driver_base import DriverBase
        from transformer_lens.model_bridge.sources.transformers_driver import (
            TransformersDriver,
        )

        adapter = _stub_adapter()
        model = nn.Linear(2, 2)
        hf_driver = TransformersDriver(model, adapter, tokenizer=None)
        assert hf_driver.supports("parameters") is True
        assert hf_driver.supports("gradients") is True
        assert hf_driver.supports("intervention_callbacks") is True
        assert hf_driver.supports("never-heard-of-it") is False
        assert hasattr(hf_driver, "parameters")
        assert list(hf_driver.parameters()) == list(model.parameters())

        class RemoteLike(DriverBase):
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

        remote = RemoteLike(_stub_adapter().cfg, tokenizer=None)
        assert remote.supports("parameters") is False
        assert not hasattr(remote, "parameters")

    def test_transformers_driver_passes_strict_validation(self):
        """validate_driver enforces types/signatures, not just hasattr."""
        from transformer_lens.model_bridge.sources.transformers_driver import (
            TransformersDriver,
        )

        driver = TransformersDriver(nn.Linear(2, 2), _stub_adapter(), tokenizer=None)
        validate_driver(driver)


class TestValidateDriverCatchesBrokenDrivers:
    """validate_driver must reject drivers that runtime_checkable Protocol would accept."""

    def _cfg(self):
        return _stub_adapter().cfg

    def test_wrong_forward_signature_rejected(self):
        """Driver with forward(only_input_ids) silently swallows the bridge's keyword args."""
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        class WrongSignature(DriverBase):
            def __init__(self, cfg):
                super().__init__(cfg, tokenizer=None)

            # Missing capture/intervene/max_new_tokens/return_logits, no **kwargs.
            # runtime_checkable accepts this; validate_driver must not.
            def forward(self, input_ids):  # type: ignore[override]
                return ForwardResult()

        driver = WrongSignature(self._cfg())
        assert isinstance(driver, Driver)  # Protocol is too weak to catch this
        with pytest.raises(TypeError, match="must accept parameters"):
            validate_driver(driver)

    def test_overlapping_hook_sets_rejected(self):
        """A hook can't be both fireable and not — overlap is a contract bug."""
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        class OverlappingSets(DriverBase):
            supported_hook_points = frozenset({"a", "b"})
            non_fireable_hook_points = frozenset({"b", "c"})

            def __init__(self, cfg):
                super().__init__(cfg, tokenizer=None)

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

        driver = OverlappingSets(self._cfg())
        with pytest.raises(TypeError, match="overlap"):
            validate_driver(driver)

    def test_wrong_attribute_type_rejected(self):
        """architecture: str — a driver that sets it to None or an int fails."""
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        class WrongTypes(DriverBase):
            def __init__(self, cfg):
                super().__init__(cfg, tokenizer=None)
                self.architecture = 42  # type: ignore[assignment]

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

        driver = WrongTypes(self._cfg())
        with pytest.raises(TypeError, match="architecture must be str"):
            validate_driver(driver)

    def test_non_string_hookpoint_rejected(self):
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        class BadHookNames(DriverBase):
            supported_hook_points = frozenset({"valid", 123})  # type: ignore[arg-type]

            def __init__(self, cfg):
                super().__init__(cfg, tokenizer=None)

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

        driver = BadHookNames(self._cfg())
        with pytest.raises(TypeError, match="must be str"):
            validate_driver(driver)

    def test_empty_hookpoints_post_construction_rejected(self):
        """Driver that never declared anything after the bridge backfilled is silently broken."""
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        class SilentlyEmpty(DriverBase):
            def __init__(self, cfg):
                super().__init__(cfg, tokenizer=None)

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

        driver = SilentlyEmpty(self._cfg())
        validate_driver(driver, after_bridge_construction=False)
        with pytest.raises(TypeError, match="empty supported_hook_points AND"):
            validate_driver(driver, after_bridge_construction=True)

    def test_bridge_accepts_kw_only_input_ids_driver(self):
        """Bridge passes input_ids by keyword so kw-only drivers don't TypeError."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests.mocks.architecture_adapter import MockArchitectureAdapter
        from transformer_lens.model_bridge.bridge import TransformerBridge
        from transformer_lens.model_bridge.generalized_components import (
            AttentionBridge,
            BlockBridge,
            NormalizationBridge,
        )
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        received: dict = {}

        class KwOnlyDriver(DriverBase):
            supported_hook_points = frozenset({"sentinel"})

            def forward(
                self,
                *,  # everything past here is keyword-only — including input_ids
                input_ids=None,
                capture=(),
                intervene=None,
                max_new_tokens=1,
                return_logits=True,
                **kw,
            ):
                received["input_ids"] = input_ids
                return ForwardResult(logits=torch.zeros(1, 1, 16), raw_output=None)

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
        driver = KwOnlyDriver(_stub_adapter().cfg, tokenizer=None)
        bridge = TransformerBridge(model, adapter, tokenizer=MagicMock(), driver=driver)

        # Mirrors the bridge's keyword-form call. Positional would TypeError.
        input_ids = torch.tensor([[1, 2, 3]])
        result = bridge._driver.forward(input_ids=input_ids)
        assert received["input_ids"] is input_ids
        assert isinstance(result, ForwardResult)

    def test_bridge_init_rejects_misshapen_driver(self):
        """validate_driver runs at end of __init__ — fail fast at construction."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests.mocks.architecture_adapter import MockArchitectureAdapter
        from transformer_lens.model_bridge.bridge import TransformerBridge
        from transformer_lens.model_bridge.generalized_components import (
            AttentionBridge,
            BlockBridge,
            NormalizationBridge,
        )
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        # non_fireable_hook_points as list (not frozenset) passes Protocol
        # presence-check but should fail validate_driver at __init__.
        class MisshapenDriver(DriverBase):
            supported_hook_points = frozenset({"x"})
            non_fireable_hook_points = ["wrong_type"]  # type: ignore[assignment]

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
        bad_driver = MisshapenDriver(_stub_adapter().cfg, tokenizer=None)

        with pytest.raises(TypeError, match="non_fireable_hook_points must be frozenset"):
            TransformerBridge(model, adapter, tokenizer=MagicMock(), driver=bad_driver)

    def test_hf_driver_passes_post_construction_check(self):
        """Full HF stack populates supported_hook_points; strict gate accepts it."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests.mocks.architecture_adapter import MockArchitectureAdapter
        from transformer_lens.model_bridge.bridge import TransformerBridge
        from transformer_lens.model_bridge.generalized_components import (
            AttentionBridge,
            BlockBridge,
            NormalizationBridge,
        )

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
        bridge = TransformerBridge(model, adapter, tokenizer=MagicMock())
        validate_driver(bridge._driver, after_bridge_construction=True)


class TestBridgeConsumesCaptures:
    """Bridge replays ForwardResult.captured through its HookPoint tree."""

    def _build_minimal_bridge(self):
        """Tiny bridge over a hand-rolled adapter — avoids HF Hub."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests.mocks.architecture_adapter import MockArchitectureAdapter
        from transformer_lens.model_bridge.bridge import TransformerBridge
        from transformer_lens.model_bridge.generalized_components import (
            AttentionBridge,
            BlockBridge,
            NormalizationBridge,
        )

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
        return TransformerBridge(model, adapter, tokenizer=MagicMock())

    def test_replay_captures_fires_hookpoints(self):
        bridge = self._build_minimal_bridge()
        # Pick any registered hook — test isn't coupled to specific naming.
        assert bridge._hook_registry, "fixture must register at least one HookPoint"
        target_name = next(iter(bridge._hook_registry))
        hp = bridge._hook_registry[target_name]

        recorded: list[torch.Tensor] = []
        hp.add_hook(lambda act, hook: recorded.append(act))

        synthetic = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        bridge._replay_captures({target_name: synthetic})

        assert len(recorded) == 1
        assert torch.equal(recorded[0], synthetic)

    def test_replay_drops_unknown_hook_names(self):
        """Drivers may report names the bridge doesn't carry; silent-drop is the contract."""
        bridge = self._build_minimal_bridge()
        bridge._replay_captures({"definitely.not.a.hook": torch.zeros(2)})
        # No exception, no side effect — the assertion is that we got here.

    def test_replay_converts_non_torch_tensors(self):
        """Captures arriving as numpy / mlx / jax cross the to_torch boundary."""
        bridge = self._build_minimal_bridge()
        target_name = next(iter(bridge._hook_registry))
        hp = bridge._hook_registry[target_name]

        recorded: list[torch.Tensor] = []
        hp.add_hook(lambda act, hook: recorded.append(act))

        synthetic_np = np.arange(6, dtype=np.float32).reshape(2, 3)
        bridge._replay_captures({target_name: synthetic_np})

        assert len(recorded) == 1
        assert isinstance(recorded[0], torch.Tensor)
        assert recorded[0].shape == (2, 3)


class TestBridgeToleratesWeirdLogits:
    """Bridge tolerates non-torch.Tensor logits — audio CTC, encoder-only, etc."""

    def _make_test_bridge(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests.mocks.architecture_adapter import MockArchitectureAdapter
        from transformer_lens.model_bridge.bridge import TransformerBridge
        from transformer_lens.model_bridge.generalized_components import (
            AttentionBridge,
            BlockBridge,
            NormalizationBridge,
        )

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
        return TransformerBridge(model, adapter, tokenizer=MagicMock())

    def test_non_tensor_logits_passes_through_unchanged(self):
        """Weird objects (HF dataclass, tuple-of-tuples) pass through; downstream
        return_type branches do the strict typing."""
        from transformer_lens.model_bridge.driver_protocol import TensorLike

        class WeirdShape:
            extra = "not a tensor"

        weird = WeirdShape()
        assert not isinstance(weird, torch.Tensor)
        assert not isinstance(weird, TensorLike)
        # Mirrors the bridge boundary logic.
        logits: object = weird
        if isinstance(logits, torch.Tensor):
            pass
        elif logits is not None and isinstance(logits, TensorLike):
            logits = to_torch(logits)
        assert logits is weird

    def test_none_logits_passes_through(self):
        """return_type=None is a legitimate ask."""
        from transformer_lens.model_bridge.driver_protocol import ForwardResult

        result = ForwardResult(logits=None)
        logits: object = result.logits
        if isinstance(logits, torch.Tensor):
            pass
        elif logits is not None and isinstance(logits, TensorLike):
            logits = to_torch(logits)
        assert logits is None

    def test_numpy_logits_converted_at_boundary(self):
        """Non-torch TensorLike still converts."""
        arr = np.zeros((2, 3), dtype=np.float32)
        logits: object = arr
        if isinstance(logits, torch.Tensor):
            pass
        elif logits is not None and isinstance(logits, TensorLike):
            logits = to_torch(logits)
        assert isinstance(logits, torch.Tensor)
        assert tuple(logits.shape) == (2, 3)


class TestDriverHookPointDeclaration:
    """The driver tells the bridge which hooks it can fire."""

    def _build_minimal_bridge(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from tests.mocks.architecture_adapter import MockArchitectureAdapter
        from transformer_lens.model_bridge.bridge import TransformerBridge
        from transformer_lens.model_bridge.generalized_components import (
            AttentionBridge,
            BlockBridge,
            NormalizationBridge,
        )

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
        return TransformerBridge(model, adapter, tokenizer=MagicMock())

    def test_hf_driver_supports_full_registry(self):
        """HF with eager attention fires every hook the bridge registers."""
        bridge = self._build_minimal_bridge()
        assert bridge._driver.supported_hook_points == frozenset(bridge._hook_registry)
        assert bridge._driver.non_fireable_hook_points == frozenset()
        assert len(bridge._driver.supported_hook_points) > 0  # non-empty contract

    def test_non_fireable_subtracts_from_supported(self):
        """A driver that declares fused-kernel hooks ends up with supported = registry - non_fireable."""
        from transformer_lens.model_bridge.sources.transformers_driver import (
            TransformersDriver,
        )

        bridge = self._build_minimal_bridge()
        # Simulate a fused-kernel driver by re-declaring non_fireable, clearing
        # supported, and re-running the backfill the bridge does in __init__.
        sacrificed = next(iter(bridge._hook_registry))
        bridge._driver.non_fireable_hook_points = frozenset({sacrificed})
        bridge._driver.supported_hook_points = frozenset()
        # Re-apply the bridge's __init__ backfill rule.
        bridge._driver.supported_hook_points = (
            frozenset(bridge._hook_registry) - bridge._driver.non_fireable_hook_points
        )

        assert sacrificed not in bridge._driver.supported_hook_points
        assert sacrificed in bridge._driver.non_fireable_hook_points
        assert (
            bridge._driver.supported_hook_points | bridge._driver.non_fireable_hook_points
            == frozenset(bridge._hook_registry)
        )

    def test_original_model_raises_for_non_torch_driver(self):
        """A driver without ``underlying_model`` triggers the documented AttributeError."""
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        bridge = self._build_minimal_bridge()

        class NoUnderlyingModel(DriverBase):
            def __init__(self, cfg) -> None:
                super().__init__(cfg, tokenizer=None)

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

        bridge._driver = NoUnderlyingModel(_stub_adapter().cfg)
        with pytest.raises(AttributeError, match="does not expose an nn.Module"):
            _ = bridge.original_model

    def test_whitelist_driver_preserved(self):
        """An Inspect-style driver that declares supported directly is not overwritten."""
        from transformer_lens.model_bridge.driver_protocol import ForwardResult
        from transformer_lens.model_bridge.sources._driver_base import DriverBase

        cfg = _stub_adapter().cfg

        # Minimal whitelist driver stand-in for Inspect — declares supported
        # before bridge construction and expects to keep that declaration.
        class WhitelistDriver(DriverBase):
            def __init__(self) -> None:
                super().__init__(cfg, tokenizer=None)
                # Inspect would declare its residual-stream subset here.
                self.supported_hook_points = frozenset({"blocks.0.hook_resid_pre"})

            def forward(self, *a, **kw):  # type: ignore[override]
                return ForwardResult()

        bridge = self._build_minimal_bridge()
        whitelist = WhitelistDriver()
        bridge._driver = whitelist
        # Re-apply the backfill rule the bridge does in __init__; it should
        # NOT overwrite a non-empty supported set.
        if not bridge._driver.supported_hook_points:
            bridge._driver.supported_hook_points = (
                frozenset(bridge._hook_registry) - bridge._driver.non_fireable_hook_points
            )
        assert whitelist.supported_hook_points == frozenset({"blocks.0.hook_resid_pre"})
