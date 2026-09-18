"""Vision benchmark (Phase 9) gating and wiring, download-free.

The real pixel forward/cache runs happen in per-model verification; here we pin
the classification plumbing, the phase gating in run_benchmark_suite, and the
shared encoder-benchmark wiring (result names, HF reference kwarg,
critical-hook selection) without loading a model.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from transformer_lens.benchmarks import main_benchmark  # noqa: E402
from transformer_lens.benchmarks.main_benchmark import (  # noqa: E402
    _adapter_applicable_phases,
    _phase_enabled,
)
from transformer_lens.benchmarks.utils import BenchmarkSeverity  # noqa: E402
from transformer_lens.benchmarks.vision import (  # noqa: E402
    benchmark_vision_cache,
    benchmark_vision_classification_decode,
    benchmark_vision_forward,
    benchmark_vision_representation_stability,
)
from transformer_lens.model_bridge.bridge import TransformerBridge  # noqa: E402
from transformer_lens.utilities.architectures import (  # noqa: E402
    classify_architecture,
    classify_model_config,
)


class TestVisionClassification:
    def test_vision_architectures_classify_as_vision(self):
        for arch in (
            "ViTModel",
            "ViTForImageClassification",
            "DeiTModel",
            "DeiTForImageClassification",
        ):
            assert classify_architecture(arch) == "vision"

    def test_classify_model_config_vision(self):
        cfg = SimpleNamespace(architectures=["ViTForImageClassification"])
        assert classify_model_config(cfg) == "vision"

    def test_audio_text_still_classifies_causal(self):
        # Audio-text models classify as causal LMs (bridge semantics).
        assert classify_architecture("Qwen2AudioForConditionalGeneration") == "causal_lm"


class TestPhaseGating:
    """should_run_phase = _phase_enabled(phase, phases filter, applicable_phases)."""

    @pytest.mark.parametrize(
        "phase,phases,applicable,expected",
        [
            # Text phases absent from applicable_phases are skipped.
            (1, None, [], False),
            (4, None, [], False),
            (3, None, [1, 2, 4], False),
            (2, None, [1, 2, 4], True),
            (1, None, [1, 2, 3, 4], True),
            # Modality phases (7/8/9) are never applicable_phases-gated.
            (7, None, [], True),
            (8, None, [], True),
            (9, None, [], True),
            # The explicit phases filter still applies to every phase.
            (1, [2], [1, 2, 3, 4], False),
            (9, [1], [1, 2, 3, 4], False),
            (9, [9], [], True),
        ],
    )
    def test_phase_enabled(self, phase, phases, applicable, expected):
        assert _phase_enabled(phase, phases, applicable) is expected

    def _stub_auto_config(self, monkeypatch, architectures):
        fake_cfg = SimpleNamespace(architectures=architectures)
        monkeypatch.setattr(
            main_benchmark,
            "AutoConfig",
            SimpleNamespace(from_pretrained=lambda *a, **k: fake_cfg),
        )

    @pytest.mark.parametrize("applicable", [[1, 4], [1, 2, 3, 4], []])
    def test_adapter_applicable_phases_resolves_fake_adapter(self, monkeypatch, applicable):
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        class _FakeAdapter:
            applicable_phases = applicable

        monkeypatch.setitem(SUPPORTED_ARCHITECTURES, "FakePhaseArch", _FakeAdapter)
        self._stub_auto_config(monkeypatch, ["FakePhaseArch"])
        assert _adapter_applicable_phases("fake/model") == applicable

    def test_adapter_applicable_phases_vision_is_p1_only(self, monkeypatch):
        # Vision encoders run P1 (HF parity on pixels); P9 is gated by
        # is_visual_model, not applicable_phases.
        self._stub_auto_config(monkeypatch, ["ViTModel"])
        assert _adapter_applicable_phases("fake/vit") == [1]

    def test_adapter_applicable_phases_unknown_arch_defaults(self, monkeypatch):
        self._stub_auto_config(monkeypatch, ["NotARealArchitecture"])
        assert _adapter_applicable_phases("fake/unknown") == [1, 2, 3, 4]

    def test_adapter_applicable_phases_config_error_defaults(self, monkeypatch):
        def _raise(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(main_benchmark, "AutoConfig", SimpleNamespace(from_pretrained=_raise))
        assert _adapter_applicable_phases("fake/offline") == [1, 2, 3, 4]


class _FakeBridge(TransformerBridge):
    """Subclass so beartype's isinstance(bridge, TransformerBridge) passes, while
    bypassing the heavy real __init__ — the benchmarks only touch cfg/adapter/
    processor and the forward/cache callables stubbed here."""

    def __init__(self, cfg, adapter=None, processor=None, forward_out=None, cache=None):
        self.cfg = cfg
        self.adapter = adapter
        self.processor = processor
        self._forward_out = forward_out
        self._cache = cache

    def __call__(self, *args, **kwargs):
        return self._forward_out

    def run_with_cache(self, *args, **kwargs):
        return self._forward_out, self._cache


def _fake_bridge(model_name="google/vit-base-patch16-224", **kwargs):
    cfg = SimpleNamespace(
        is_visual_model=True,
        is_multimodal=False,
        model_name=model_name,
        n_layers=2,
        d_model=8,
        device="cpu",
    )
    return _FakeBridge(cfg, **kwargs)


_PIXELS = torch.zeros(1, 3, 4, 4)


def test_forward_passes_on_finite_tensor():
    res = benchmark_vision_forward(_fake_bridge(forward_out=torch.randn(1, 5, 8)), _PIXELS)
    assert res.name == "vision_forward"
    assert res.passed is True


def test_forward_fails_on_nan_output():
    res = benchmark_vision_forward(
        _fake_bridge(forward_out=torch.full((1, 5, 8), float("nan"))), _PIXELS
    )
    assert res.passed is False
    assert res.severity == BenchmarkSeverity.DANGER


def test_forward_feeds_reference_pixel_values():
    seen = {}

    class _Ref(torch.nn.Module):
        def forward(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(logits=torch.ones(1, 5, 8))

    res = benchmark_vision_forward(
        _fake_bridge(forward_out=torch.ones(1, 5, 8)), _PIXELS, reference_model=_Ref()
    )
    assert "pixel_values" in seen
    assert res.passed is True


def test_cache_requires_declared_components_and_edge_blocks():
    hooks = {
        "embed.hook_out": torch.ones(1, 5, 8),
        "ln_final.hook_out": torch.ones(1, 5, 8),
        "unembed.hook_out": torch.ones(1, 10),
        "blocks.0.hook_out": torch.ones(1, 5, 8),
        "blocks.1.hook_out": torch.ones(1, 5, 8),
    }
    adapter = SimpleNamespace(component_mapping={"embed": 1, "ln_final": 1, "unembed": 1})
    res = benchmark_vision_cache(_fake_bridge(adapter=adapter, cache=hooks), _PIXELS)
    assert res.name == "vision_cache"
    assert res.passed is True
    assert res.details["critical_found"] == 5


def test_cache_reports_missing_classifier_hook():
    hooks = {
        "embed.hook_out": torch.ones(1, 5, 8),
        "blocks.0.hook_out": torch.ones(1, 5, 8),
        "blocks.1.hook_out": torch.ones(1, 5, 8),
    }
    adapter = SimpleNamespace(component_mapping={"embed": 1, "ln_final": 1, "unembed": 1})
    res = benchmark_vision_cache(_fake_bridge(adapter=adapter, cache=hooks), _PIXELS)
    assert res.severity == BenchmarkSeverity.WARNING
    assert "ln_final.hook_out" in res.details["missing"]
    assert "unembed.hook_out" in res.details["missing"]


def test_stability_skips_tiny_models():
    res = benchmark_vision_representation_stability(
        _fake_bridge(model_name="hf-internal-testing/tiny-random-vit"), _PIXELS
    )
    assert res.severity == BenchmarkSeverity.SKIPPED
    assert res.name == "vision_representation_stability"


def test_decode_skips_tiny_models_bare_encoders_and_missing_processor():
    res = benchmark_vision_classification_decode(
        _fake_bridge(model_name="hf-internal-testing/tiny-random-vit")
    )
    assert res.severity == BenchmarkSeverity.SKIPPED
    # Bare encoder: no classifier head declared in the component mapping.
    res2 = benchmark_vision_classification_decode(
        _fake_bridge(adapter=SimpleNamespace(component_mapping={"embed": 1}))
    )
    assert res2.severity == BenchmarkSeverity.SKIPPED
    assert "no classifier head" in res2.message
    # Classifier head declared but no image processor available.
    res3 = benchmark_vision_classification_decode(
        _fake_bridge(adapter=SimpleNamespace(component_mapping={"unembed": 1}), processor=None)
    )
    assert res3.severity == BenchmarkSeverity.SKIPPED
    assert "processor" in res3.message
