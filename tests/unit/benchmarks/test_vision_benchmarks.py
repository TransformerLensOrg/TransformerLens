"""Vision benchmark (Phase 9) gating and wiring, download-free.

The real pixel forward/cache runs happen in per-model verification; here we pin
the skip/gate logic and the shared encoder-benchmark wiring (result names, HF
reference kwarg, critical-hook selection) without loading a model.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from transformer_lens.benchmarks.utils import BenchmarkSeverity  # noqa: E402
from transformer_lens.benchmarks.vision import (  # noqa: E402
    benchmark_vision_cache,
    benchmark_vision_classification_decode,
    benchmark_vision_forward,
    benchmark_vision_representation_stability,
)
from transformer_lens.model_bridge.bridge import TransformerBridge  # noqa: E402


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
