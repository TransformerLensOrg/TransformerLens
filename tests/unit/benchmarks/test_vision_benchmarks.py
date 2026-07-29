"""Vision benchmark (Phase 9) gating and behavior tests (download-free).

Vision-only encoders (ViT/DeiT) have no text tower: text phases 1-4 can't run
against them, and HookedTransformer can't represent them. Phase 9 feeds
synthetic pixel_values through the bridge and checks parity/caching against the
wrapped HF model. These tests pin the classification plumbing, the phase gating
in run_benchmark_suite, and the benchmark verdicts themselves using fake
bridges (the real-checkpoint coverage lives in the ViT adapter integration
tests).
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
    benchmark_vision_forward,
)
from transformer_lens.model_bridge.bridge import TransformerBridge  # noqa: E402
from transformer_lens.utilities.architectures import (  # noqa: E402
    AUDIO_TEXT_ARCHITECTURES,
    NO_HT_COMPARISON_ARCHITECTURES,
    VISION_ARCHITECTURES,
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

    def test_vision_skips_ht_comparison(self):
        # No text tower — HookedTransformer cannot represent vision encoders.
        assert VISION_ARCHITECTURES <= NO_HT_COMPARISON_ARCHITECTURES

    def test_audio_text_skips_ht_comparison(self):
        # Seq2seq-loading audio-conditioned decoders have the same HT gap.
        assert AUDIO_TEXT_ARCHITECTURES <= NO_HT_COMPARISON_ARCHITECTURES

    def test_audio_text_still_classifies_causal(self):
        # NO_HT membership must not leak into classification (bridge semantics).
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

    def test_adapter_applicable_phases_vision_is_empty(self, monkeypatch):
        self._stub_auto_config(monkeypatch, ["ViTModel"])
        assert _adapter_applicable_phases("fake/vit") == []

    def test_adapter_applicable_phases_unknown_arch_defaults(self, monkeypatch):
        self._stub_auto_config(monkeypatch, ["NotARealArchitecture"])
        assert _adapter_applicable_phases("fake/unknown") == [1, 2, 3, 4]

    def test_adapter_applicable_phases_config_error_defaults(self, monkeypatch):
        def _raise(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr(main_benchmark, "AutoConfig", SimpleNamespace(from_pretrained=_raise))
        assert _adapter_applicable_phases("fake/offline") == [1, 2, 3, 4]


class _FakeHFVisionModel(torch.nn.Module):
    """Stands in for the wrapped HF model (config + forward). Must be a real
    nn.Module — the bridge's original_model property is annotated to return one."""

    def __init__(self, output, image_size=8, num_channels=3):
        super().__init__()
        self.config = SimpleNamespace(image_size=image_size, num_channels=num_channels)
        self._output = output

    def forward(self, pixel_values):
        return self._output


class _FakeVisionBridge(TransformerBridge):
    """Subclass so beartype's isinstance(bridge, TransformerBridge) passes, while
    bypassing the heavy real __init__ — the benchmarks only touch cfg,
    original_model (via _driver), forward, and run_with_cache."""

    def __init__(self, cfg, hf_model, forward_out=None, cache=None, with_unembed=False):
        self.cfg = cfg
        # original_model is a property reading self._driver.underlying_model.
        self._driver = SimpleNamespace(underlying_model=hf_model)
        self._forward_out = forward_out
        self._cache = cache
        if with_unembed:
            self.unembed = SimpleNamespace()

    def forward(self, *args, **kwargs):
        return self._forward_out

    def run_with_cache(self, *args, **kwargs):
        return self._forward_out, self._cache


def _vision_cfg(is_visual_model=True):
    return SimpleNamespace(
        is_visual_model=is_visual_model,
        model_name="fake/vit",
        device="cpu",
        dtype=torch.float32,
    )


def _classifier_logits():
    return torch.randn(1, 10)


class TestBenchmarkVisionForward:
    def test_skips_non_vision(self):
        bridge = _FakeVisionBridge(_vision_cfg(is_visual_model=False), _FakeHFVisionModel(None))
        res = benchmark_vision_forward(bridge)
        assert res.severity == BenchmarkSeverity.SKIPPED
        assert res.name == "vision_forward"

    def test_parity_pass_classifier_logits(self):
        logits = _classifier_logits()
        hf = _FakeHFVisionModel(SimpleNamespace(logits=logits))
        bridge = _FakeVisionBridge(_vision_cfg(), hf, forward_out=logits)
        res = benchmark_vision_forward(bridge)
        assert res.passed is True
        assert res.severity == BenchmarkSeverity.INFO
        assert res.details["max_diff"] == 0.0

    def test_parity_pass_bare_encoder_last_hidden_state(self):
        hidden = torch.randn(1, 5, 4)
        hf = _FakeHFVisionModel(SimpleNamespace(last_hidden_state=hidden))
        bridge = _FakeVisionBridge(_vision_cfg(), hf, forward_out=hidden)
        res = benchmark_vision_forward(bridge)
        assert res.passed is True

    def test_parity_mismatch_fails(self):
        logits = _classifier_logits()
        hf = _FakeHFVisionModel(SimpleNamespace(logits=logits))
        bridge = _FakeVisionBridge(_vision_cfg(), hf, forward_out=logits + 1.0)
        res = benchmark_vision_forward(bridge)
        assert res.passed is False
        assert res.severity == BenchmarkSeverity.DANGER

    def test_nan_output_fails(self):
        logits = _classifier_logits()
        hf = _FakeHFVisionModel(SimpleNamespace(logits=logits))
        bad = logits.clone()
        bad[0, 0] = float("nan")
        bridge = _FakeVisionBridge(_vision_cfg(), hf, forward_out=bad)
        res = benchmark_vision_forward(bridge)
        assert res.passed is False
        assert "NaN" in res.message

    def test_shape_mismatch_fails(self):
        hf = _FakeHFVisionModel(SimpleNamespace(logits=torch.randn(1, 7)))
        bridge = _FakeVisionBridge(_vision_cfg(), hf, forward_out=torch.randn(1, 5))
        res = benchmark_vision_forward(bridge)
        assert res.passed is False
        assert "shape" in res.message


def _full_cache(final_key):
    t = torch.randn(1, 5, 4)
    return {
        "embed.hook_in": torch.randn(1, 3, 8, 8),
        "blocks.0.attn.q.hook_out": t,
        "blocks.0.attn.o.hook_out": t,
        final_key: t,
    }


class TestBenchmarkVisionCache:
    def test_skips_non_vision(self):
        bridge = _FakeVisionBridge(_vision_cfg(is_visual_model=False), _FakeHFVisionModel(None))
        res = benchmark_vision_cache(bridge)
        assert res.severity == BenchmarkSeverity.SKIPPED
        assert res.name == "vision_cache"

    def test_classifier_requires_unembed_hook(self):
        bridge = _FakeVisionBridge(
            _vision_cfg(),
            _FakeHFVisionModel(None),
            cache=_full_cache("unembed.hook_out"),
            with_unembed=True,
        )
        res = benchmark_vision_cache(bridge)
        assert res.passed is True
        assert "unembed.hook_out" in res.details["checked_hooks"]

    def test_bare_encoder_requires_ln_final_hook(self):
        bridge = _FakeVisionBridge(
            _vision_cfg(),
            _FakeHFVisionModel(None),
            cache=_full_cache("ln_final.hook_normalized"),
        )
        res = benchmark_vision_cache(bridge)
        assert res.passed is True
        assert "ln_final.hook_normalized" in res.details["checked_hooks"]

    def test_missing_hook_fails(self):
        cache = _full_cache("ln_final.hook_normalized")
        del cache["blocks.0.attn.q.hook_out"]
        bridge = _FakeVisionBridge(_vision_cfg(), _FakeHFVisionModel(None), cache=cache)
        res = benchmark_vision_cache(bridge)
        assert res.passed is False
        assert "blocks.0.attn.q.hook_out" in res.message

    def test_non_finite_activation_fails(self):
        cache = _full_cache("ln_final.hook_normalized")
        cache["embed.hook_in"] = torch.full((1, 3, 8, 8), float("inf"))
        bridge = _FakeVisionBridge(_vision_cfg(), _FakeHFVisionModel(None), cache=cache)
        res = benchmark_vision_cache(bridge)
        assert res.passed is False
        assert "NaN/Inf" in res.message

    def test_empty_cache_fails(self):
        bridge = _FakeVisionBridge(_vision_cfg(), _FakeHFVisionModel(None), cache={})
        res = benchmark_vision_cache(bridge)
        assert res.passed is False
        assert "empty cache" in res.message
