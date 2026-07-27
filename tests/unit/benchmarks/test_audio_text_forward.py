"""Audio-text forward benchmark gating (download-free).

Audio-text decoders (Qwen2Audio, GLM-ASR, ...) declare is_multimodal but take
processed audio features, not pixel_values (Phase 7) or a raw waveform (Phase 8
encoder). benchmark_audio_text_forward covers that path; here we pin its
skip/gate logic without loading a model (the real finite-logits check runs in the
per-model verification).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from transformer_lens.benchmarks.audio import (  # noqa: E402
    _prepare_audio_text_inputs,
    benchmark_audio_text_forward,
)
from transformer_lens.benchmarks.utils import BenchmarkSeverity  # noqa: E402
from transformer_lens.model_bridge.bridge import TransformerBridge  # noqa: E402


class _FakeBridge(TransformerBridge):
    """Subclass so beartype's isinstance(bridge, TransformerBridge) passes, while
    bypassing the heavy real __init__ — the benchmark only touches cfg/processor."""

    def __init__(self, cfg, processor):
        self.cfg = cfg
        self.processor = processor


def _fake_bridge(is_multimodal=True, processor=None, model_name="some/audio-model"):
    cfg = SimpleNamespace(
        is_multimodal=is_multimodal,
        is_audio_model=False,
        model_name=model_name,
        d_vocab=100,
        device="cpu",
    )
    return _FakeBridge(cfg, processor)


def test_skips_non_multimodal():
    res = benchmark_audio_text_forward(_fake_bridge(is_multimodal=False))
    assert res.severity == BenchmarkSeverity.SKIPPED
    assert res.name == "audio_text_forward"


def test_skips_when_no_processor_or_audio_token():
    # No processor at all.
    res = benchmark_audio_text_forward(_fake_bridge(processor=None))
    assert res.severity == BenchmarkSeverity.SKIPPED
    # Processor present but no audio_token attribute.
    res2 = benchmark_audio_text_forward(_fake_bridge(processor=SimpleNamespace()))
    assert res2.severity == BenchmarkSeverity.SKIPPED


def test_prepare_returns_none_without_audio_token():
    ids, extra = _prepare_audio_text_inputs(_fake_bridge(processor=SimpleNamespace()))
    assert ids is None and extra is None


def test_tiny_model_is_info_pass_not_skipped():
    # Tiny/test models pass trivially (INFO) so Phase 8 isn't NULL for fixtures.
    res = benchmark_audio_text_forward(
        _fake_bridge(
            processor=SimpleNamespace(audio_token="<|AUDIO|>"),
            model_name="hf-internal-testing/tiny-random-qwen2audio",
        )
    )
    assert res.severity == BenchmarkSeverity.INFO
    assert res.passed is True
