"""Tests for the feature-extractor-driven audio input prep."""

from types import SimpleNamespace

import torch

from transformer_lens.benchmarks.audio import _prepare_audio_encoder_input


class _FakeFE:
    """Stands in for an AST-style feature extractor (spectrogram features, own rate)."""

    sampling_rate = 8000

    def __call__(self, waveforms, sampling_rate, return_tensors):
        assert sampling_rate == 8000
        batch = len(waveforms)
        return {"input_features": torch.ones(batch, 12, 4)}


class _Bridge(SimpleNamespace):
    pass


def _make_bridge(processor):
    return _Bridge(processor=processor, cfg=SimpleNamespace(device="cpu", dtype=torch.float32))


def test_uses_feature_extractor_output_and_rate():
    bridge = _make_bridge(_FakeFE())
    prepared = _prepare_audio_encoder_input(bridge)
    # feature-space output, not the raw waveform; synthetic waveform used the FE's rate
    assert prepared.shape == (1, 12, 4)
    assert torch.equal(prepared, torch.ones(1, 12, 4))


def test_unwraps_composite_processor():
    processor = SimpleNamespace(feature_extractor=_FakeFE())
    bridge = _make_bridge(processor)
    prepared = _prepare_audio_encoder_input(bridge)
    assert prepared.shape == (1, 12, 4)


def test_falls_back_to_raw_waveform_without_processor():
    bridge = _make_bridge(None)
    prepared = _prepare_audio_encoder_input(bridge)
    assert prepared.shape == (1, 16000)


def test_existing_waveform_passes_through_feature_extractor():
    bridge = _make_bridge(_FakeFE())
    waveform = torch.randn(2, 8000)
    prepared = _prepare_audio_encoder_input(bridge, waveform)
    assert prepared.shape == (2, 12, 4)
