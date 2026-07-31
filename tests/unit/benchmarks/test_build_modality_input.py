"""``build_modality_input`` must shape its tensor from the architecture, not a default.

Phase 1 and Phase 8 previously hardcoded a 16 kHz waveform, which is only correct for
waveform encoders (HuBERT, wav2vec2). Spectrogram encoders (AST) take
``[batch, max_length, num_mel_bins]`` and vision encoders take
``[batch, num_channels, image_size, image_size]``, so both failed verification with an
``IndexError`` before reaching any real comparison.
"""

from types import SimpleNamespace

import torch

from transformer_lens.benchmarks.utils import build_modality_input


def _bridge(hf_config=None, **cfg_flags):
    """Minimal bridge stand-in: build_modality_input only reads cfg flags and the HF config."""
    return SimpleNamespace(
        cfg=SimpleNamespace(**cfg_flags),
        original_model=SimpleNamespace(config=hf_config) if hf_config is not None else None,
    )


def test_spectrogram_audio_uses_config_dimensions() -> None:
    """AST-style encoders get [batch, max_length, num_mel_bins] from the HF config."""
    bridge = _bridge(
        hf_config=SimpleNamespace(num_mel_bins=128, max_length=1024),
        is_audio_model=True,
    )
    got = build_modality_input(bridge)
    assert got is not None
    assert got.shape == (1, 1024, 128)


def test_spectrogram_dimensions_are_not_hardcoded() -> None:
    """A checkpoint with non-default mel/length dimensions is honoured."""
    bridge = _bridge(
        hf_config=SimpleNamespace(num_mel_bins=64, max_length=512),
        is_audio_model=True,
    )
    got = build_modality_input(bridge, batch_size=3)
    assert got is not None
    assert got.shape == (3, 512, 64)


def test_waveform_audio_without_mel_bins() -> None:
    """num_mel_bins is what separates a spectrogram encoder from a waveform one."""
    bridge = _bridge(hf_config=SimpleNamespace(), is_audio_model=True)
    got = build_modality_input(bridge)
    assert got is not None
    assert got.shape == (1, 16000)


def test_vision_uses_config_image_size() -> None:
    """Vision encoders get pixel-shaped input sized from the HF config."""
    bridge = _bridge(
        hf_config=SimpleNamespace(image_size=224, num_channels=3),
        is_visual_model=True,
    )
    got = build_modality_input(bridge)
    assert got is not None
    assert got.shape == (1, 3, 224, 224)


def test_vision_dimensions_are_not_hardcoded() -> None:
    """A non-224 checkpoint is honoured rather than forced to the default."""
    bridge = _bridge(
        hf_config=SimpleNamespace(image_size=384, num_channels=1),
        is_visual_model=True,
    )
    got = build_modality_input(bridge)
    assert got is not None
    assert got.shape == (1, 1, 384, 384)


def test_text_model_returns_none() -> None:
    """Text models keep the token-id path — the caller falls back to to_tokens()."""
    bridge = _bridge(hf_config=SimpleNamespace(vocab_size=50257))
    assert build_modality_input(bridge) is None


def test_dtype_and_device_are_applied() -> None:
    """Callers pass the benchmark's dtype so the input matches the model's precision."""
    bridge = _bridge(
        hf_config=SimpleNamespace(num_mel_bins=128, max_length=1024),
        is_audio_model=True,
    )
    got = build_modality_input(bridge, device="cpu", dtype=torch.float64)
    assert got is not None
    assert got.dtype == torch.float64
