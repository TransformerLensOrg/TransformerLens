"""Audio frame entry: run the encoder from precomputed frames.

Mirrors HookedAudioEncoder.encoder_output, the audio-path analogue of
start_at_layer. Deletion evidence for that method: without it the bridge can
only enter at the waveform, so injecting frames means re-running the conv front
end. start_at_layer stays refused for audio — this is a separate entry point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "facebook/hubert-base-ls960"
SAMPLE_RATE = 16000
FRAMES_HOOK = "feat_proj.hook_out"


@pytest.fixture(scope="module")
def audio_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.fixture(scope="module")
def waveform() -> torch.Tensor:
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)
    return torch.tensor(0.1 * np.sin(2 * math.pi * 440.0 * t))[None, :]


@pytest.fixture(scope="module")
def full_run(audio_bridge, waveform):
    last = f"blocks.{audio_bridge.cfg.n_layers - 1}.hook_out"
    _, cache = audio_bridge.run_with_cache(
        waveform, names_filter=[FRAMES_HOOK, "blocks.0.hook_out", last]
    )
    return cache, last


def test_frame_entry_matches_the_full_waveform_run(audio_bridge, full_run):
    """Re-entering at the frames reproduces the encoder output exactly."""
    cache, last = full_run
    resid = audio_bridge.encoder_output(cache[FRAMES_HOOK])
    torch.testing.assert_close(resid, cache[last], atol=0.0, rtol=0.0)


def test_hooks_fire_from_frame_entry(audio_bridge, full_run):
    """Block hooks fire on the frame path, so caching composes with it."""
    cache, last = full_run
    wanted = {"blocks.0.hook_out", last}
    cached, fwd_hooks, _ = audio_bridge.get_caching_hooks(names_filter=lambda name: name in wanted)
    with audio_bridge.hooks(fwd_hooks=fwd_hooks):
        audio_bridge.encoder_output(cache[FRAMES_HOOK])

    assert set(cached) == wanted
    for name in wanted:
        torch.testing.assert_close(cached[name], cache[name], atol=0.0, rtol=0.0)


def test_padding_mask_changes_the_encoding(audio_bridge, full_run):
    """The mask is applied, not ignored."""
    cache, last = full_run
    frames = cache[FRAMES_HOOK]
    mask = torch.ones(frames.shape[:2], dtype=torch.long)
    mask[:, -10:] = 0

    masked = audio_bridge.encoder_output(frames, one_zero_attention_mask=mask)
    assert not torch.allclose(masked, cache[last])


def test_waveform_shaped_input_is_rejected(audio_bridge, waveform):
    """A 2D waveform is not frames; say so instead of silently mis-running."""
    with pytest.raises(ValueError, match=r"\[batch, frames, d_model\]"):
        audio_bridge.encoder_output(waveform)


def test_start_at_layer_remains_refused_for_audio(audio_bridge, waveform):
    """Frame entry is a separate API; the residual-injection guard is untouched."""
    with pytest.raises(NotImplementedError, match="audio models"):
        audio_bridge(waveform, start_at_layer=1)


def test_text_models_reject_the_audio_frame_entry():
    """Non-audio bridges have no conv-frame stage to re-enter."""
    bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
    with pytest.raises(NotImplementedError, match="not an audio model"):
        bridge.encoder_output(torch.zeros(1, 4, bridge.cfg.d_model))


def test_spectrogram_encoders_reject_the_frame_entry():
    """AST has no conv feature extractor, so there is no frame stage to bypass."""
    bridge = TransformerBridge.boot_transformers(
        "MIT/ast-finetuned-audioset-10-10-0.4593", device="cpu"
    )
    with pytest.raises(NotImplementedError, match="convolutional front end"):
        bridge.encoder_output(torch.zeros(1, 4, bridge.cfg.d_model))
