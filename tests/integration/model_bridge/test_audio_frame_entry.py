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


def test_masked_frame_entry_matches_hf_encoder(audio_bridge, waveform):
    """Under a padding mask, frame entry must match HF's encoder on real frames.

    HF zeroes pad frames before pos_conv_embed; without that, the kernel-128
    conv smears pad content into real frames (~62% relative error here).
    Frames come from an unmasked pass — HF mutates hidden_states in place, so
    masked-run frames are already pre-zeroed and would make this comparison
    self-fulfilling. "Changes the output" is not asserted anywhere: a wrong
    mask also changes the output.
    """
    hf = audio_bridge.original_model
    padded = torch.cat([waveform, torch.zeros(1, 4000)], dim=1)
    sample_mask = torch.cat([torch.ones_like(waveform), torch.zeros(1, 4000)], dim=1).long()

    with torch.no_grad():
        ref = hf(padded, attention_mask=sample_mask).last_hidden_state
        feats = hf.feature_extractor(padded).transpose(1, 2)
        frames = hf.feature_projection(feats)
        frame_mask = hf._get_feature_vector_attention_mask(frames.shape[1], sample_mask)
        out = audio_bridge.encoder_output(frames, one_zero_attention_mask=frame_mask.long())

    real = frame_mask[0].bool()
    assert not real.all(), "padding produced no masked frames; test setup is broken"
    torch.testing.assert_close(out[:, real], ref[:, real], atol=1e-4, rtol=1e-4)


def test_masked_frame_entry_leaves_caller_frames_untouched(audio_bridge):
    """masked_fill, not HF's in-place write: the caller's tensor survives."""
    frames = torch.randn(1, 8, audio_bridge.cfg.d_model)
    keep = frames.clone()
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    with torch.no_grad():
        audio_bridge.encoder_output(frames, one_zero_attention_mask=mask)
    assert torch.equal(frames, keep)


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
