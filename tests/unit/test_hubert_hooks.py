"""Tests for HookedAudioEncoder (HuBERT) hook and ablation functionality."""

import math

import numpy as np
import pytest
import torch

import transformer_lens.utils as utils
from transformer_lens import HookedAudioEncoder

SAMPLE_RATE = 16000
DURATION_S = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_output_tensor(out):
    """Extract tensor from model output (handles dict or tensor)."""
    if isinstance(out, torch.Tensor):
        return out
    try:
        return out["predictions"]
    except (KeyError, TypeError):
        return out


def make_sine(frequency=440.0, sr=SAMPLE_RATE, duration=DURATION_S, amplitude=0.1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return amplitude * np.sin(2 * math.pi * frequency * t)


@pytest.fixture(scope="module")
def audio_model():
    return HookedAudioEncoder.from_pretrained("facebook/hubert-base-ls960", device=DEVICE)


@pytest.fixture(scope="module")
def frames_and_mask(audio_model):
    wav = make_sine()
    frames, frame_mask = audio_model.to_frames(
        [wav], sampling_rate=SAMPLE_RATE, move_to_device=True
    )
    return frames, frame_mask


class TestHubertRunWithCache:
    def test_cache_contains_attention_pattern(self, audio_model, frames_and_mask):
        frames, frame_mask = frames_and_mask
        _, cache = audio_model.run_with_cache(
            frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True
        )
        layer = 0
        pattern_name = utils.get_act_name("pattern", layer)
        if pattern_name in cache:
            attn = cache[pattern_name]
        elif ("pattern", layer, "attn") in cache:
            attn = cache["pattern", layer, "attn"]
        else:
            pytest.fail(f"Attention pattern not found in cache. Keys: {list(cache.keys())[:10]}")

    def test_cache_attention_pattern_shape(self, audio_model, frames_and_mask):
        frames, frame_mask = frames_and_mask
        _, cache = audio_model.run_with_cache(
            frames, one_zero_attention_mask=frame_mask, remove_batch_dim=True
        )
        pattern_name = utils.get_act_name("pattern", 0)
        if pattern_name in cache:
            attn = cache[pattern_name]
        elif ("pattern", 0, "attn") in cache:
            attn = cache["pattern", 0, "attn"]
        else:
            pytest.fail(f"Attention pattern not found in cache. Keys: {list(cache.keys())[:10]}")
        # Should be (n_heads, seq, seq) or (seq, n_heads, seq)
        assert attn.ndim == 3, f"Expected 3D attention pattern, got {attn.ndim}D"


class TestHubertHeadAblation:
    def test_ablation_changes_output(self, audio_model, frames_and_mask):
        frames, frame_mask = frames_and_mask
        head_to_ablate = 0
        layer_to_ablate = 0
        v_act_name = utils.get_act_name("v", layer_to_ablate)

        def head_ablation_hook(value, hook):
            v = value.clone()
            if v.ndim == 4:
                v[:, :, head_to_ablate, :] = 0.0
            elif v.ndim == 3:
                v[:, head_to_ablate, :] = 0.0
            return v

        # Baseline
        baseline_out = audio_model.run_with_hooks(
            frames, fwd_hooks=[], one_zero_attention_mask=frame_mask
        )
        baseline_tensor = _get_output_tensor(baseline_out)

        # Ablated
        ablated_out = audio_model.run_with_hooks(
            frames,
            fwd_hooks=[(v_act_name, head_ablation_hook)],
            one_zero_attention_mask=frame_mask,
        )
        ablated_tensor = _get_output_tensor(ablated_out)

        # Outputs should differ after ablation
        assert not torch.allclose(
            baseline_tensor, ablated_tensor, atol=1e-6
        ), "Ablating a head had no effect on the output"
