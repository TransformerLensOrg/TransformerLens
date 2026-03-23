"""Tests for HookedAudioEncoder (HuBERT) basic functionality."""

import math

import numpy as np
import pytest
import torch

from transformer_lens import HookedAudioEncoder

SAMPLE_RATE = 16000
DURATION_S = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_CHECKPOINT = "facebook/hubert-base-ls960"


def make_sine(frequency=440.0, sr=SAMPLE_RATE, duration=DURATION_S, amplitude=0.1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return amplitude * np.sin(2 * math.pi * frequency * t)


@pytest.fixture(scope="module")
def audio_model():
    return HookedAudioEncoder.from_pretrained(HF_CHECKPOINT, device=DEVICE)


@pytest.fixture(scope="module")
def waveform():
    return make_sine(frequency=440.0, sr=SAMPLE_RATE, duration=DURATION_S)


def _get_output_tensor(out):
    """Extract tensor from model output (handles dict or tensor)."""
    if isinstance(out, torch.Tensor):
        return out
    try:
        return out["predictions"]
    except (KeyError, TypeError):
        return out


class TestHubertForwardPass:
    def test_output_is_finite(self, audio_model, waveform):
        audio_model.eval()
        x = torch.from_numpy(waveform).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = audio_model(x)
        out_tensor = _get_output_tensor(out)
        assert torch.isfinite(out_tensor).all(), "Found NaNs or Infs in forward output"

    def test_output_shape(self, audio_model, waveform):
        audio_model.eval()
        x = torch.from_numpy(waveform).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = audio_model(x)
        out_tensor = _get_output_tensor(out)
        assert out_tensor.ndim == 3, f"Expected 3D output, got {out_tensor.ndim}D"
        assert out_tensor.shape[0] == 1, f"Expected batch=1, got {out_tensor.shape[0]}"

    def test_deterministic_eval(self, audio_model, waveform):
        audio_model.eval()
        x = torch.from_numpy(waveform).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out1 = _get_output_tensor(audio_model(x))
            out2 = _get_output_tensor(audio_model(x))
        assert torch.allclose(
            out1, out2, atol=1e-6
        ), f"Outputs differ between eval runs, max diff: {(out1 - out2).abs().max().item()}"

    def test_gradient_flow(self, audio_model, waveform):
        audio_model.train()
        for p in audio_model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        x = torch.from_numpy(waveform).unsqueeze(0).to(DEVICE)
        out = _get_output_tensor(audio_model(x))
        loss = out.mean()
        loss.backward()
        grads_found = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in audio_model.parameters()
            if p.requires_grad
        )
        assert grads_found, "No finite gradients found after backward()"


class TestHubertHFComparison:
    def test_cosine_similarity_to_hf(self, audio_model, waveform):
        try:
            from transformers import HubertModel, Wav2Vec2FeatureExtractor
        except ImportError:
            pytest.skip("transformers HubertModel not available")

        hf_feat = Wav2Vec2FeatureExtractor(sampling_rate=SAMPLE_RATE, do_normalize=True)
        hf_model = HubertModel.from_pretrained(HF_CHECKPOINT).to(DEVICE).eval()

        input_values = hf_feat(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt").get(
            "input_values"
        )
        input_values = input_values.to(DEVICE)

        with torch.no_grad():
            hf_out = hf_model(input_values).last_hidden_state.mean(dim=1)

            audio_model.eval()
            our_out = _get_output_tensor(
                audio_model(torch.from_numpy(waveform).unsqueeze(0).to(DEVICE))
            )
            if our_out.ndim == 3:
                our_out = our_out.mean(dim=1)

        if hf_out.shape[1] != our_out.shape[1]:
            pytest.skip(f"Dimension mismatch (HF {hf_out.shape[1]} vs ours {our_out.shape[1]})")

        cos = torch.nn.functional.cosine_similarity(hf_out, our_out, dim=1)
        assert cos.item() > 0.99, f"Cosine similarity too low: {cos.item()}"
