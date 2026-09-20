"""facebook/wav2vec2-base through the bridge — deletion evidence for
HookedAudioEncoder's wav2vec2 support (registry ships the checkpoints)."""

from __future__ import annotations

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "facebook/wav2vec2-base"


@pytest.fixture(scope="module")
def w2v_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


def test_pretraining_checkpoint_boots_as_encoder(w2v_bridge):
    """The checkpoint declares Wav2Vec2ForPreTraining; the bridge loads its encoder."""
    assert type(w2v_bridge.original_model).__name__ == "Wav2Vec2Model"
    assert w2v_bridge.cfg.is_audio_model


def test_forward_matches_hf(w2v_bridge):
    from transformers import Wav2Vec2Model

    hf = Wav2Vec2Model.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    torch.manual_seed(0)
    wave = torch.randn(1, 16000) * 0.1
    with torch.no_grad():
        ref = hf(wave).last_hidden_state
        out = w2v_bridge(wave)
    out_t = out if isinstance(out, torch.Tensor) else out["last_hidden_state"]
    torch.testing.assert_close(out_t, ref, atol=1e-4, rtol=1e-4)


def test_masked_frame_entry_matches_hf(w2v_bridge):
    """encoder_output honors the padding mask on wav2vec2 exactly as on HuBERT."""
    hf = w2v_bridge.original_model
    torch.manual_seed(0)
    wave = torch.randn(1, 16000) * 0.1
    padded = torch.cat([wave, torch.zeros(1, 4000)], dim=1)
    sample_mask = torch.cat([torch.ones(1, 16000), torch.zeros(1, 4000)], dim=1).long()
    with torch.no_grad():
        ref = hf(padded, attention_mask=sample_mask).last_hidden_state
        feats = hf.feature_extractor(padded).transpose(1, 2)
        frames, _ = hf.feature_projection(feats)
        frame_mask = hf._get_feature_vector_attention_mask(frames.shape[1], sample_mask)
        out = w2v_bridge.encoder_output(frames, one_zero_attention_mask=frame_mask.long())
    real = frame_mask[0].bool()
    assert not real.all()
    torch.testing.assert_close(out[:, real], ref[:, real], atol=1e-4, rtol=1e-4)
