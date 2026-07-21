"""Integration tests for the Qwen2-Audio architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "trl-internal-testing/tiny-Qwen2AudioForConditionalGeneration"


@pytest.fixture(scope="module")
def audio_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(audio_bridge):
    torch.manual_seed(0)
    return torch.randint(0, audio_bridge.cfg.d_vocab - 10, (1, 8))


class TestQwen2AudioBridgeCreation:
    def test_adapter_and_components(self, audio_bridge):
        from transformer_lens.model_bridge.supported_architectures.qwen2_audio import (
            Qwen2AudioArchitectureAdapter,
        )

        assert isinstance(audio_bridge.adapter, Qwen2AudioArchitectureAdapter)
        assert hasattr(audio_bridge, "audio_encoder")
        assert hasattr(audio_bridge, "audio_projector")

    def test_audio_tower_is_live(self, audio_bridge):
        hf_model = audio_bridge.original_model
        assert audio_bridge.audio_encoder is hf_model.model.audio_tower
        assert audio_bridge.audio_projector is hf_model.model.multi_modal_projector


class TestQwen2AudioForwardEquivalence:
    def test_text_forward_matches_hf(self, audio_bridge, sample_tokens):
        hf_model = audio_bridge.original_model
        with torch.no_grad():
            bridge_out = audio_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestQwen2AudioHooks:
    def test_text_hooks_fire(self, audio_bridge, sample_tokens):
        d_model = audio_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.1.hook_out"]
        with torch.no_grad():
            audio_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
