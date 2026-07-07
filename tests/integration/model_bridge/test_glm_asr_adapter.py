"""Integration tests for the GLM-ASR architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "zai-org/GLM-ASR-Nano-2512"


@pytest.fixture(scope="module")
def glmasr_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(glmasr_bridge):
    torch.manual_seed(0)
    return torch.randint(0, glmasr_bridge.cfg.d_vocab - 10, (1, 12))


class TestGlmAsrBridgeCreation:
    def test_adapter_selected(self, glmasr_bridge):
        from transformer_lens.model_bridge.supported_architectures.glm_asr import (
            GlmAsrArchitectureAdapter,
        )

        assert isinstance(glmasr_bridge.adapter, GlmAsrArchitectureAdapter)


class TestGlmAsrForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, glmasr_bridge, sample_tokens):
        from transformers import AutoModelForSeq2SeqLM

        fresh = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = glmasr_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestGlmAsrHooks:
    def test_hooks_fire(self, glmasr_bridge, sample_tokens):
        d_model = glmasr_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            glmasr_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestGlmAsrGeneration:
    def test_generate(self, glmasr_bridge):
        text = glmasr_bridge.generate(
            "The capital of France is", max_new_tokens=5, do_sample=False, verbose=False
        )
        assert isinstance(text, str)
        assert "Paris" in text
