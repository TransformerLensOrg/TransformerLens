"""Integration tests for the Gemma 4 text-only architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "veyra-ai/Kairo-5M-Gemma4-Base"


@pytest.fixture(scope="module")
def gemma4_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(gemma4_bridge):
    torch.manual_seed(0)
    return torch.randint(0, gemma4_bridge.cfg.d_vocab - 10, (1, 12))


class TestGemma4TextBridgeCreation:
    def test_adapter_selected(self, gemma4_bridge):
        from transformer_lens.model_bridge.supported_architectures.gemma4_text import (
            Gemma4TextArchitectureAdapter,
        )

        assert isinstance(gemma4_bridge.adapter, Gemma4TextArchitectureAdapter)


class TestGemma4TextForwardEquivalence:
    def test_forward_matches_fresh_hf(self, gemma4_bridge, sample_tokens):
        from transformers import AutoModelForCausalLM

        fresh = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = gemma4_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestGemma4TextHooks:
    def test_hooks_fire(self, gemma4_bridge, sample_tokens):
        d_model = gemma4_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
            "blocks.0.ln1_post.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            gemma4_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestGemma4TextGeneration:
    def test_generate(self, gemma4_bridge):
        text = gemma4_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
