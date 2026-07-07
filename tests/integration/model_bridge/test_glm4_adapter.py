"""Integration tests for the GLM-4-0414 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/glm-4"


@pytest.fixture(scope="module")
def glm4_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(glm4_bridge):
    torch.manual_seed(0)
    return torch.randint(0, glm4_bridge.cfg.d_vocab - 10, (1, 12))


class TestGlm4BridgeCreation:
    def test_adapter_selected(self, glm4_bridge):
        from transformer_lens.model_bridge.supported_architectures.glm4 import (
            Glm4ArchitectureAdapter,
        )

        assert isinstance(glm4_bridge.adapter, Glm4ArchitectureAdapter)


class TestGlm4ForwardEquivalence:
    def test_forward_matches_fresh_hf(self, glm4_bridge, sample_tokens):
        from transformers import AutoModelForCausalLM

        fresh = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = glm4_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestGlm4Hooks:
    def test_hooks_fire_including_sandwich_norms(self, glm4_bridge, sample_tokens):
        d_model = glm4_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
            "blocks.0.ln1_post.hook_out": (1, seq, d_model),
            "blocks.0.ln2_post.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            glm4_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in expected])
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"

    def test_sandwich_norm_edit_propagates(self, glm4_bridge, sample_tokens):
        with torch.no_grad():
            baseline = glm4_bridge(sample_tokens)
            edited = glm4_bridge.run_with_hooks(
                sample_tokens,
                fwd_hooks=[("blocks.0.ln1_post.hook_out", lambda t, hook: torch.zeros_like(t))],
            )
        assert not torch.allclose(edited, baseline)


class TestGlm4Generation:
    def test_generate(self, glm4_bridge):
        text = glm4_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
