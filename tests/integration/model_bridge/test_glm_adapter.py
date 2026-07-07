"""Integration tests for the dense GLM architecture adapter."""

import os

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "zai-org/glm-edge-1.5b-chat"

pytestmark = pytest.mark.skipif(
    bool(os.getenv("CI")), reason="glm-edge-1.5b download too large for CI budget"
)


@pytest.fixture(scope="module")
def glm_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(glm_bridge):
    return glm_bridge.tokenizer("The capital of France is", return_tensors="pt").input_ids


class TestGlmBridgeCreation:
    def test_adapter_selected(self, glm_bridge):
        from transformer_lens.model_bridge.supported_architectures.glm import (
            GlmArchitectureAdapter,
        )

        assert isinstance(glm_bridge.adapter, GlmArchitectureAdapter)


class TestGlmForwardEquivalence:
    def test_forward_matches_hf(self, glm_bridge, sample_tokens):
        hf_model = glm_bridge.original_model
        with torch.no_grad():
            bridge_out = glm_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestGlmHooks:
    def test_hooks_fire(self, glm_bridge, sample_tokens):
        d_model = glm_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out"]
        with torch.no_grad():
            glm_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"


class TestGlmGeneration:
    def test_greedy_generation_is_coherent(self, glm_bridge):
        out = glm_bridge.generate(
            "The capital of France is", max_new_tokens=5, do_sample=False, verbose=False
        )
        assert "Paris" in out
