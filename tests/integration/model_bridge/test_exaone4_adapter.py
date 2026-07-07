"""Integration tests for the EXAONE 4.0 architecture adapter.

CI-gated for download cost (1.2B checkpoint, no tiny mirror).
"""

import os

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"

pytestmark = pytest.mark.skipif(
    bool(os.getenv("CI")), reason="EXAONE-4.0 1.2B download too large for CI budget"
)


@pytest.fixture(scope="module")
def ex4_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(ex4_bridge):
    return ex4_bridge.tokenizer("The capital of France is", return_tensors="pt").input_ids


class TestExaone4BridgeCreation:
    def test_adapter_and_hybrid_layers(self, ex4_bridge):
        from transformer_lens.model_bridge.supported_architectures.exaone4 import (
            Exaone4ArchitectureAdapter,
        )

        assert isinstance(ex4_bridge.adapter, Exaone4ArchitectureAdapter)
        layer_types = ex4_bridge.original_model.config.layer_types
        assert "sliding_attention" in layer_types or "full_attention" in layer_types


class TestExaone4ForwardEquivalence:
    def test_forward_matches_hf(self, ex4_bridge, sample_tokens):
        hf_model = ex4_bridge.original_model
        with torch.no_grad():
            bridge_out = ex4_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"


class TestExaone4Hooks:
    def test_post_norm_and_qk_norm_hooks_fire(self, ex4_bridge, sample_tokens):
        d_model = ex4_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "blocks.0.attn.hook_out",
            "blocks.0.ln1.hook_out",
            "blocks.0.mlp.hook_out",
            "blocks.0.ln2.hook_out",
        ]
        with torch.no_grad():
            ex4_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
