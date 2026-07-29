"""Integration tests for the BitNet architecture adapter.

Uses the bf16 master-weight checkpoint (the packed 1.58-bit repo needs
custom kernels). CI-gated for download cost.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"

# BitNet 2.4B download/load — excluded from CI tiers via -m "not slow".
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(bridge):
    return bridge.tokenizer("The capital of France is", return_tensors="pt").input_ids


class TestBitNetBridgeCreation:
    def test_adapter_and_sub_norms(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.bitnet import (
            BitNetArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, BitNetArchitectureAdapter)
        hf_attn = bridge.blocks[0].attn.original_component
        assert hasattr(hf_attn, "attn_sub_norm")


class TestBitNetForwardEquivalence:
    def test_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestBitNetHooks:
    def test_hooks_fire(self, bridge, sample_tokens):
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out"]
        with torch.no_grad():
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        d_model = bridge.cfg.d_model
        assert captured["blocks.0.attn.hook_out"] == (1, seq, d_model)
        assert captured["blocks.0.mlp.hook_out"] == (1, seq, d_model)
