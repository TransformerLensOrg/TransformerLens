"""Integration tests for the ERNIE 4.5 MoE architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "yujiepan/ernie-4.5-moe-tiny-random"


@pytest.fixture(scope="module")
def moe_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(moe_bridge):
    torch.manual_seed(0)
    return torch.randint(0, moe_bridge.cfg.d_vocab - 10, (1, 8))


class TestErnie4_5MoeBridgeCreation:
    def test_adapter_selected(self, moe_bridge):
        from transformer_lens.model_bridge.supported_architectures.ernie4_5_moe import (
            Ernie4_5_MoeArchitectureAdapter,
        )

        assert isinstance(moe_bridge.adapter, Ernie4_5_MoeArchitectureAdapter)


class TestErnie4_5MoeForwardEquivalence:
    def test_forward_matches_hf(self, moe_bridge, sample_tokens):
        hf_model = moe_bridge.original_model
        with torch.no_grad():
            bridge_out = moe_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestErnie4_5MoeHooks:
    def test_hooks_fire(self, moe_bridge, sample_tokens):
        d_model = moe_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.1.mlp.hook_out"]
        with torch.no_grad():
            moe_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
