"""Integration tests for the StarCoder2 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "optimum-intel-internal-testing/tiny-random-Starcoder2ForCausalLM"


@pytest.fixture(scope="module")
def sc2_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(sc2_bridge):
    torch.manual_seed(0)
    return torch.randint(0, sc2_bridge.cfg.d_vocab - 10, (1, 8))


class TestStarcoder2BridgeCreation:
    def test_adapter_selected(self, sc2_bridge):
        from transformer_lens.model_bridge.supported_architectures.starcoder2 import (
            Starcoder2ArchitectureAdapter,
        )

        assert isinstance(sc2_bridge.adapter, Starcoder2ArchitectureAdapter)

    def test_biased_projections_delegate(self, sc2_bridge):
        hf_config = sc2_bridge.original_model.config
        q = sc2_bridge.blocks[0].attn.q.original_component
        assert (q.bias is not None) == bool(getattr(hf_config, "use_bias", False))


class TestStarcoder2ForwardEquivalence:
    def test_forward_matches_hf(self, sc2_bridge, sample_tokens):
        hf_model = sc2_bridge.original_model
        with torch.no_grad():
            bridge_out = sc2_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestStarcoder2Hooks:
    def test_hooks_fire(self, sc2_bridge, sample_tokens):
        d_model = sc2_bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out", "blocks.1.hook_out"]
        with torch.no_grad():
            sc2_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"
