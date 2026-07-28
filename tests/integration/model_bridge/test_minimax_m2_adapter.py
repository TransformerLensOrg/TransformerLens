"""Integration tests for the MiniMax-M2 architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/minimax-m2"


@pytest.fixture(scope="module")
def m2_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(m2_bridge):
    torch.manual_seed(0)
    return torch.randint(0, m2_bridge.cfg.d_vocab - 10, (1, 8))


class TestMiniMaxM2BridgeCreation:
    def test_adapter_and_block_count(self, m2_bridge):
        from transformer_lens.model_bridge.supported_architectures.minimax_m2 import (
            MiniMaxM2ArchitectureAdapter,
        )

        assert isinstance(m2_bridge.adapter, MiniMaxM2ArchitectureAdapter)
        assert len(m2_bridge.blocks) == 2

    def test_gqa_config_propagated(self, m2_bridge):
        hf_config = m2_bridge.original_model.config
        assert m2_bridge.cfg.n_key_value_heads == hf_config.num_key_value_heads
        assert m2_bridge.cfg.d_head == hf_config.head_dim


class TestMiniMaxM2ForwardEquivalence:
    def test_forward_matches_hf(self, m2_bridge, sample_tokens):
        hf_model = m2_bridge.original_model
        with torch.no_grad():
            bridge_out = m2_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestMiniMaxM2HFDelegation:
    def test_components_are_shared_wrappers(self, m2_bridge):
        hf_model = m2_bridge.original_model
        assert m2_bridge.blocks[0].attn.q is hf_model.model.layers[0].self_attn.q_proj
        assert m2_bridge.blocks[0].mlp is hf_model.model.layers[0].mlp


class TestMiniMaxM2Hooks:
    def test_hook_shapes(self, m2_bridge, sample_tokens):
        d_model = m2_bridge.cfg.d_model
        n_heads = m2_bridge.original_model.config.num_attention_heads
        head_dim = m2_bridge.original_model.config.head_dim
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = [
            "blocks.0.attn.hook_out",
            "blocks.0.attn.q_norm.hook_out",
            "blocks.0.mlp.hook_out",
            "blocks.1.hook_out",
        ]
        with torch.no_grad():
            m2_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        assert captured["blocks.0.attn.hook_out"] == (1, seq, d_model)
        # Q-norm runs over the full projection width, pre-head-reshape.
        assert captured["blocks.0.attn.q_norm.hook_out"] == (1, seq, n_heads * head_dim)
        assert captured["blocks.0.mlp.hook_out"] == (1, seq, d_model)
        assert captured["blocks.1.hook_out"] == (1, seq, d_model)


class TestMiniMaxM2SigmoidRouter:
    def test_router_uses_correction_bias_buffer(self, m2_bridge):
        """DeepSeek-V3-style sigmoid routing with a trained bias buffer."""
        moe_block = m2_bridge.original_model.model.layers[0].mlp
        n_experts = m2_bridge.original_model.config.num_local_experts
        assert moe_block.e_score_correction_bias.shape == (n_experts,)
        assert moe_block.gate.weight.shape[0] == n_experts
