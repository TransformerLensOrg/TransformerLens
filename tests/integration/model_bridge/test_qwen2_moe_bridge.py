"""Integration tests for the Qwen2-MoE TransformerBridge."""

import copy

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def tiny_qwen2_moe_config() -> Qwen2MoeConfig:
    return Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


def tiny_qwen2_moe_bridge() -> tuple[TransformerBridge, Qwen2MoeForCausalLM]:
    torch.manual_seed(0)
    cfg = tiny_qwen2_moe_config()
    cfg._attn_implementation = "eager"
    hf_reference = Qwen2MoeForCausalLM(cfg).eval()
    hf_reference.config._attn_implementation = "eager"
    for layer in hf_reference.model.layers:
        layer.self_attn.config._attn_implementation = "eager"

    hf_model = Qwen2MoeForCausalLM(copy.deepcopy(cfg)).eval()
    hf_model.load_state_dict(hf_reference.state_dict())
    bridge = build_bridge_from_module(
        hf_model,
        "Qwen2MoeForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.adapter.setup_component_testing(hf_model, bridge_model=bridge)
    return bridge, hf_reference


def _tokens() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4, 5]])


class TestQwen2MoeBridge:
    def test_bridge_structure(self) -> None:
        bridge, _ = tiny_qwen2_moe_bridge()

        assert len(bridge.blocks) == 2
        assert isinstance(bridge.blocks[0].attn, PositionEmbeddingsAttentionBridge)
        assert isinstance(bridge.blocks[0].mlp, MoEBridge)
        assert isinstance(bridge.blocks[0].mlp.shared_expert, GatedMLPBridge)
        assert hasattr(bridge.blocks[0].mlp, "shared_expert_gate")

    def test_forward_matches_hf(self) -> None:
        bridge, hf_reference = tiny_qwen2_moe_bridge()
        tokens = _tokens()

        with torch.no_grad():
            bridge_out = bridge(tokens)
            hf_out = hf_reference(tokens).logits

        assert bridge_out.shape == (1, 5, 128)
        assert not torch.isnan(bridge_out).any()
        assert not torch.isinf(bridge_out).any()
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_run_with_cache_captures_moe_hooks(self) -> None:
        bridge, _ = tiny_qwen2_moe_bridge()

        _, cache = bridge.run_with_cache(_tokens())

        for layer_idx in range(len(bridge.blocks)):
            for hook_name in (
                "gate.hook_out",
                "experts.hook_out",
                "shared_expert.hook_out",
                "shared_expert_gate.hook_out",
                "hook_out",
            ):
                key = f"blocks.{layer_idx}.mlp.{hook_name}"
                assert key in cache, f"Missing cache key: {key}"
