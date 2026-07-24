"""Integration tests for the GLM-MoE-DSA architecture adapter."""

import copy

import torch
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def tiny_glm_moe_dsa_config() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=128,
        max_position_embeddings=32,
        index_topk=4,
        index_head_dim=16,
        index_n_heads=2,
        mlp_layer_types=["dense", "sparse", "sparse"],
        indexer_types=["full", "shared", "full"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )


def tiny_glm_moe_dsa_bridge() -> tuple[TransformerBridge, GlmMoeDsaForCausalLM]:
    torch.manual_seed(0)
    cfg = tiny_glm_moe_dsa_config()
    hf_reference = GlmMoeDsaForCausalLM(cfg).eval()
    hf_reference.config._attn_implementation = "eager"
    for layer in hf_reference.model.layers:
        layer.self_attn.config._attn_implementation = "eager"
    hf_model = GlmMoeDsaForCausalLM(copy.deepcopy(cfg)).eval()
    hf_model.load_state_dict(hf_reference.state_dict())
    bridge = build_bridge_from_module(
        hf_model,
        "GlmMoeDsaForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    bridge.adapter.setup_component_testing(hf_model, bridge_model=bridge)
    return bridge, hf_reference


class TestGlmMoeDsaBridge:
    def test_bridge_structure(self) -> None:
        bridge, _ = tiny_glm_moe_dsa_bridge()

        assert len(bridge.blocks) == 3
        assert hasattr(bridge, "embed")
        assert hasattr(bridge, "unembed")
        assert hasattr(bridge, "ln_final")
        # final_rms -> the final norm is selected as an RMSNorm bridge
        assert isinstance(bridge.ln_final, RMSNormalizationBridge)
        # positional_embedding_type == "rotary" -> a rotary embedding bridge is
        # installed and exposes cos/sin hooks the attention consumes
        assert isinstance(bridge.rotary_emb, RotaryEmbeddingBridge)
        assert hasattr(bridge.rotary_emb, "hook_cos")
        assert hasattr(bridge.rotary_emb, "hook_sin")

    def test_forward_matches_hf(self) -> None:
        bridge, hf_reference = tiny_glm_moe_dsa_bridge()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        with torch.no_grad():
            bridge_out = bridge(tokens)
            hf_out = hf_reference(tokens).logits

        assert bridge_out.shape == (1, 8, 128)
        assert not torch.isnan(bridge_out).any()
        assert not torch.isinf(bridge_out).any()
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"

    def test_run_with_cache_captures_mla_and_dsa_hooks(self) -> None:
        bridge, _ = tiny_glm_moe_dsa_bridge()
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        _, cache = bridge.run_with_cache(tokens)

        assert "blocks.0.attn.hook_q_latent" in cache
        assert "blocks.0.attn.hook_kv_latent" in cache
        assert "blocks.0.attn.hook_topk_indices" in cache
        assert "blocks.0.attn.hook_dsa_mask" in cache
        assert "blocks.0.mlp.hook_in" in cache
        assert "blocks.1.mlp.shared_experts.hook_out" in cache
        assert cache["blocks.0.attn.hook_topk_indices"].shape[-1] == 4
        dsa_mask = cache["blocks.0.attn.hook_dsa_mask"]
        assert torch.isneginf(dsa_mask).any()
        assert (dsa_mask == 0).any()
