"""Integration tests for DeepSeek V3 architecture adapter."""

import tempfile

import pytest
import torch
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def tiny_deepseek_bridge():
    tiny_config = DeepseekV3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=1000,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        max_position_embeddings=128,
        moe_intermediate_size=256,
    )
    hf_model = DeepseekV3ForCausalLM(tiny_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_model.save_pretrained(tmpdir)
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.save_pretrained(tmpdir)
        bridge = TransformerBridge.boot_transformers(tmpdir, device="cpu")
        yield bridge


class TestDeepSeekBridgeCreation:
    def test_bridge_has_correct_block_count(self, tiny_deepseek_bridge):
        assert len(tiny_deepseek_bridge.blocks) == 4

    def test_bridge_has_embed_and_unembed(self, tiny_deepseek_bridge):
        assert hasattr(tiny_deepseek_bridge, "embed")
        assert hasattr(tiny_deepseek_bridge, "unembed")
        assert hasattr(tiny_deepseek_bridge, "ln_final")

    def test_attention_is_mla(self, tiny_deepseek_bridge):
        from transformer_lens.model_bridge.generalized_components.mla_attention import (
            MLAAttentionBridge,
        )

        assert isinstance(tiny_deepseek_bridge.blocks[0].attn, MLAAttentionBridge)


class TestDeepSeekForwardPass:
    def test_forward_returns_logits(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = tiny_deepseek_bridge(tokens)
        assert output.shape == (1, 4, 1000)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(self, tiny_deepseek_bridge):
        """SDPA vs manual matmul — small float32 differences expected."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        hf_model = tiny_deepseek_bridge.original_model
        with torch.no_grad():
            bridge_out = tiny_deepseek_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 0.15, f"Bridge vs HF max diff = {max_diff}"


class TestDeepSeekDenseVsMoELayers:
    def test_dense_layer_has_no_moe_hooks(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        cache_keys = set(cache.keys())
        assert not any("blocks.0.mlp.gate" in k for k in cache_keys)
        assert not any("blocks.0.mlp.shared_experts" in k for k in cache_keys)

    def test_moe_layer_has_gate_hooks(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        assert any("blocks.1.mlp.gate" in k for k in cache.keys())

    def test_moe_layer_has_shared_experts_hooks(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        assert any("blocks.1.mlp.shared_experts" in k for k in cache.keys())

    def test_both_layers_have_mlp_hooks(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        for i in [0, 1]:
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_both_layers_produce_non_nan(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        for i in [0, 1]:
            assert not torch.isnan(cache[f"blocks.{i}.mlp.hook_out"]).any()


class TestDeepSeekAttentionHooks:
    def test_attention_hooks_fire_all_layers(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        for i in range(4):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache

    def test_mla_latent_hooks_fire(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        assert any("hook_q_latent" in k for k in cache.keys())
        assert any("hook_kv_latent" in k for k in cache.keys())
