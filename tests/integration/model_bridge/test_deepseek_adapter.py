"""Integration tests for DeepSeek V3 architecture adapter.

Uses a tiny programmatic DeepseekV3 model (~6.5M params) saved to a temp dir,
since no small pretrained DeepSeek V3/R1 models exist (smallest is 671B).
"""

import tempfile

import pytest
import torch
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3ForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def tiny_deepseek_bridge():
    """Create a TransformerBridge wrapping a tiny programmatic DeepSeek V3 model."""
    tiny_config = DeepseekV3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,  # 1 dense + 3 MoE
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
    """Test bridge creation and structural validation."""

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

        attn = tiny_deepseek_bridge.blocks[0].attn
        assert isinstance(attn, MLAAttentionBridge)


class TestDeepSeekForwardPass:
    """Test forward pass produces valid output."""

    def test_forward_returns_logits(self, tiny_deepseek_bridge):
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = tiny_deepseek_bridge(tokens)
        assert output.shape == (1, 4, 1000)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(self, tiny_deepseek_bridge):
        """Bridge output should be close to HF model output."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        hf_model = tiny_deepseek_bridge.original_model
        with torch.no_grad():
            bridge_out = tiny_deepseek_bridge(tokens)
            hf_out = hf_model(tokens).logits
        # SDPA vs eager difference expected
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 0.2, f"Bridge vs HF max diff = {max_diff}"


class TestDeepSeekDenseVsMoELayers:
    """Test that dense and MoE layers are handled correctly."""

    def test_dense_layer_has_no_moe_hooks(self, tiny_deepseek_bridge):
        """Dense layer (blocks[0].mlp) should NOT have gate/shared_experts hooks."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        cache_keys = set(cache.keys())

        assert not any(
            "blocks.0.mlp.gate" in k for k in cache_keys
        ), "Dense layer should not have gate hooks"
        assert not any(
            "blocks.0.mlp.shared_experts" in k for k in cache_keys
        ), "Dense layer should not have shared_experts hooks"

    def test_moe_layer_has_gate_hooks(self, tiny_deepseek_bridge):
        """MoE layer (blocks[1].mlp) SHOULD have gate hooks."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        cache_keys = set(cache.keys())

        assert any("blocks.1.mlp.gate" in k for k in cache_keys), "MoE layer should have gate hooks"

    def test_moe_layer_has_shared_experts_hooks(self, tiny_deepseek_bridge):
        """MoE layer should have shared_experts hooks."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        cache_keys = set(cache.keys())

        assert any(
            "blocks.1.mlp.shared_experts" in k for k in cache_keys
        ), "MoE layer should have shared_experts hooks"

    def test_both_layers_have_mlp_hooks(self, tiny_deepseek_bridge):
        """Both dense and MoE layers should have basic hook_in/hook_out."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        cache_keys = set(cache.keys())

        assert "blocks.0.mlp.hook_in" in cache_keys
        assert "blocks.0.mlp.hook_out" in cache_keys
        assert "blocks.1.mlp.hook_in" in cache_keys
        assert "blocks.1.mlp.hook_out" in cache_keys

    def test_both_layers_produce_non_nan(self, tiny_deepseek_bridge):
        """Both dense and MoE layers should produce non-NaN output."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)

        for i in [0, 1]:
            key = f"blocks.{i}.mlp.hook_out"
            assert key in cache, f"Missing {key}"
            assert not torch.isnan(cache[key]).any(), f"NaN in {key}"


class TestDeepSeekAttentionHooks:
    """Test MLA attention hooks fire on all layers."""

    def test_attention_hooks_fire_all_layers(self, tiny_deepseek_bridge):
        """Attention hooks should fire on every layer."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        cache_keys = set(cache.keys())

        for i in range(4):
            assert f"blocks.{i}.attn.hook_in" in cache_keys, f"Layer {i} missing attn.hook_in"
            assert f"blocks.{i}.attn.hook_out" in cache_keys, f"Layer {i} missing attn.hook_out"

    def test_mla_latent_hooks_fire(self, tiny_deepseek_bridge):
        """MLA-specific latent hooks should fire."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = tiny_deepseek_bridge.run_with_cache(tokens)
        cache_keys = set(cache.keys())

        assert any("hook_q_latent" in k for k in cache_keys), "hook_q_latent should fire"
        assert any("hook_kv_latent" in k for k in cache_keys), "hook_kv_latent should fire"
