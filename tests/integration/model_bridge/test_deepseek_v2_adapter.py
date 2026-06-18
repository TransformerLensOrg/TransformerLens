"""Integration tests for DeepSeek V2 architecture adapter.

Covers two distinct variants of DeepseekV2ForCausalLM:
- V2-full (q_lora_rank set): Q is compressed via two-stage LoRA projection.
- V2-Lite (q_lora_rank=None): Q uses a direct linear projection; no compression.
"""

import tempfile

import pytest
import torch
from transformers import AutoTokenizer, DeepseekV2Config, DeepseekV2ForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge


def _make_bridge(q_lora_rank):
    """Build a tiny DeepseekV2 bridge with the given q_lora_rank (None = V2-Lite)."""
    cfg = DeepseekV2Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        vocab_size=1000,
        first_k_dense_replace=1,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        moe_intermediate_size=256,
    )
    hf_model = DeepseekV2ForCausalLM(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_model.save_pretrained(tmpdir)
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.save_pretrained(tmpdir)
        return TransformerBridge.boot_transformers(tmpdir, device="cpu")


@pytest.fixture(scope="module")
def tiny_deepseek_v2_bridge():
    """V2-full: q_lora_rank=64 — two-stage Q compression (same as V3)."""
    return _make_bridge(q_lora_rank=64)


@pytest.fixture(scope="module")
def tiny_deepseek_v2_lite_bridge():
    """V2-Lite: q_lora_rank=None — direct Q projection, no LoRA compression."""
    return _make_bridge(q_lora_rank=None)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tokens():
    return torch.tensor([[1, 2, 3, 4]])


# ---------------------------------------------------------------------------
# V2-full tests
# ---------------------------------------------------------------------------

class TestDeepSeekV2BridgeCreation:
    def test_block_count(self, tiny_deepseek_v2_bridge):
        assert len(tiny_deepseek_v2_bridge.blocks) == 4

    def test_has_embed_unembed_ln_final(self, tiny_deepseek_v2_bridge):
        assert hasattr(tiny_deepseek_v2_bridge, "embed")
        assert hasattr(tiny_deepseek_v2_bridge, "unembed")
        assert hasattr(tiny_deepseek_v2_bridge, "ln_final")

    def test_attention_is_mla(self, tiny_deepseek_v2_bridge):
        from transformer_lens.model_bridge.generalized_components.mla_attention import (
            MLAAttentionBridge,
        )
        assert isinstance(tiny_deepseek_v2_bridge.blocks[0].attn, MLAAttentionBridge)


class TestDeepSeekV2ForwardPass:
    def test_forward_returns_correct_shape(self, tiny_deepseek_v2_bridge):
        tokens = _tokens()
        with torch.no_grad():
            out = tiny_deepseek_v2_bridge(tokens)
        assert out.shape == (1, 4, 1000)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_matches_hf(self, tiny_deepseek_v2_bridge):
        tokens = _tokens()
        hf_model = tiny_deepseek_v2_bridge.original_model
        with torch.no_grad():
            bridge_out = tiny_deepseek_v2_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 0.15, f"Bridge vs HF max diff = {max_diff}"


class TestDeepSeekV2DenseVsMoELayers:
    def test_dense_layer_has_no_moe_hooks(self, tiny_deepseek_v2_bridge):
        _, cache = tiny_deepseek_v2_bridge.run_with_cache(_tokens())
        assert not any("blocks.0.mlp.gate" in k for k in cache)
        assert not any("blocks.0.mlp.shared_experts" in k for k in cache)

    def test_moe_layer_has_shared_expert_hooks(self, tiny_deepseek_v2_bridge):
        # DeepseekV2Moe.forward() routes via nn.functional.linear(..., self.gate.weight)
        # directly — not self.gate(hidden_states) — so the gate module's forward() is
        # never called and its bridge hooks cannot fire. shared_experts IS called via
        # __call__, so GatedMLPBridge hooks fire correctly.
        _, cache = tiny_deepseek_v2_bridge.run_with_cache(_tokens())
        assert not any("blocks.1.mlp.gate" in k for k in cache), (
            "gate hooks should not appear — gate is called via functional.linear, not forward()"
        )
        assert any("blocks.1.mlp.shared_experts" in k for k in cache)

    def test_all_layers_have_mlp_hooks(self, tiny_deepseek_v2_bridge):
        _, cache = tiny_deepseek_v2_bridge.run_with_cache(_tokens())
        for i in range(4):
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache
            assert not torch.isnan(cache[f"blocks.{i}.mlp.hook_out"]).any()


class TestDeepSeekV2AttentionHooks:
    def test_attn_hooks_fire_all_layers(self, tiny_deepseek_v2_bridge):
        _, cache = tiny_deepseek_v2_bridge.run_with_cache(_tokens())
        for i in range(4):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache

    def test_mla_latent_hooks_fire(self, tiny_deepseek_v2_bridge):
        _, cache = tiny_deepseek_v2_bridge.run_with_cache(_tokens())
        assert any("hook_q_latent" in k for k in cache)
        assert any("hook_kv_latent" in k for k in cache)


# ---------------------------------------------------------------------------
# V2-Lite tests (q_lora_rank=None — direct q_proj, no compression)
# ---------------------------------------------------------------------------

class TestDeepSeekV2LiteBridgeCreation:
    def test_block_count(self, tiny_deepseek_v2_lite_bridge):
        assert len(tiny_deepseek_v2_lite_bridge.blocks) == 4

    def test_attention_is_mla(self, tiny_deepseek_v2_lite_bridge):
        from transformer_lens.model_bridge.generalized_components.mla_attention import (
            MLAAttentionBridge,
        )
        assert isinstance(tiny_deepseek_v2_lite_bridge.blocks[0].attn, MLAAttentionBridge)


class TestDeepSeekV2LiteForwardPass:
    def test_forward_returns_correct_shape(self, tiny_deepseek_v2_lite_bridge):
        tokens = _tokens()
        with torch.no_grad():
            out = tiny_deepseek_v2_lite_bridge(tokens)
        assert out.shape == (1, 4, 1000)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_matches_hf(self, tiny_deepseek_v2_lite_bridge):
        tokens = _tokens()
        hf_model = tiny_deepseek_v2_lite_bridge.original_model
        with torch.no_grad():
            bridge_out = tiny_deepseek_v2_lite_bridge(tokens)
            hf_out = hf_model(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 0.15, f"V2-Lite bridge vs HF max diff = {max_diff}"


class TestDeepSeekV2LiteNoQLatentHook:
    def test_hook_q_latent_absent_without_q_lora_rank(self, tiny_deepseek_v2_lite_bridge):
        """V2-Lite skips Q compression — hook_q_latent should not fire."""
        _, cache = tiny_deepseek_v2_lite_bridge.run_with_cache(_tokens())
        assert not any("hook_q_latent" in k for k in cache)

    def test_hook_kv_latent_still_fires(self, tiny_deepseek_v2_lite_bridge):
        """KV compression is always present regardless of q_lora_rank."""
        _, cache = tiny_deepseek_v2_lite_bridge.run_with_cache(_tokens())
        assert any("hook_kv_latent" in k for k in cache)

    def test_all_layers_produce_non_nan(self, tiny_deepseek_v2_lite_bridge):
        _, cache = tiny_deepseek_v2_lite_bridge.run_with_cache(_tokens())
        for i in range(4):
            assert not torch.isnan(cache[f"blocks.{i}.attn.hook_out"]).any()
