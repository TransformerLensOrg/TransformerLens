"""Unit tests for MLAAttentionBridge (DeepSeek Multi-Head Latent Attention)."""

import pytest
import torch
from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

from transformer_lens.model_bridge.generalized_components.mla_attention import (
    MLAAttentionBridge,
)


@pytest.fixture(scope="module")
def tiny_config():
    return DeepseekV3Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=1,
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


@pytest.fixture(scope="module")
def tiny_model(tiny_config):
    return DeepseekV3ForCausalLM(tiny_config)


@pytest.fixture(scope="module")
def hf_attn(tiny_model):
    return tiny_model.model.layers[0].self_attn


@pytest.fixture(scope="module")
def mla_bridge(tiny_config, hf_attn, tiny_model):
    bridge = MLAAttentionBridge(name="self_attn", config=tiny_config, submodules={})
    bridge.set_original_component(hf_attn)
    bridge.set_rotary_emb(tiny_model.model.rotary_emb)
    return bridge


class TestMLAAttentionBridgeHooks:
    def test_all_expected_hooks_exist(self, mla_bridge):
        for hook_name in [
            "hook_in",
            "hook_out",
            "hook_q_latent",
            "hook_kv_latent",
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_rot_q",
            "hook_rot_k",
            "hook_attn_scores",
            "hook_pattern",
            "hook_cos",
            "hook_sin",
        ]:
            assert hasattr(mla_bridge, hook_name), f"Missing hook: {hook_name}"

    def test_W_Q_raises_not_implemented(self, mla_bridge):
        with pytest.raises(NotImplementedError, match="not available on MLA"):
            _ = mla_bridge.W_Q

    def test_W_K_raises_not_implemented(self, mla_bridge):
        with pytest.raises(NotImplementedError, match="not available on MLA"):
            _ = mla_bridge.W_K


class TestMLAAttentionBridgeForward:
    @pytest.fixture
    def sample_inputs(self, tiny_config, tiny_model):
        batch, seq = 2, 8
        hidden_states = torch.randn(batch, seq, tiny_config.hidden_size)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        cos, sin = tiny_model.model.rotary_emb(hidden_states, position_ids)
        return hidden_states, (cos, sin)

    def test_output_matches_hf(self, mla_bridge, hf_attn, sample_inputs, tiny_model):
        """HF uses SDPA, bridge uses manual matmul — small float32 differences expected."""
        hidden_states, position_embeddings = sample_inputs

        with torch.no_grad():
            hf_attn_out = hf_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )[0]
            bridge_attn_out = mla_bridge(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )[0]

        max_diff = (hf_attn_out - bridge_attn_out).abs().max().item()
        mean_diff = (hf_attn_out - bridge_attn_out).abs().mean().item()
        assert max_diff < 0.15, f"Output too different: max diff = {max_diff}"
        assert mean_diff < 0.02, f"Output too different: mean diff = {mean_diff}"

    def test_hooks_fire_and_have_correct_shapes(self, mla_bridge, sample_inputs, tiny_config):
        hidden_states, position_embeddings = sample_inputs
        batch, seq = hidden_states.shape[:2]
        captured = {}

        hooks_to_check = [
            "hook_q_latent",
            "hook_kv_latent",
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_rot_q",
            "hook_rot_k",
            "hook_attn_scores",
            "hook_pattern",
        ]

        handles = []
        for name in hooks_to_check:

            def make_capture(n):
                def hook_fn(module, input, output):
                    captured[n] = output.shape

                return hook_fn

            handles.append(getattr(mla_bridge, name).register_forward_hook(make_capture(name)))

        try:
            with torch.no_grad():
                mla_bridge(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=None
                )
        finally:
            for h in handles:
                h.remove()

        n_heads = tiny_config.num_attention_heads
        qk_head_dim = tiny_config.qk_nope_head_dim + tiny_config.qk_rope_head_dim

        for name in hooks_to_check:
            assert name in captured, f"Hook {name} did not fire"

        assert captured["hook_q_latent"] == (batch, seq, tiny_config.q_lora_rank)
        assert captured["hook_kv_latent"] == (batch, seq, tiny_config.kv_lora_rank)
        assert captured["hook_q"] == (batch, n_heads, seq, qk_head_dim)
        assert captured["hook_k"] == (batch, n_heads, seq, qk_head_dim)
        assert captured["hook_v"] == (batch, n_heads, seq, tiny_config.v_head_dim)
        assert captured["hook_attn_scores"] == (batch, n_heads, seq, seq)
        assert captured["hook_pattern"] == (batch, n_heads, seq, seq)

    def test_hook_q_is_post_rope(self, mla_bridge, sample_inputs):
        """hook_q's rope portion should match hook_rot_q."""
        hidden_states, position_embeddings = sample_inputs
        q_values: list[torch.Tensor] = []
        rot_q_values: list[torch.Tensor] = []

        h1 = mla_bridge.hook_q.register_forward_hook(lambda m, i, o: q_values.append(o.clone()))
        h2 = mla_bridge.hook_rot_q.register_forward_hook(
            lambda m, i, o: rot_q_values.append(o.clone())
        )

        try:
            with torch.no_grad():
                mla_bridge(
                    hidden_states, position_embeddings=position_embeddings, attention_mask=None
                )
        finally:
            h1.remove()
            h2.remove()

        qk_rope_dim = mla_bridge._qk_rope_head_dim
        assert torch.allclose(q_values[0][..., -qk_rope_dim:], rot_q_values[0], atol=1e-5)
