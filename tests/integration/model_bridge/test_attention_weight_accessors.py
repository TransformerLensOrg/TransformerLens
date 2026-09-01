"""Numeric checks that booted-bridge W_Q/W_K/W_V/W_O accessors match the wrapped HF projections.

Regression tests for the square-weight layout ambiguity: nn.Linear stores
[out, in] while Conv1D stores [in, out], so a shape heuristic cannot tell them
apart when d_model == n_heads * d_head. Tiny randomly-initialized models, CPU,
no Hub access.
"""
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge


@pytest.fixture(scope="module")
def llama_bridge():
    """Tiny Llama: square nn.Linear o_proj (the historically-broken case), GQA k/v."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from transformer_lens.model_bridge.supported_architectures.llama import (
        LlamaArchitectureAdapter,
    )

    torch.manual_seed(0)
    hf_model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=128,
        )
    ).eval()
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_layers=2,
        n_ctx=128,
        d_vocab=256,
        n_key_value_heads=2,
        architecture="LlamaForCausalLM",
    )
    return TransformerBridge(hf_model, LlamaArchitectureAdapter(cfg), tokenizer=MagicMock()), (
        hf_model
    )


@pytest.fixture(scope="module")
def tiny_gpt2_bridge():
    """Tiny GPT-2: square Conv1D c_proj (control — no-transpose path must stay correct)."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from transformer_lens.model_bridge.supported_architectures.gpt2 import (
        GPT2ArchitectureAdapter,
    )

    torch.manual_seed(0)
    hf_model = GPT2LMHeadModel(
        GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=256, n_positions=128)
    ).eval()
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_layers=2,
        n_ctx=128,
        d_vocab=256,
        architecture="GPT2LMHeadModel",
    )
    return TransformerBridge(hf_model, GPT2ArchitectureAdapter(cfg), tokenizer=MagicMock()), (
        hf_model
    )


class TestLinearAccessorParity:
    def test_w_o_reproduces_o_proj(self, llama_bridge):
        bridge, hf_model = llama_bridge
        w_o = bridge.blocks[0].attn.W_O
        assert w_o.shape == (4, 16, 64)
        z = torch.randn(2, 4, 16)
        expected = hf_model.model.layers[0].self_attn.o_proj(z.reshape(2, 64))
        actual = torch.einsum("bhd,hdm->bm", z, w_o)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_w_q_reproduces_q_proj(self, llama_bridge):
        bridge, hf_model = llama_bridge
        w_q = bridge.blocks[0].attn.W_Q
        assert w_q.shape == (4, 64, 16)
        x = torch.randn(2, 64)
        expected = hf_model.model.layers[0].self_attn.q_proj(x).reshape(2, 4, 16)
        actual = torch.einsum("bm,hmd->bhd", x, w_q)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_w_k_uses_kv_heads(self, llama_bridge):
        bridge, hf_model = llama_bridge
        w_k = bridge.blocks[0].attn.W_K
        assert w_k.shape == (2, 64, 16)
        x = torch.randn(2, 64)
        expected = hf_model.model.layers[0].self_attn.k_proj(x).reshape(2, 2, 16)
        actual = torch.einsum("bm,hmd->bhd", x, w_k)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_w_v_uses_kv_heads(self, llama_bridge):
        bridge, hf_model = llama_bridge
        w_v = bridge.blocks[0].attn.W_V
        assert w_v.shape == (2, 64, 16)
        x = torch.randn(2, 64)
        expected = hf_model.model.layers[0].self_attn.v_proj(x).reshape(2, 2, 16)
        actual = torch.einsum("bm,hmd->bhd", x, w_v)
        assert torch.allclose(actual, expected, atol=1e-5)


class TestConv1DAccessorParity:
    def test_w_o_reproduces_c_proj(self, tiny_gpt2_bridge):
        bridge, hf_model = tiny_gpt2_bridge
        attn = bridge.blocks[0].attn
        w_o = attn.W_O
        assert w_o.shape == (4, 16, 64)
        z = torch.randn(2, 4, 16)
        c_proj = hf_model.transformer.h[0].attn.c_proj
        expected = c_proj(z.reshape(2, 64))
        actual = torch.einsum("bhd,hdm->bm", z, w_o) + attn.b_O
        assert torch.allclose(actual, expected, atol=1e-5)


class TestWeightCircuitsGQA:
    """QK/OV/composition circuits expand grouped K/V to n_heads (issue #1553).

    Pre-fix, every property below raised at FactoredMatrix construction on GQA
    models because the grouped [n_kv_heads] axis cannot broadcast against the
    per-query-head [n_heads] axis.
    """

    def test_qk_factors_align_to_query_heads(self, llama_bridge):
        bridge, _ = llama_bridge
        QK = bridge.QK
        assert QK.A.shape == (2, 4, 64, 16)
        assert QK.B.shape == (2, 4, 16, 64)
        # Query head h reads kv head h // (n_heads // n_kv_heads).
        for h in range(4):
            assert torch.equal(QK.B[0, h], bridge.blocks[0].attn.W_K[h // 2].T)

    def test_ov_factors_align_to_query_heads(self, llama_bridge):
        bridge, _ = llama_bridge
        OV = bridge.OV
        assert OV.A.shape == (2, 4, 64, 16)
        assert OV.B.shape == (2, 4, 16, 64)
        for h in range(4):
            assert torch.equal(OV.A[1, h], bridge.blocks[1].attn.W_V[h // 2])

    def test_for_attn_layers_variants_align(self, llama_bridge):
        bridge, _ = llama_bridge
        indices, QK = bridge.QK_for_attn_layers()
        assert indices == [0, 1]
        assert QK.A.shape == (2, 4, 64, 16)
        assert QK.B.shape == (2, 4, 16, 64)
        _, OV = bridge.OV_for_attn_layers()
        assert OV.A.shape == (2, 4, 64, 16)
        assert OV.B.shape == (2, 4, 16, 64)

    @pytest.mark.parametrize("mode", ["Q", "K", "V"])
    def test_composition_scores_cover_all_query_heads(self, llama_bridge, mode):
        bridge, _ = llama_bridge
        result = bridge.all_composition_scores(mode)
        assert result.scores.shape == (2, 4, 2, 4)
        assert len(result.head_labels) == 8

    def test_raw_kv_stacks_stay_grouped(self, llama_bridge):
        bridge, _ = llama_bridge
        assert bridge.W_K.shape == (2, 2, 64, 16)
        assert bridge.W_V.shape == (2, 2, 64, 16)

    def test_mha_circuits_untouched(self, tiny_gpt2_bridge):
        bridge, _ = tiny_gpt2_bridge
        QK, OV = bridge.QK, bridge.OV
        assert torch.equal(QK.A, bridge.W_Q)
        assert torch.equal(QK.B, bridge.W_K.transpose(-2, -1))
        assert torch.equal(OV.A, bridge.W_V)
        assert torch.equal(OV.B, bridge.W_O)


class TestProjectionKernelGQA:
    """Head affinity keeps grouped K/V heads native rather than expanding them."""

    @pytest.mark.parametrize("role", ["K", "V"])
    def test_native_kv_axes_and_sample_parity(self, llama_bridge, role):
        from transformer_lens.tools.analysis.projection_kernel import (
            attention_head_subspace_affinity,
            orthonormal_subspace,
            projection_kernel,
        )

        bridge, _ = llama_bridge
        result = attention_head_subspace_affinity(bridge, target_role=role)
        target_weight = getattr(bridge.blocks[1].attn, f"W_{role}")[1]
        expected = projection_kernel(
            orthonormal_subspace(bridge.blocks[0].attn.W_O[0].T),
            orthonormal_subspace(target_weight),
        )

        assert result.scores.shape == (2, 4, 2, 2)
        assert int(result.valid_mask.sum()) == 8
        assert result.target_head_kind == "kv"
        assert result.scores[0, 0, 1, 1].item() == pytest.approx(
            expected.score.item(), rel=1e-5, abs=1e-5
        )
