"""Unit tests for CodeGenArchitectureAdapter.

Tests cover:
- Config attribute validation (all required attributes are set correctly)
- Component mapping structure (correct bridge types, no ln2)
- Weight conversion keys and structure
- split_qkv_matrix correctness (numerical test with known weights)
- Factory registration (CodeGenForCausalLM maps to the right adapter)
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CodeGenAttentionBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.codegen import (
    CodeGenArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for CodeGen adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="CodeGenForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> CodeGenArchitectureAdapter:
    return CodeGenArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestCodeGenAdapterConfig:
    """Tests that the adapter sets required config attributes correctly."""

    def test_normalization_type_is_ln(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_rotary(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_false(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_parallel_attn_mlp_is_true(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.cfg.parallel_attn_mlp is True


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestCodeGenAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and structure."""

    def test_embed_is_embedding_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_blocks_is_block_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.h"

    def test_ln_final_is_normalization_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_blocks_ln1_is_normalization_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_blocks_ln1_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "ln_1"

    def test_no_ln2_in_blocks(self, adapter: CodeGenArchitectureAdapter) -> None:
        """CodeGen uses parallel attn+MLP sharing ln_1 — there must be no ln2."""
        blocks = adapter.component_mapping["blocks"]
        assert "ln2" not in blocks.submodules, "CodeGen parallel block must not have ln2"

    def test_attn_is_codegen_attention_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], CodeGenAttentionBridge)

    def test_attn_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attn"

    def test_mlp_is_mlp_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "mlp"

    def test_mlp_in_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["in"].name == "fc_in"

    def test_mlp_out_name(self, adapter: CodeGenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["out"].name == "fc_out"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestCodeGenAdapterWeightConversions:
    """Tests that weight_processing_conversions has the expected keys."""

    def test_q_weight_key_present(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4


# ---------------------------------------------------------------------------
# split_qkv_matrix numerical correctness tests
# ---------------------------------------------------------------------------


class TestCodeGenSplitQKVMatrix:
    """Numerical tests verifying the mp_num=4 QKV split logic."""

    def _make_adapter_with_dmodel(self, d_model: int, n_heads: int) -> CodeGenArchitectureAdapter:
        cfg = _make_cfg(d_model=d_model, n_heads=n_heads)
        return CodeGenArchitectureAdapter(cfg)

    def _make_attn_component(self, d_model: int) -> Any:
        """Create a minimal attn component with a qkv_proj linear."""
        attn = SimpleNamespace()
        attn.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        return attn

    def test_returns_three_linear_modules(self) -> None:
        """split_qkv_matrix must return exactly three nn.Linear modules."""
        adapter = self._make_adapter_with_dmodel(64, 4)
        attn = self._make_attn_component(64)
        q, k, v = adapter.split_qkv_matrix(attn)
        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)

    def test_output_shapes_are_correct(self) -> None:
        """Each of Q, K, V must have weight shape [n_embd, n_embd]."""
        d_model = 64
        adapter = self._make_adapter_with_dmodel(d_model, 4)
        attn = self._make_attn_component(d_model)
        q, k, v = adapter.split_qkv_matrix(attn)
        assert q.weight.shape == (d_model, d_model)
        assert k.weight.shape == (d_model, d_model)
        assert v.weight.shape == (d_model, d_model)

    def test_no_bias_on_outputs(self) -> None:
        """The split linears must have no bias, matching qkv_proj."""
        adapter = self._make_adapter_with_dmodel(64, 4)
        attn = self._make_attn_component(64)
        q, k, v = adapter.split_qkv_matrix(attn)
        assert q.bias is None
        assert k.bias is None
        assert v.bias is None

    def test_q_k_v_are_distinct(self) -> None:
        """With a non-trivial weight, Q, K, V must differ from each other."""
        adapter = self._make_adapter_with_dmodel(64, 4)
        attn = self._make_attn_component(64)
        # Fill qkv_proj with distinct values per row
        nn.init.normal_(attn.qkv_proj.weight)
        q, k, v = adapter.split_qkv_matrix(attn)
        # All three must differ
        assert not torch.allclose(q.weight, k.weight), "Q and K weights must differ"
        assert not torch.allclose(q.weight, v.weight), "Q and V weights must differ"
        assert not torch.allclose(k.weight, v.weight), "K and V weights must differ"

    def test_known_partition_ordering(self) -> None:
        """Verify the mp_num=4 partition layout: within each partition [Q_part, V_part, K_part].

        We construct a weight where partition index and slot index are embedded
        in the values, then verify that Q, K, V extract the correct slices.
        """
        mp_num = 4
        d_model = 64
        n_heads = 4
        local_dim = d_model // mp_num  # 16

        adapter = self._make_adapter_with_dmodel(d_model, n_heads)
        attn = self._make_attn_component(d_model)

        # Build a structured weight: rows are indexed 0..3*d_model-1.
        # Reshape as [mp_num=4, 3, local_dim=16, d_model=64], set each slice
        # to a unique constant so we can track which slot goes where.
        w = torch.zeros(mp_num, 3, local_dim, d_model)
        # slot 0 = Q_part → fill with 1.0
        w[:, 0, :, :] = 1.0
        # slot 1 = V_part → fill with 2.0
        w[:, 1, :, :] = 2.0
        # slot 2 = K_part → fill with 3.0
        w[:, 2, :, :] = 3.0

        # Flatten back to [3*d_model, d_model] as qkv_proj expects
        attn.qkv_proj.weight = nn.Parameter(w.reshape(3 * d_model, d_model))

        q, k, v = adapter.split_qkv_matrix(attn)

        assert torch.all(q.weight == 1.0), "Q should come from slot 0 (Q_part)"
        assert torch.all(k.weight == 3.0), "K should come from slot 2 (K_part)"
        assert torch.all(v.weight == 2.0), "V should come from slot 1 (V_part)"

    def test_forward_output_shape_with_split(self) -> None:
        """After split, Q/K/V linears should produce correct output shapes."""
        d_model = 64
        adapter = self._make_adapter_with_dmodel(d_model, 4)
        attn = self._make_attn_component(d_model)
        q_lin, k_lin, v_lin = adapter.split_qkv_matrix(attn)

        batch, seq = 2, 10
        x = torch.randn(batch, seq, d_model)
        assert q_lin(x).shape == (batch, seq, d_model)
        assert k_lin(x).shape == (batch, seq, d_model)
        assert v_lin(x).shape == (batch, seq, d_model)


# ---------------------------------------------------------------------------
# Factory registration test
# ---------------------------------------------------------------------------


class TestCodeGenFactoryRegistration:
    """Tests that the factory maps CodeGenForCausalLM to the correct adapter.

    Note: Phase D (registration) is required for these tests to pass.  They
    are included here so that registration is verified as part of the Phase D
    commit rather than needing a separate test file.
    """

    def test_factory_returns_codegen_adapter(self) -> None:
        """ArchitectureAdapterFactory must return a CodeGenArchitectureAdapter."""
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(
            adapter, CodeGenArchitectureAdapter
        ), f"Expected CodeGenArchitectureAdapter, got {type(adapter).__name__}"

    def test_factory_key_is_codegen_for_causal_lm(self) -> None:
        """SUPPORTED_ARCHITECTURES must have a 'CodeGenForCausalLM' key."""
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert (
            "CodeGenForCausalLM" in SUPPORTED_ARCHITECTURES
        ), "CodeGenForCausalLM must be registered in SUPPORTED_ARCHITECTURES"
