"""Unit tests for CodeGenArchitectureAdapter."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CodeGenAttentionBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
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


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> CodeGenArchitectureAdapter:
    return CodeGenArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestCodeGenAdapterConfig:
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
        """Parallel attn+MLP shares ln_1; no ln2 exists."""
        blocks = adapter.component_mapping["blocks"]
        assert "ln2" not in blocks.submodules

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


class TestCodeGenAdapterWeightConversionSemantics:
    """Each Q/K/V/O wraps a RearrangeTensorConversion with the right pattern and n axis."""

    @pytest.fixture(scope="class")
    def adapter(self) -> CodeGenArchitectureAdapter:
        return CodeGenArchitectureAdapter(_make_cfg())

    @pytest.mark.parametrize("slot", ["q", "k", "v"])
    def test_qkv_uses_split_heads_pattern(
        self, adapter: CodeGenArchitectureAdapter, slot: str
    ) -> None:
        conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_o_uses_merge_heads_pattern(self, adapter: CodeGenArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_n_kv_heads_on_cfg_does_not_change_kv_conversions(self) -> None:
        # CodeGen is MHA-only: K/V pinned to n_heads regardless of n_key_value_heads.
        cfg = _make_cfg()
        cfg.n_key_value_heads = 1  # type: ignore[attr-defined]
        adapter = CodeGenArchitectureAdapter(cfg)
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


class TestCodeGenAdapterComponentTypesExtras:
    """Bridge-type assertions for joint-QKV, MLP submodules, and parallel block class."""

    @pytest.fixture(scope="class")
    def adapter(self) -> CodeGenArchitectureAdapter:
        return CodeGenArchitectureAdapter(_make_cfg())

    def test_blocks_is_parallel_block_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        # Parallel attn+MLP: must use ParallelBlockBridge, not sequential BlockBridge.
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, ParallelBlockBridge)

    def test_attn_qkv_is_linear_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        qkv = attn.submodules["qkv"]
        assert isinstance(qkv, LinearBridge)
        assert qkv.name == "qkv_proj"

    def test_attn_o_is_linear_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        o = attn.submodules["o"]
        assert isinstance(o, LinearBridge)
        assert o.name == "out_proj"

    def test_mlp_in_is_linear_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["in"], LinearBridge)

    def test_mlp_out_is_linear_bridge(self, adapter: CodeGenArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["out"], LinearBridge)

    def test_no_gate_in_mlp(self, adapter: CodeGenArchitectureAdapter) -> None:
        """Non-gated MLP: no 'gate' submodule."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert "gate" not in mlp.submodules


class TestCodeGenArchitectureGuards:
    """RoPE-in-attention (no top-level rotary_emb), no learned pos, no Gemma offsets."""

    @pytest.fixture(scope="class")
    def adapter(self) -> CodeGenArchitectureAdapter:
        return CodeGenArchitectureAdapter(_make_cfg())

    def test_no_top_level_rotary_emb(self, adapter: CodeGenArchitectureAdapter) -> None:
        # Rotary is applied inside attention forward; no standalone HF module to bind.
        assert "rotary_emb" not in adapter.component_mapping

    def test_no_pos_embed_component(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert "pos_embed" not in adapter.component_mapping

    def test_no_norm_offset_conversions(self, adapter: CodeGenArchitectureAdapter) -> None:
        # LN-only: no Gemma-style ln1/ln2 offsets.
        for key in adapter.weight_processing_conversions:
            assert "ln1.weight" not in key
            assert "ln2.weight" not in key
            assert "ln_final.weight" not in key

    def test_only_qkvo_conversion_keys(self, adapter: CodeGenArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }


# ---------------------------------------------------------------------------
# split_qkv_matrix numerical correctness tests
# ---------------------------------------------------------------------------


class TestCodeGenSplitQKVMatrix:
    """Numerical tests for the mp_num=4 QKV split."""

    def _make_adapter_with_dmodel(self, d_model: int, n_heads: int) -> CodeGenArchitectureAdapter:
        cfg = _make_cfg(d_model=d_model, n_heads=n_heads)
        return CodeGenArchitectureAdapter(cfg)

    def _make_attn_component(self, d_model: int) -> Any:
        """Minimal attn with a qkv_proj linear."""
        attn = SimpleNamespace()
        attn.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        return attn

    def test_returns_three_linear_modules(self) -> None:
        adapter = self._make_adapter_with_dmodel(64, 4)
        attn = self._make_attn_component(64)
        q, k, v = adapter.split_qkv_matrix(attn)
        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)

    def test_output_shapes_are_correct(self) -> None:
        d_model = 64
        adapter = self._make_adapter_with_dmodel(d_model, 4)
        attn = self._make_attn_component(d_model)
        q, k, v = adapter.split_qkv_matrix(attn)
        assert q.weight.shape == (d_model, d_model)
        assert k.weight.shape == (d_model, d_model)
        assert v.weight.shape == (d_model, d_model)

    def test_no_bias_on_outputs(self) -> None:
        adapter = self._make_adapter_with_dmodel(64, 4)
        attn = self._make_attn_component(64)
        q, k, v = adapter.split_qkv_matrix(attn)
        assert q.bias is None
        assert k.bias is None
        assert v.bias is None

    def test_q_k_v_are_distinct(self) -> None:
        adapter = self._make_adapter_with_dmodel(64, 4)
        attn = self._make_attn_component(64)
        nn.init.normal_(attn.qkv_proj.weight)
        q, k, v = adapter.split_qkv_matrix(attn)
        assert not torch.allclose(q.weight, k.weight)
        assert not torch.allclose(q.weight, v.weight)
        assert not torch.allclose(k.weight, v.weight)

    def test_known_partition_ordering(self) -> None:
        """mp_num=4 layout within each partition is [Q_part, V_part, K_part]."""
        mp_num = 4
        d_model = 64
        n_heads = 4
        local_dim = d_model // mp_num  # 16

        adapter = self._make_adapter_with_dmodel(d_model, n_heads)
        attn = self._make_attn_component(d_model)

        # Tag each slot with a unique constant to track its destination.
        w = torch.zeros(mp_num, 3, local_dim, d_model)
        w[:, 0, :, :] = 1.0  # Q_part
        w[:, 1, :, :] = 2.0  # V_part
        w[:, 2, :, :] = 3.0  # K_part

        attn.qkv_proj.weight = nn.Parameter(w.reshape(3 * d_model, d_model))

        q, k, v = adapter.split_qkv_matrix(attn)

        assert torch.all(q.weight == 1.0), "Q should come from slot 0 (Q_part)"
        assert torch.all(k.weight == 3.0), "K should come from slot 2 (K_part)"
        assert torch.all(v.weight == 2.0), "V should come from slot 1 (V_part)"

    def test_forward_output_shape_with_split(self) -> None:
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
    def test_factory_returns_codegen_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, CodeGenArchitectureAdapter)

    def test_factory_key_is_codegen_for_causal_lm(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "CodeGenForCausalLM" in SUPPORTED_ARCHITECTURES
