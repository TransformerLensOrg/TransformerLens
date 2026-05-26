"""Unit tests for Qwen3ArchitectureAdapter.

Tests cover:
- Config attributes
- Component mapping structure and HF module names (incl. q_norm/k_norm)
- Weight conversion keys/types (GQA: k/v use n_key_value_heads)
- _preprocess_gated_q_proj static helper (gated q_proj slicing)
- Factory registration
"""

from typing import Any

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 8,
    n_key_value_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 100,
    n_ctx: int = 128,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for Qwen3 adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        default_prepend_bos=False,
        architecture="Qwen3ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> Qwen3ArchitectureAdapter:
    return Qwen3ArchitectureAdapter(cfg)


class TestQwen3AdapterConfig:
    """
    Config attribute tests
    """

    def test_normalization_type(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_default_prepend_bos_false(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False

    def test_attn_implementation_eager(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_implementation == "eager"

    def test_n_key_value_heads_preserved(self, adapter: Qwen3ArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 4


class TestQwen3AdapterComponentMapping:
    """
    Testcases for component mapping setup
    """

    @staticmethod
    def _mapping(adapter: Qwen3ArchitectureAdapter) -> dict[str, Any]:
        mapping = adapter.component_mapping
        assert mapping is not None
        return mapping

    def test_embed_type_and_name(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_type_and_name(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert mapping["blocks"].name == "model.layers"

    def test_ln_final(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert mapping["ln_final"].name == "model.norm"

    def test_unembed(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_ln1(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_ln2(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"

    def test_attn_type_and_name(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_qkvo_names(self, adapter: Qwen3ArchitectureAdapter) -> None:
        attn = self._mapping(adapter)["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attn_qk_norms(self, adapter: Qwen3ArchitectureAdapter) -> None:
        """Qwen3-specific Q/K head norms."""
        attn = self._mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert attn.submodules["q_norm"].name == "q_norm"
        assert isinstance(attn.submodules["k_norm"], RMSNormalizationBridge)
        assert attn.submodules["k_norm"].name == "k_norm"

    def test_mlp(self, adapter: Qwen3ArchitectureAdapter) -> None:
        mlp = self._mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "mlp"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_no_linear_attn_when_dense(self, adapter: Qwen3ArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert "linear_attn" not in blocks.submodules


class TestQwen3AdapterWeightConversions:
    """
    Weights conversion tests
    """

    def test_four_conversion_keys(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert len(convs) == 4

    def test_qkvo_keys_present(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        for key in [
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        ]:
            assert key in convs

    def test_q_uses_n_heads(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_k_uses_n_key_value_heads(self, adapter: Qwen3ArchitectureAdapter) -> None:
        """GQA: K is split along n_key_value_heads, not n_heads."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.k.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_v_uses_n_key_value_heads(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.v.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_o_pattern(self, adapter: Qwen3ArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


class TestPreprocessGatedQProj:
    """
    Tests for _preprocess_gated_q_proj
    """

    def test_slices_query_half(self) -> None:
        """Interleaved [query, gate] rows per head must be reduced to query-only."""
        n_heads, d_head, d_model = 4, 8, 16
        # Build q_proj.weight as (n_heads, d_head*2, d_model): query=1.0, gate=9.0
        w = torch.empty(n_heads, d_head * 2, d_model)
        w[:, :d_head, :] = 1.0
        w[:, d_head:, :] = 9.0
        w_flat = w.reshape(n_heads * d_head * 2, d_model)

        state_dict = {"model.layers.0.self_attn.q_proj.weight": w_flat.clone()}
        out = Qwen3ArchitectureAdapter._preprocess_gated_q_proj(state_dict, n_heads, d_head)

        result = out["model.layers.0.self_attn.q_proj.weight"]
        assert result.shape == (n_heads * d_head, d_model)
        assert torch.all(result == 1.0), "gate rows must be dropped"

    def test_only_q_proj_keys_modified(self) -> None:
        n_heads, d_head, d_model = 2, 4, 8
        q_w = torch.ones(n_heads * d_head * 2, d_model)
        other = torch.full((d_model, d_model), 7.0)
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": q_w,
            "model.layers.0.self_attn.k_proj.weight": other.clone(),
            "model.layers.0.mlp.gate_proj.weight": other.clone(),
        }
        out = Qwen3ArchitectureAdapter._preprocess_gated_q_proj(state_dict, n_heads, d_head)
        assert torch.equal(out["model.layers.0.self_attn.k_proj.weight"], other)
        assert torch.equal(out["model.layers.0.mlp.gate_proj.weight"], other)

    def test_multiple_layers(self) -> None:
        n_heads, d_head, d_model = 2, 4, 8
        state_dict = {
            f"model.layers.{i}.self_attn.q_proj.weight": torch.ones(n_heads * d_head * 2, d_model)
            for i in range(3)
        }
        out = Qwen3ArchitectureAdapter._preprocess_gated_q_proj(state_dict, n_heads, d_head)
        for i in range(3):
            assert out[f"model.layers.{i}.self_attn.q_proj.weight"].shape == (
                n_heads * d_head,
                d_model,
            )


class TestQwen3FactoryRegistration:
    """Factory registeration Tests"""

    def test_factory_key_registered(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "Qwen3ForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_returns_qwen3_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        cfg.architecture = "Qwen3ForCausalLM"
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, Qwen3ArchitectureAdapter)

    def test_import_from_init(self) -> None:
        from transformer_lens.model_bridge.supported_architectures import (
            Qwen3ArchitectureAdapter as FromInit,
        )

        assert FromInit is Qwen3ArchitectureAdapter
