"""Unit tests for Gemma3ArchitectureAdapter (bridge structural).

Legacy `get_pretrained_model_config` tests live in test_gemma3_config.py.
"""

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import (
    ArithmeticTensorConversion,
    RearrangeTensorConversion,
    TransposeTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.arithmetic_tensor_conversion import (
    OperationTypes,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.gemma3 import (
    Gemma3ArchitectureAdapter,
)


def _make_gemma3_cfg(**overrides):
    """TransformerBridgeConfig for Gemma3 270M (text-only)."""
    defaults = dict(
        d_model=640,
        d_head=256,
        n_heads=4,
        n_layers=18,
        n_ctx=8192,
        d_vocab=262144,
        architecture="Gemma3ForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


class TestGemma3AdapterConfig:
    """Gemma3ArchitectureAdapter cfg attributes."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3ArchitectureAdapter(_make_gemma3_cfg())

    def test_rmsnorm_uses_offset(self, adapter):
        # Gemma uses (1 + weight); offset must be advertised on cfg.
        assert adapter.cfg.rmsnorm_uses_offset is True


class TestGemma3ComponentMappingPresence:
    """Component slots must exist (deletion guard)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3ArchitectureAdapter(_make_gemma3_cfg())

    def test_no_vision_components(self, adapter):
        assert "vision_encoder" not in adapter.component_mapping
        assert "vision_projector" not in adapter.component_mapping


class TestGemma3ComponentMappingPaths:
    """HF module paths for each component slot (refactor-drift guard)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3ArchitectureAdapter(_make_gemma3_cfg())

    def test_embed_path(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path(self, adapter):
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path(self, adapter):
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter):
        assert adapter.component_mapping["unembed"].name == "lm_head"


class TestGemma3ComponentTypes:
    """Component bridge classes — guards against silent type substitution."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3ArchitectureAdapter(_make_gemma3_cfg())

    def test_embed_type(self, adapter):
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_rotary_emb_type(self, adapter):
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_blocks_type(self, adapter):
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_ln_final_type(self, adapter):
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_unembed_type(self, adapter):
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)


class TestGemma3BlockSubmodules:
    """BlockBridge wires Gemma3 dual-norm submodules."""

    @pytest.fixture(scope="class")
    def blocks(self):
        adapter = Gemma3ArchitectureAdapter(_make_gemma3_cfg())
        return adapter.component_mapping["blocks"]

    def test_dual_normalization_pre_and_post(self, blocks):
        for name in ("ln1", "ln1_post", "ln2", "ln2_post"):
            sub = blocks.submodules[name]
            assert isinstance(sub, RMSNormalizationBridge)

    def test_ln1_path(self, blocks):
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_ln1_post_path(self, blocks):
        assert blocks.submodules["ln1_post"].name == "post_attention_layernorm"

    def test_ln2_path(self, blocks):
        assert blocks.submodules["ln2"].name == "pre_feedforward_layernorm"

    def test_ln2_post_path(self, blocks):
        assert blocks.submodules["ln2_post"].name == "post_feedforward_layernorm"

    def test_attn_is_position_embeddings_attention(self, blocks):
        attn = blocks.submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"

    def test_attn_qkvo_submodule_paths(self, blocks):
        attn = blocks.submodules["attn"]
        for sub_name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "o_proj"),
        ):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path

    def test_attn_has_qk_norm_submodules(self, blocks):
        # Gemma3 specifically applies RMSNorm to Q and K inside attention.
        attn = blocks.submodules["attn"]
        for sub_name in ("q_norm", "k_norm"):
            assert sub_name in attn.submodules
            sub = attn.submodules[sub_name]
            assert isinstance(sub, RMSNormalizationBridge)
            assert sub.name == sub_name

    def test_mlp_is_gated(self, blocks):
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "mlp"

    def test_mlp_submodule_paths(self, blocks):
        mlp = blocks.submodules["mlp"]
        for sub_name, expected_path in (
            ("gate", "gate_proj"),
            ("in", "up_proj"),
            ("out", "down_proj"),
        ):
            sub = mlp.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path


class TestGemma3GQASupport:
    """n_key_value_heads must propagate to K/V conversions only."""

    def test_no_gqa_when_not_set(self):
        # Unset n_key_value_heads falls back to n_heads (the shared helper's contract).
        adapter = Gemma3ArchitectureAdapter(_make_gemma3_cfg())
        kv_conv = adapter.weight_processing_conversions["blocks.{i}.attn.k.weight"]
        assert kv_conv.tensor_conversion.axes_lengths["n"] == 4
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 4

    def test_gqa_propagates_to_kv_conversions(self):
        cfg = _make_gemma3_cfg(n_heads=8, n_key_value_heads=4)
        adapter = Gemma3ArchitectureAdapter(cfg)
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == 4

    def test_gqa_does_not_change_q_or_o_conversions(self):
        cfg = _make_gemma3_cfg(n_heads=8, n_key_value_heads=4)
        adapter = Gemma3ArchitectureAdapter(cfg)
        # Q and O always use n_heads, regardless of GQA grouping.
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 8
        assert o_conv.tensor_conversion.axes_lengths["n"] == 8


class TestGemma3WeightProcessingConversions:
    """Conversion entries have the right semantics, not just presence."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3ArchitectureAdapter(_make_gemma3_cfg())

    def test_qkvo_conversion_classes_and_patterns(self, adapter):
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(o_conv.tensor_conversion, RearrangeTensorConversion)
        assert o_conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_norm_offset_conversion_semantics(self, adapter):
        # Each norm-weight conversion must be ADDITION-by-1.0 (Gemma's +1 trick).
        for key in (
            "blocks.{i}.ln1.weight",
            "blocks.{i}.ln1_post.weight",
            "blocks.{i}.ln2.weight",
            "blocks.{i}.ln2_post.weight",
            "ln_final.weight",
            "blocks.{i}.attn.q_norm.weight",
            "blocks.{i}.attn.k_norm.weight",
        ):
            conv = adapter.weight_processing_conversions[key]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, ArithmeticTensorConversion)
            assert conv.tensor_conversion.operation == OperationTypes.ADDITION
            assert conv.tensor_conversion.value == 1.0

    def test_mlp_uses_transpose_conversion(self, adapter):
        for slot in ("gate", "in", "out"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.mlp.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, TransposeTensorConversion)

    def test_unembed_uses_transpose_conversion(self, adapter):
        conv = adapter.weight_processing_conversions["unembed.weight"]
        assert isinstance(conv.tensor_conversion, TransposeTensorConversion)

    def test_no_attention_bias_conversions(self, adapter):
        # Gemma-3 has bias=None on q/k/v/o_proj — no bias conversion keys expected.
        for key in adapter.weight_processing_conversions:
            assert not key.endswith(".bias"), f"unexpected bias key {key}"
