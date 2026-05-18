"""Unit tests for the Qwen3MoeArchitectureAdapter — programmatic configs only, no downloads."""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_moe import (
    Qwen3MoeArchitectureAdapter,
)


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=2,
        d_vocab=256,
        architecture="Qwen3MoeForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> Qwen3MoeArchitectureAdapter:
    return Qwen3MoeArchitectureAdapter(cfg)


class TestQwen3MoeAdapterConfig:
    def test_normalization_type_is_rms(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(
        self, adapter: Qwen3MoeArchitectureAdapter
    ) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_true(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """OLMoE sets final_rms=False; Qwen3MoE must not drift to that."""
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_is_true(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_uses_rms_norm_is_true(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_attn_implementation_is_eager(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.attn_implementation == "eager"

    def test_default_prepend_bos_is_false(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False

    def test_n_kv_heads_propagated(self) -> None:
        cfg = TransformerBridgeConfig(
            d_model=64,
            d_head=16,
            n_layers=2,
            n_ctx=128,
            n_heads=4,
            n_key_value_heads=2,
            d_vocab=256,
            architecture="Qwen3MoeForCausalLM",
        )
        adapter = Qwen3MoeArchitectureAdapter(cfg)
        assert adapter.cfg.n_key_value_heads == 2


class TestQwen3MoeWeightConversions:
    def test_has_qkvo_keys(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert "blocks.{i}.attn.q.weight" in convs
        assert "blocks.{i}.attn.k.weight" in convs
        assert "blocks.{i}.attn.v.weight" in convs
        assert "blocks.{i}.attn.o.weight" in convs

    def test_q_rearrange_uses_n_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        q_conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(q_conv, ParamProcessingConversion)
        assert isinstance(q_conv.tensor_conversion, RearrangeTensorConversion)
        axes = q_conv.tensor_conversion.axes_lengths
        assert axes.get("n") == 4

    def test_kv_rearrange_uses_n_kv_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """GQA: K/V follow n_key_value_heads (2), not n_heads."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        k_conv = convs["blocks.{i}.attn.k.weight"]
        v_conv = convs["blocks.{i}.attn.v.weight"]
        assert isinstance(k_conv, ParamProcessingConversion)
        assert isinstance(v_conv, ParamProcessingConversion)
        assert isinstance(k_conv.tensor_conversion, RearrangeTensorConversion)
        assert isinstance(v_conv.tensor_conversion, RearrangeTensorConversion)
        assert k_conv.tensor_conversion.axes_lengths.get("n") == 2
        assert v_conv.tensor_conversion.axes_lengths.get("n") == 2

    def test_o_rearrange_uses_n_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        o_conv = convs["blocks.{i}.attn.o.weight"]
        assert isinstance(o_conv, ParamProcessingConversion)
        assert isinstance(o_conv.tensor_conversion, RearrangeTensorConversion)
        assert o_conv.tensor_conversion.axes_lengths.get("n") == 4


class TestQwen3MoeComponentMapping:
    def test_has_required_top_level_keys(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        for key in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_blocks_has_required_submodules(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        blocks = mapping["blocks"]
        for key in ("ln1", "ln2", "attn", "mlp"):
            assert key in blocks.submodules, f"Missing blocks submodule: {key!r}"

    def test_attn_has_all_submodules(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        attn = mapping["blocks"].submodules["attn"]
        for key in ("q", "k", "v", "o", "q_norm", "k_norm"):
            assert key in attn.submodules, f"Missing attn submodule: {key!r}"

    def test_ln1_ln2_are_rms_norm_bridges(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        subs = mapping["blocks"].submodules
        assert isinstance(subs["ln1"], RMSNormalizationBridge)
        assert isinstance(subs["ln2"], RMSNormalizationBridge)

    def test_mlp_is_moe_bridge(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        mlp = mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)

    def test_mlp_has_gate_submodule(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        mlp = mapping["blocks"].submodules["mlp"]
        assert "gate" in mlp.submodules

    def test_q_norm_k_norm_are_rms_norm_bridges(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        attn_subs = mapping["blocks"].submodules["attn"].submodules
        assert isinstance(attn_subs["q_norm"], RMSNormalizationBridge)
        assert isinstance(attn_subs["k_norm"], RMSNormalizationBridge)

    def test_hf_module_paths(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"
        assert mapping["blocks"].name == "model.layers"
        subs = mapping["blocks"].submodules
        assert subs["ln1"].name == "input_layernorm"
        assert subs["ln2"].name == "post_attention_layernorm"
        assert subs["attn"].name == "self_attn"
        assert subs["mlp"].name == "mlp"


class TestQwen3MoeFactoryRegistration:
    def test_factory_lookup_returns_adapter_class(self) -> None:
        assert SUPPORTED_ARCHITECTURES["Qwen3MoeForCausalLM"] is Qwen3MoeArchitectureAdapter

    def test_factory_selects_correct_adapter(self) -> None:
        cfg = TransformerBridgeConfig(
            d_model=64,
            d_head=16,
            n_layers=2,
            n_ctx=128,
            n_heads=4,
            n_key_value_heads=2,
            d_vocab=256,
            architecture="Qwen3MoeForCausalLM",
        )
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, Qwen3MoeArchitectureAdapter)


class TestQwen3MoeComponentTypes:
    """Top-level bridge classes — guards against silent type substitution."""

    def test_embed_is_embedding_bridge(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_rotary_emb_is_rotary_bridge(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_blocks_is_block_bridge(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_ln_final_is_rms_norm_bridge(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_unembed_is_unembedding_bridge(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)


class TestQwen3MoeBlockSubmodules:
    """BlockBridge submodule types and HF paths."""

    def test_attn_is_position_embeddings_attention(
        self, adapter: Qwen3MoeArchitectureAdapter
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_attn_requires_attention_mask_and_position_embeddings(
        self, adapter: Qwen3MoeArchitectureAdapter
    ) -> None:
        """RoPE attention requires both an attention mask and position embeddings."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

    def test_attn_qkvo_submodule_types_and_paths(
        self, adapter: Qwen3MoeArchitectureAdapter
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for sub_name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "o_proj"),
        ):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path

    def test_attn_q_norm_k_norm_paths(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Per-head Q/K-norm RMSNorm."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q_norm"].name == "q_norm"
        assert attn.submodules["k_norm"].name == "k_norm"

    def test_mlp_gate_submodule_type(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Router is a LinearBridge so the routing logits can be hooked."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["gate"], LinearBridge)


class TestQwen3MoeWeightConversionPatterns:
    """Rearrange patterns on weight conversions."""

    def test_qkv_pattern_is_split_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        for slot in ("q", "k", "v"):
            conv = convs[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_o_pattern_is_merge_heads(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"


class TestQwen3MoeGQA:
    """GQA: K/V follow n_key_value_heads; Q/O always follow n_heads."""

    def test_no_gqa_fallback_to_n_heads(self) -> None:
        """Without n_key_value_heads, K/V fall back to n_heads."""
        cfg = TransformerBridgeConfig(
            d_model=64,
            d_head=16,
            n_layers=2,
            n_ctx=128,
            n_heads=4,
            d_vocab=256,
            architecture="Qwen3MoeForCausalLM",
        )
        adapter = Qwen3MoeArchitectureAdapter(cfg)
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == 4

    def test_gqa_does_not_affect_q_or_o(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 4
        assert o_conv.tensor_conversion.axes_lengths["n"] == 4


class TestQwen3MoeMoEStructure:
    """MoE structural invariants distinguishing Qwen3MoE from dense Qwen3."""

    def test_mlp_is_moe_not_gated_mlp(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert not isinstance(mlp, GatedMLPBridge)

    def test_mlp_has_only_gate_submodule(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Experts are batched 3D tensors inside the MoE block — only the router is mapped."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate"}


class TestQwen3MoeArchitectureGuards:
    """Guards against drift from Qwen3 conventions."""

    def test_no_norm_offset_conversions(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """LLaMA-style RMSNorm — no +1 offset like Gemma."""
        for key in adapter.weight_processing_conversions:
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key

    def test_weight_conversions_are_only_qkvo(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Expert/gate weights pass through untouched."""
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_attn_is_not_optional(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Non-hybrid: every layer has self_attn."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert getattr(attn, "optional", False) is False

    def test_no_linear_attn_submodule(self, adapter: Qwen3MoeArchitectureAdapter) -> None:
        """Non-hybrid: no GatedDeltaNet linear-attention submodule."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert "linear_attn" not in submodules
