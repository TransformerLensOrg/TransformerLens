"""Unit tests for Gemma3 multimodal architecture adapter registration."""

from types import SimpleNamespace

import pytest

from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
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
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CLIPVisionEncoderBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SiglipVisionEncoderBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.gemma3_multimodal import (
    Gemma3MultimodalArchitectureAdapter,
)


def _make_gemma3_mm_cfg(with_vision_config: bool = True, **overrides):
    """Create a TransformerBridgeConfig for Gemma3 4B multimodal."""
    defaults = dict(
        d_model=2560,
        d_head=256,
        n_heads=8,
        n_layers=34,
        n_ctx=8192,
        d_vocab=262208,
        n_key_value_heads=4,
        architecture="Gemma3ForConditionalGeneration",
    )
    defaults.update(overrides)
    cfg = TransformerBridgeConfig(**defaults)
    if with_vision_config:
        # Gemma3 multimodal pulls vision dims from cfg.vision_config (SigLIP).
        cfg.vision_config = SimpleNamespace(
            model_type="siglip_vision_model",
            hidden_size=1152,
            num_hidden_layers=27,
            num_attention_heads=16,
        )
    return cfg


class TestGemma3MultimodalRegistration:
    """Test that Gemma3MultimodalArchitectureAdapter is properly registered."""

    def test_architecture_in_supported_architectures(self):
        assert "Gemma3ForConditionalGeneration" in SUPPORTED_ARCHITECTURES

    def test_architecture_maps_to_correct_adapter(self):
        assert (
            SUPPORTED_ARCHITECTURES["Gemma3ForConditionalGeneration"]
            is Gemma3MultimodalArchitectureAdapter
        )

    def test_factory_selects_correct_adapter(self):
        cfg = _make_gemma3_mm_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, Gemma3MultimodalArchitectureAdapter)


class TestGemma3MultimodalAdapterConfig:
    """Test Gemma3MultimodalArchitectureAdapter configuration."""

    @pytest.fixture(scope="class")
    def adapter(self):
        cfg = _make_gemma3_mm_cfg()
        return Gemma3MultimodalArchitectureAdapter(cfg)

    def test_is_multimodal(self, adapter):
        assert adapter.cfg.is_multimodal is True

    def test_gated_mlp(self, adapter):
        assert adapter.cfg.gated_mlp is True

    def test_uses_rms_norm(self, adapter):
        assert adapter.cfg.uses_rms_norm is True

    def test_normalization_type(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"

    def test_rmsnorm_uses_offset(self, adapter):
        # Required to keep fold_ln from setting identity to 1.0 (Gemma's +1 trick).
        assert adapter.cfg.rmsnorm_uses_offset is True

    def test_positional_embedding_type(self, adapter):
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_attn_implementation_eager(self, adapter):
        assert adapter.cfg.attn_implementation == "eager"

    def test_vision_config_extracted(self, adapter):
        assert adapter.cfg.vision_hidden_size == 1152
        assert adapter.cfg.vision_num_layers == 27
        assert adapter.cfg.vision_num_heads == 16

    def test_mm_tokens_per_image_passthrough_when_set(self):
        # When cfg.mm_tokens_per_image is set, the adapter passes it through.
        cfg = _make_gemma3_mm_cfg()
        cfg.mm_tokens_per_image = 128
        adapter = Gemma3MultimodalArchitectureAdapter(cfg)
        assert adapter.cfg.mm_tokens_per_image == 128


class TestGemma3MultimodalComponentMappingPresence:
    """Top-level component slots must exist (deletion guard)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3MultimodalArchitectureAdapter(_make_gemma3_mm_cfg())

    def test_has_vision_components(self, adapter):
        assert "vision_encoder" in adapter.component_mapping
        assert "vision_projector" in adapter.component_mapping

    def test_has_language_model_components(self, adapter):
        for name in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert name in adapter.component_mapping


class TestGemma3MultimodalComponentMappingPaths:
    """HF module paths for each component slot (refactor-drift guard)."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3MultimodalArchitectureAdapter(_make_gemma3_mm_cfg())

    def test_vision_encoder_path(self, adapter):
        assert adapter.component_mapping["vision_encoder"].name == "model.vision_tower"

    def test_vision_projector_path(self, adapter):
        assert adapter.component_mapping["vision_projector"].name == "model.multi_modal_projector"

    def test_embed_path(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.language_model.embed_tokens"

    def test_rotary_emb_path(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.language_model.rotary_emb"

    def test_blocks_path(self, adapter):
        assert adapter.component_mapping["blocks"].name == "model.language_model.layers"

    def test_ln_final_path(self, adapter):
        assert adapter.component_mapping["ln_final"].name == "model.language_model.norm"

    def test_unembed_path(self, adapter):
        assert adapter.component_mapping["unembed"].name == "lm_head"


class TestGemma3MultimodalComponentTypes:
    """Component bridge classes — guards against silent type substitution."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3MultimodalArchitectureAdapter(_make_gemma3_mm_cfg())

    def test_vision_encoder_is_siglip_bridge(self, adapter):
        # Gemma3 multimodal hard-wires SigLIP — must NOT be CLIP.
        assert isinstance(adapter.component_mapping["vision_encoder"], SiglipVisionEncoderBridge)
        assert not isinstance(adapter.component_mapping["vision_encoder"], CLIPVisionEncoderBridge)

    def test_vision_projector_type(self, adapter):
        assert isinstance(adapter.component_mapping["vision_projector"], VisionProjectionBridge)

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


class TestGemma3MultimodalBlockSubmodules:
    """The BlockBridge must wire Gemma3 dual-norm submodules in the language model."""

    @pytest.fixture(scope="class")
    def blocks(self):
        adapter = Gemma3MultimodalArchitectureAdapter(_make_gemma3_mm_cfg())
        return adapter.component_mapping["blocks"]

    def test_block_has_required_submodules(self, blocks):
        for name in ("ln1", "ln1_post", "ln2", "ln2_post", "attn", "mlp"):
            assert name in blocks.submodules, f"BlockBridge missing submodule '{name}'"

    def test_dual_normalization_pre_and_post(self, blocks):
        for name in ("ln1", "ln1_post", "ln2", "ln2_post"):
            sub = blocks.submodules[name]
            assert isinstance(sub, RMSNormalizationBridge)

    def test_ln_submodule_paths(self, blocks):
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln1_post"].name == "post_attention_layernorm"
        assert blocks.submodules["ln2"].name == "pre_feedforward_layernorm"
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


class TestGemma3MultimodalGQASupport:
    """GQA variants — n_key_value_heads must propagate to K/V conversions only."""

    def test_default_4b_has_gqa(self):
        # The 4B-style fixture already has n_key_value_heads=4 and n_heads=8.
        adapter = Gemma3MultimodalArchitectureAdapter(_make_gemma3_mm_cfg())
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == 4
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 8
        assert o_conv.tensor_conversion.axes_lengths["n"] == 8

    def test_no_gqa_falls_back_to_n_heads(self):
        # Multimodal adapter uses 'or self.cfg.n_heads' fallback — None coerces.
        cfg = _make_gemma3_mm_cfg(n_heads=8, n_key_value_heads=None)
        adapter = Gemma3MultimodalArchitectureAdapter(cfg)
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == 8


class TestGemma3MultimodalWeightProcessingConversions:
    """Conversion entries are not just present — they have the right semantics."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return Gemma3MultimodalArchitectureAdapter(_make_gemma3_mm_cfg())

    def test_qkvo_conversion_classes_and_patterns(self, adapter):
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert o_conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_norm_offset_keys_present(self, adapter):
        for key in (
            "blocks.{i}.ln1.weight",
            "blocks.{i}.ln1_post.weight",
            "blocks.{i}.ln2.weight",
            "blocks.{i}.ln2_post.weight",
            "ln_final.weight",
            "blocks.{i}.attn.q_norm.weight",
            "blocks.{i}.attn.k_norm.weight",
        ):
            assert key in adapter.weight_processing_conversions, f"missing {key}"

    def test_norm_offset_conversion_semantics(self, adapter):
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
            assert isinstance(conv.tensor_conversion, ArithmeticTensorConversion)
            assert conv.tensor_conversion.operation == OperationTypes.ADDITION
            assert conv.tensor_conversion.value == 1.0

    def test_mlp_uses_transpose_conversion(self, adapter):
        for slot in ("gate", "in", "out"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.mlp.{slot}.weight"]
            assert isinstance(conv.tensor_conversion, TransposeTensorConversion)

    def test_unembed_uses_transpose_conversion(self, adapter):
        conv = adapter.weight_processing_conversions["unembed.weight"]
        assert isinstance(conv.tensor_conversion, TransposeTensorConversion)
