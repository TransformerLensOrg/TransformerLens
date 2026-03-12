"""Unit tests for LLava architecture adapter and configuration."""

import pytest

from transformer_lens.config.TransformerBridgeConfig import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)


def _make_llava_cfg(**overrides):
    """Create a TransformerBridgeConfig for LLava 1.5 7B."""
    defaults = dict(
        d_model=4096,
        d_head=128,
        n_heads=32,
        n_layers=32,
        n_ctx=4096,
        d_vocab=32064,
        architecture="LlavaForConditionalGeneration",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


class TestLlavaRegistration:
    """Test that LlavaArchitectureAdapter is properly registered."""

    def test_architecture_in_supported_architectures(self):
        assert "LlavaForConditionalGeneration" in SUPPORTED_ARCHITECTURES

    def test_architecture_maps_to_correct_adapter(self):
        assert SUPPORTED_ARCHITECTURES["LlavaForConditionalGeneration"] is LlavaArchitectureAdapter

    def test_factory_selects_correct_adapter(self):
        cfg = _make_llava_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, LlavaArchitectureAdapter)


class TestLlavaAdapterConfig:
    """Test LlavaArchitectureAdapter configuration."""

    @pytest.fixture
    def adapter(self):
        cfg = _make_llava_cfg()
        return LlavaArchitectureAdapter(cfg)

    def test_is_multimodal(self, adapter):
        assert adapter.cfg.is_multimodal is True

    def test_gated_mlp(self, adapter):
        assert adapter.cfg.gated_mlp is True

    def test_uses_rms_norm(self, adapter):
        assert adapter.cfg.uses_rms_norm is True

    def test_normalization_type(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter):
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_attn_implementation(self, adapter):
        assert adapter.cfg.attn_implementation == "eager"

    def test_has_vision_encoder_component(self, adapter):
        assert "vision_encoder" in adapter.component_mapping

    def test_has_vision_projector_component(self, adapter):
        assert "vision_projector" in adapter.component_mapping

    def test_has_language_model_components(self, adapter):
        assert "embed" in adapter.component_mapping
        assert "rotary_emb" in adapter.component_mapping
        assert "blocks" in adapter.component_mapping
        assert "ln_final" in adapter.component_mapping
        assert "unembed" in adapter.component_mapping

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

    def test_weight_processing_conversions_exist(self, adapter):
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_no_norm_offset_conversions(self, adapter):
        """LLava (LLaMA-based) should NOT have +1 norm offset like Gemma."""
        for key in adapter.weight_processing_conversions:
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key
