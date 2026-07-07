"""Unit tests for the Qwen2_5_VLArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.qwen2_5_vl import (
    Qwen2_5_VLArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Qwen2_5_VLForConditionalGeneration",
    )


@pytest.fixture(scope="class")
def adapter() -> Qwen2_5_VLArchitectureAdapter:
    return Qwen2_5_VLArchitectureAdapter(_make_cfg())


class TestQwen2_5_VLComponentMapping:
    def test_text_attention_stays_native(self, adapter):
        """mRoPE splits three position streams across rotary channels; the
        generic reconstruction would be wrong for image runs."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True
        assert attn.name == "self_attn"

    def test_text_stack_nested_under_language_model(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.language_model.embed_tokens"
        assert mapping["blocks"].name == "model.language_model.layers"
        assert mapping["unembed"].name == "lm_head"
        assert adapter.cfg.is_multimodal is True

    def test_vision_tower_overrides(self, adapter):
        """Qwen2.5-VL's tower has rotary (not learned pos) and a gated MLP."""
        tower = adapter.component_mapping["vision_encoder"]
        assert tower.name == "model.visual"
        assert tower.submodules["pos_embed"].name == "rotary_pos_emb"
        block = tower.submodules["blocks"]
        assert block.submodules["mlp"].submodules["gate"].name == "gate_proj"
        assert adapter.component_mapping["vision_projector"].name == "model.visual.merger"


class TestQwen2_5_VLRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Qwen2_5_VLForConditionalGeneration"]
            is Qwen2_5_VLArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen2_5_vl", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen2_5_VLForConditionalGeneration"

    def test_multimodal_loader_class(self):
        from transformer_lens.utilities.architectures import MULTIMODAL_ARCHITECTURES

        assert "Qwen2_5_VLForConditionalGeneration" in MULTIMODAL_ARCHITECTURES
