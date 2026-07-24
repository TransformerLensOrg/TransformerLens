"""Unit tests for the Idefics3ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    SiglipVisionEncoderBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.supported_architectures.idefics3 import (
    Idefics3ArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    cfg = make_bridge_cfg(
        "Idefics3ForConditionalGeneration",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )
    cfg.vision_config = SimpleNamespace(hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
    return cfg


@pytest.fixture(scope="class")
def adapter() -> Idefics3ArchitectureAdapter:
    return Idefics3ArchitectureAdapter(_make_cfg())


class TestIdefics3AdapterConfig:
    def test_multimodal_flags(self, adapter):
        assert adapter.cfg.is_multimodal is True
        assert adapter.cfg.vision_hidden_size == 32
        assert adapter.cfg.vision_num_layers == 2

    def test_text_model_is_llama_shaped(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.gated_mlp is True


class TestIdefics3ComponentMapping:
    def test_vision_components(self, adapter):
        mapping = adapter.component_mapping
        vision = mapping["vision_encoder"]
        assert isinstance(vision, SiglipVisionEncoderBridge)
        assert vision.name == "model.vision_model"
        projector = mapping["vision_projector"]
        assert isinstance(projector, VisionProjectionBridge)
        assert projector.name == "model.connector"

    def test_text_model_paths_are_nested(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.text_model.embed_tokens"
        assert mapping["blocks"].name == "model.text_model.layers"
        assert mapping["ln_final"].name == "model.text_model.norm"
        # lm_head sits at the top level, unlike llava's nested language_model.
        assert mapping["unembed"].name == "lm_head"


class TestIdefics3Registration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Idefics3ForConditionalGeneration"]
            is Idefics3ArchitectureAdapter
        )

    def test_loads_via_image_text_to_text(self):
        from transformers import AutoModelForImageTextToText

        from transformer_lens.model_bridge.sources.transformers import (
            get_hf_model_class_for_architecture,
        )

        assert (
            get_hf_model_class_for_architecture("Idefics3ForConditionalGeneration")
            is AutoModelForImageTextToText
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="idefics3", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Idefics3ForConditionalGeneration"
