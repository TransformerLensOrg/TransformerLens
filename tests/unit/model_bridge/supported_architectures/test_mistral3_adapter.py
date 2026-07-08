"""Unit tests for the Mistral3ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.vision_projection import (
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.supported_architectures.mistral3 import (
    Mistral3ArchitectureAdapter,
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
        architecture="Mistral3ForConditionalGeneration",
    )


@pytest.fixture(scope="class")
def adapter() -> Mistral3ArchitectureAdapter:
    return Mistral3ArchitectureAdapter(_make_cfg())


class TestMistral3ComponentMapping:
    def test_vision_components_delegated(self, adapter):
        """Pixtral's 2D-RoPE attention has no CLIP/SigLIP-shaped bridge; the
        projector inherits Llava's VisionProjectionBridge, whose *args
        passthrough carries the extra image_sizes positional."""
        mapping = adapter.component_mapping
        assert type(mapping["vision_encoder"]) is GeneralizedComponent
        assert mapping["vision_encoder"].name == "model.vision_tower"
        assert isinstance(mapping["vision_projector"], VisionProjectionBridge)
        assert mapping["vision_projector"].name == "model.multi_modal_projector"

    def test_llava_text_stack(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.language_model.embed_tokens"
        assert mapping["blocks"].name == "model.language_model.layers"
        assert mapping["unembed"].name == "lm_head"
        assert adapter.cfg.is_multimodal is True


class TestMistral3Registration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Mistral3ForConditionalGeneration"]
            is Mistral3ArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="mistral3", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Mistral3ForConditionalGeneration"

    def test_multimodal_loader_class(self):
        from transformer_lens.utilities.architectures import MULTIMODAL_ARCHITECTURES

        assert "Mistral3ForConditionalGeneration" in MULTIMODAL_ARCHITECTURES
