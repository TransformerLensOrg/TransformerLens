"""Unit tests for the Florence2ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.florence2 import (
    Florence2ArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    cfg = make_bridge_cfg(
        "Florence2ForConditionalGeneration",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        default_prepend_bos=True,
    )
    cfg.encoder_layers = 2
    cfg.decoder_layers = 2
    return cfg


@pytest.fixture(scope="class")
def adapter() -> Florence2ArchitectureAdapter:
    return Florence2ArchitectureAdapter(_make_cfg())


class TestFlorence2ComponentMapping:
    def test_bart_stack_reprefixed(self, adapter):
        """The BART text stack lives under model.language_model."""
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.language_model.encoder.embed_tokens"
        assert mapping["encoder_blocks"].name == "model.language_model.encoder.layers"
        assert mapping["decoder_blocks"].name == "model.language_model.decoder.layers"
        assert mapping["unembed"].name == "lm_head"

    def test_vision_components_delegated(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["vision_encoder"].name == "model.vision_tower"
        assert mapping["vision_projector"].name == "model.multi_modal_projector"


class TestFlorence2Registration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Florence2ForConditionalGeneration"]
            is Florence2ArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="florence2", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Florence2ForConditionalGeneration"

    def test_multimodal_loader_class(self):
        from transformer_lens.utilities.architectures import MULTIMODAL_ARCHITECTURES

        assert "Florence2ForConditionalGeneration" in MULTIMODAL_ARCHITECTURES
