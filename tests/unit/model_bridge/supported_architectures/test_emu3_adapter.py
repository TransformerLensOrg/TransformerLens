"""Unit tests for Emu3ArchitectureAdapter.

Unified next-token text+image generation: a Llama-shaped decoder at
model.text_model with the VQ tokenizer riding along opaquely.
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.emu3 import (
    Emu3ArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    overrides.setdefault("n_key_value_heads", 4)
    return make_bridge_cfg("Emu3ForConditionalGeneration", **overrides)


@pytest.fixture
def adapter() -> Emu3ArchitectureAdapter:
    return Emu3ArchitectureAdapter(_make_cfg())


class TestEmu3Mapping:
    def test_text_stack_under_text_model(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.text_model.embed_tokens"
        assert mapping["blocks"].name == "model.text_model.layers"
        assert mapping["ln_final"].name == "model.text_model.norm"
        assert mapping["unembed"].name == "lm_head"
        assert adapter.cfg.is_multimodal is True

    def test_vq_tokenizer_unmapped(self, adapter):
        """Emu3VQVAE has no forward (encode/decode only); it is deliberately
        outside the mapping and reachable via original_model."""
        assert "image_tokenizer" not in adapter.component_mapping

    def test_attention_reimplemented(self, adapter):
        """Uniform Llama attention: full reimplementation, not delegation."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.submodules["q"].name == "q_proj"

    def test_qkvo_conversions_ship(self, adapter):
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["Emu3ForConditionalGeneration"] is Emu3ArchitectureAdapter
