"""Unit tests for the Llama4MultimodalArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.llama4_multimodal import (
    Llama4MultimodalArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "Llama4ForConditionalGeneration",
        d_model=32,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=64,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> Llama4MultimodalArchitectureAdapter:
    return Llama4MultimodalArchitectureAdapter(_make_cfg())


class TestLlama4MultimodalComponentMapping:
    def test_text_stack_reprefixed(self, adapter):
        """The composite holds a full Llama4ForCausalLM at language_model."""
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "language_model.model.embed_tokens"
        assert mapping["blocks"].name == "language_model.model.layers"
        assert mapping["unembed"].name == "language_model.lm_head"
        assert adapter.cfg.is_multimodal is True

    def test_vision_components_delegated(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["vision_encoder"].name == "vision_model"
        assert mapping["vision_projector"].name == "multi_modal_projector"

    def test_attention_stays_native(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True


class TestLlama4MultimodalRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Llama4ForConditionalGeneration"]
            is Llama4MultimodalArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="llama4", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Llama4ForConditionalGeneration"

    def test_multimodal_loader_class(self):
        from transformer_lens.utilities.architectures import MULTIMODAL_ARCHITECTURES

        assert "Llama4ForConditionalGeneration" in MULTIMODAL_ARCHITECTURES
