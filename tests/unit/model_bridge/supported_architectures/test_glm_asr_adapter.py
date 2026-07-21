"""Unit tests for the GlmAsrArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.glm_asr import (
    GlmAsrArchitectureAdapter,
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
        architecture="GlmAsrForConditionalGeneration",
    )


@pytest.fixture(scope="class")
def adapter() -> GlmAsrArchitectureAdapter:
    return GlmAsrArchitectureAdapter(_make_cfg())


class TestGlmAsrComponentMapping:
    def test_qwen2_audio_layout(self, adapter):
        """Same audio_tower / projector / language_model layout as
        Qwen2-Audio; the Llama text stack shares Qwen2's module names."""
        mapping = adapter.component_mapping
        assert mapping["audio_encoder"].name == "model.audio_tower"
        assert mapping["audio_projector"].name == "model.multi_modal_projector"
        assert mapping["embed"].name == "model.language_model.embed_tokens"
        assert mapping["unembed"].name == "lm_head"
        assert adapter.cfg.is_multimodal is True


class TestGlmAsrRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["GlmAsrForConditionalGeneration"] is GlmAsrArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="glmasr", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "GlmAsrForConditionalGeneration"

    def test_audio_text_loader_class(self):
        from transformer_lens.utilities.architectures import AUDIO_TEXT_ARCHITECTURES

        assert "GlmAsrForConditionalGeneration" in AUDIO_TEXT_ARCHITECTURES
