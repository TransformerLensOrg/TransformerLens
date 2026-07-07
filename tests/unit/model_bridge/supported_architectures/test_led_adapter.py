"""Unit tests for the LEDArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.led import (
    LEDArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        architecture="LEDForConditionalGeneration",
    )
    cfg.encoder_layers = 2
    cfg.decoder_layers = 2
    return cfg


@pytest.fixture(scope="class")
def adapter() -> LEDArchitectureAdapter:
    return LEDArchitectureAdapter(_make_cfg())


class TestLEDComponentMapping:
    def test_led_prefix(self, adapter):
        """LED's base model attribute is `led`, not `model`."""
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "led.encoder.embed_tokens"
        assert mapping["encoder_blocks"].name == "led.encoder.layers"
        assert mapping["decoder_blocks"].name == "led.decoder.layers"
        assert mapping["unembed"].name == "lm_head"

    def test_longformer_encoder_attention(self, adapter):
        """Encoder q/k/v live inside longformer_self_attn; the output
        projection is `output`, and window chunking stays HF-native."""
        attn = adapter.component_mapping["encoder_blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True
        assert attn.submodules["q"].name == "longformer_self_attn.query"
        assert attn.submodules["o"].name == "output"

    def test_decoder_is_bart_layout(self, adapter):
        dec = adapter.component_mapping["decoder_blocks"].submodules
        assert dec["self_attn"].submodules["q"].name == "q_proj"
        assert dec["cross_attn"].name == "encoder_attn"


class TestLEDRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["LEDForConditionalGeneration"] is LEDArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="led", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "LEDForConditionalGeneration"

    def test_seq2seq_loader_class(self):
        from transformer_lens.utilities.architectures import SEQ2SEQ_ARCHITECTURES

        assert "LEDForConditionalGeneration" in SEQ2SEQ_ARCHITECTURES
