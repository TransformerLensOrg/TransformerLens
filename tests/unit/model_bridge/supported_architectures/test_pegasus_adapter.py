"""Unit tests for the PegasusArchitectureAdapter.

Download-free: tiny synthetic configs and structural assertions only.
"""

from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    NormalizationBridge,
    PosEmbedBridge,
)
from transformer_lens.model_bridge.supported_architectures.pegasus import (
    PegasusArchitectureAdapter,
)


def _base_cfg() -> TransformerBridgeConfig:
    cfg = make_bridge_cfg(
        "PegasusForConditionalGeneration",
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
    cfg.encoder_attention_heads = 4
    cfg.decoder_attention_heads = 4
    cfg.encoder_ffn_dim = 256
    cfg.decoder_ffn_dim = 256
    return cfg


@pytest.fixture(scope="class")
def adapter() -> PegasusArchitectureAdapter:
    return PegasusArchitectureAdapter(_base_cfg())


def _mapping(adapter: PegasusArchitectureAdapter) -> dict[str, Any]:
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


class TestPegasusComponentMapping:
    # Pre-norm + per-stack final norms, sinusoidal positions, no
    # layernorm_embedding (unlike Bart/MBart).
    EXPECTED_TOP_LEVEL_KEYS = {
        "embed",
        "pos_embed",
        "encoder_blocks",
        "encoder_ln_final",
        "decoder_embed",
        "decoder_pos_embed",
        "decoder_blocks",
        "decoder_ln_final",
        "unembed",
    }

    def test_top_level_keys_are_expected(self, adapter):
        assert set(_mapping(adapter)) == self.EXPECTED_TOP_LEVEL_KEYS

    def test_stack_final_norms(self, adapter):
        mapping = _mapping(adapter)
        assert isinstance(mapping["encoder_ln_final"], NormalizationBridge)
        assert mapping["encoder_ln_final"].name == "model.encoder.layer_norm"
        assert mapping["decoder_ln_final"].name == "model.decoder.layer_norm"

    def test_sinusoidal_positions_wrapped(self, adapter):
        assert isinstance(_mapping(adapter)["pos_embed"], PosEmbedBridge)
        assert _mapping(adapter)["pos_embed"].name == "model.encoder.embed_positions"

    def test_decoder_cross_attention(self, adapter):
        block = _mapping(adapter)["decoder_blocks"]
        assert isinstance(block, BlockBridge)
        cross = block.submodules["cross_attn"]
        assert isinstance(cross, AttentionBridge)
        assert cross.name == "encoder_attn"
        assert cross.is_cross_attention is True

    def test_scale_embedding_defaults_true(self, adapter):
        # Applied in the stack forward (Marian-style) — embed hooks see
        # unscaled output.
        assert adapter.cfg.scale_embedding is True


class TestPegasusRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["PegasusForConditionalGeneration"] is PegasusArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformers import PegasusConfig

        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = PegasusConfig(
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
        )
        assert determine_architecture_from_hf_config(cfg) == "PegasusForConditionalGeneration"


class TestPegasusSymmetryGuards:
    def test_rejects_asymmetric_layers(self):
        cfg = _base_cfg()
        cfg.decoder_layers = 3
        with pytest.raises(ValueError, match="encoder_layers=2, decoder_layers=3"):
            PegasusArchitectureAdapter(cfg)
