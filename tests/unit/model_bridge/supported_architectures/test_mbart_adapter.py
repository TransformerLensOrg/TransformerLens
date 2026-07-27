"""Unit tests for the MBartArchitectureAdapter.

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
)
from transformer_lens.model_bridge.supported_architectures.mbart import (
    MBartArchitectureAdapter,
)


def _base_cfg() -> TransformerBridgeConfig:
    cfg = make_bridge_cfg(
        "MBartForConditionalGeneration",
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
def adapter() -> MBartArchitectureAdapter:
    return MBartArchitectureAdapter(_base_cfg())


def _mapping(adapter: MBartArchitectureAdapter) -> dict[str, Any]:
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


class TestMBartComponentMapping:
    # MBart combines Bart's layernorm_embedding with M2M100's pre-norm layout
    # and per-stack final norms.
    EXPECTED_TOP_LEVEL_KEYS = {
        "embed",
        "pos_embed",
        "embed_ln",
        "encoder_blocks",
        "encoder_ln_final",
        "decoder_embed",
        "decoder_pos_embed",
        "decoder_embed_ln",
        "decoder_blocks",
        "decoder_ln_final",
        "unembed",
    }

    def test_top_level_keys_are_expected(self, adapter):
        assert set(_mapping(adapter)) == self.EXPECTED_TOP_LEVEL_KEYS

    def test_embedding_and_stack_norms(self, adapter):
        mapping = _mapping(adapter)
        assert mapping["embed_ln"].name == "model.encoder.layernorm_embedding"
        assert mapping["decoder_embed_ln"].name == "model.decoder.layernorm_embedding"
        assert mapping["encoder_ln_final"].name == "model.encoder.layer_norm"
        assert mapping["decoder_ln_final"].name == "model.decoder.layer_norm"
        for key in ("embed_ln", "decoder_embed_ln", "encoder_ln_final", "decoder_ln_final"):
            assert isinstance(mapping[key], NormalizationBridge)

    def test_encoder_block_submodules(self, adapter):
        block = _mapping(adapter)["encoder_blocks"]
        assert isinstance(block, BlockBridge)
        assert set(block.submodules) == {"ln1", "attn", "ln2", "mlp"}
        assert block.submodules["ln1"].name == "self_attn_layer_norm"
        assert block.submodules["ln2"].name == "final_layer_norm"

    def test_decoder_cross_attention(self, adapter):
        block = _mapping(adapter)["decoder_blocks"]
        assert set(block.submodules) == {"ln1", "self_attn", "ln2", "cross_attn", "ln3", "mlp"}
        cross = block.submodules["cross_attn"]
        assert isinstance(cross, AttentionBridge)
        assert cross.name == "encoder_attn"
        assert cross.is_cross_attention is True


class TestMBartRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["MBartForConditionalGeneration"] is MBartArchitectureAdapter

    def test_model_type_detection(self):
        from transformers import MBartConfig

        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = MBartConfig(
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
        )
        assert determine_architecture_from_hf_config(cfg) == "MBartForConditionalGeneration"


class TestMBartSymmetryGuards:
    def test_rejects_asymmetric_layers(self):
        cfg = _base_cfg()
        cfg.decoder_layers = 3
        with pytest.raises(ValueError, match="encoder_layers=2, decoder_layers=3"):
            MBartArchitectureAdapter(cfg)
