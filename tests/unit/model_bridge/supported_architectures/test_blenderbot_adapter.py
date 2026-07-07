"""Unit tests for the BlenderbotArchitectureAdapter.

Download-free: tiny synthetic configs and structural assertions only.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    NormalizationBridge,
    PosEmbedBridge,
)
from transformer_lens.model_bridge.supported_architectures.blenderbot import (
    BlenderbotArchitectureAdapter,
)


def _base_cfg(encoder_layers: int = 2, decoder_layers: int = 4) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=decoder_layers,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        architecture="BlenderbotForConditionalGeneration",
    )
    cfg.encoder_layers = encoder_layers
    cfg.decoder_layers = decoder_layers
    cfg.encoder_attention_heads = 4
    cfg.decoder_attention_heads = 4
    cfg.encoder_ffn_dim = 256
    cfg.decoder_ffn_dim = 256
    return cfg


@pytest.fixture(scope="class")
def adapter() -> BlenderbotArchitectureAdapter:
    return BlenderbotArchitectureAdapter(_base_cfg())


class TestBlenderbotAsymmetricStacks:
    def test_accepts_asymmetric_layer_counts(self, adapter):
        """All public checkpoints are asymmetric (2 enc / 12+ dec)."""
        assert adapter.cfg.n_layers == 4  # follows the decoder

    def test_rejects_asymmetric_heads(self):
        cfg = _base_cfg()
        cfg.decoder_attention_heads = 8
        with pytest.raises(ValueError, match="attention heads"):
            BlenderbotArchitectureAdapter(cfg)

    def test_rejects_asymmetric_ffn(self):
        cfg = _base_cfg()
        cfg.decoder_ffn_dim = 512
        with pytest.raises(ValueError, match="FFN dims"):
            BlenderbotArchitectureAdapter(cfg)


class TestBlenderbotComponentMapping:
    def test_pegasus_layout_with_learned_positions(self, adapter):
        """Pre-norm + stack-final norms like Pegasus, but learned positions."""
        mapping = adapter.component_mapping
        assert isinstance(mapping["pos_embed"], PosEmbedBridge)
        assert isinstance(mapping["encoder_ln_final"], NormalizationBridge)
        assert mapping["encoder_ln_final"].name == "model.encoder.layer_norm"
        assert mapping["decoder_ln_final"].name == "model.decoder.layer_norm"
        assert "embed_ln" not in mapping

    def test_decoder_block_submodules(self, adapter):
        block = mapping = adapter.component_mapping["decoder_blocks"]
        assert set(block.submodules) == {"ln1", "self_attn", "ln2", "cross_attn", "ln3", "mlp"}


class TestBlenderbotRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["BlenderbotForConditionalGeneration"]
            is BlenderbotArchitectureAdapter
        )

    def test_model_type_detection(self):
        from types import SimpleNamespace

        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="blenderbot", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "BlenderbotForConditionalGeneration"
