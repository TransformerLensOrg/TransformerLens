"""Unit tests for the M2M100ArchitectureAdapter (M2M100 / NLLB-200).

Download-free: tiny synthetic configs and structural assertions only.
"""

from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    NormalizationBridge,
    PosEmbedBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.m2m100 import (
    M2M100ArchitectureAdapter,
)


def _base_cfg() -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        architecture="M2M100ForConditionalGeneration",
    )
    cfg.encoder_layers = 2
    cfg.decoder_layers = 2
    cfg.encoder_attention_heads = 4
    cfg.decoder_attention_heads = 4
    cfg.encoder_ffn_dim = 256
    cfg.decoder_ffn_dim = 256
    return cfg


@pytest.fixture(scope="class")
def adapter() -> M2M100ArchitectureAdapter:
    return M2M100ArchitectureAdapter(_base_cfg())


def _mapping(adapter: M2M100ArchitectureAdapter) -> dict[str, Any]:
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


class TestM2M100AdapterConfig:
    def test_pre_ln_processing_guards(self, adapter):
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "standard"
        assert adapter.cfg.gated_mlp is False
        assert adapter.supports_fold_ln is False
        assert adapter.supports_center_writing_weights is False
        assert adapter.weight_processing_conversions == {}

    def test_scale_embedding_defaults_true(self, adapter):
        # Baked into M2M100ScaledWordEmbedding.forward — embed hooks see the
        # scaled output (unlike Marian, where scaling happens outside).
        assert adapter.cfg.scale_embedding is True


class TestM2M100ComponentMapping:
    # Pre-norm with extra final norms after each stack — Bart/Marian lack these.
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
        assert isinstance(mapping["decoder_ln_final"], NormalizationBridge)
        assert mapping["decoder_ln_final"].name == "model.decoder.layer_norm"

    def test_top_level_types_and_paths(self, adapter):
        mapping = _mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.encoder.embed_tokens"
        assert isinstance(mapping["pos_embed"], PosEmbedBridge)
        assert mapping["pos_embed"].name == "model.encoder.embed_positions"
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_encoder_block_submodules(self, adapter):
        block = _mapping(adapter)["encoder_blocks"]
        assert isinstance(block, BlockBridge)
        assert set(block.submodules) == {"ln1", "attn", "ln2", "mlp"}
        assert block.submodules["ln1"].name == "self_attn_layer_norm"
        assert block.submodules["ln2"].name == "final_layer_norm"
        mlp = block.submodules["mlp"]
        assert isinstance(mlp, SymbolicBridge)
        assert mlp.submodules["in"].name == "fc1"
        assert mlp.submodules["out"].name == "fc2"

    def test_decoder_block_submodules(self, adapter):
        block = _mapping(adapter)["decoder_blocks"]
        assert set(block.submodules) == {"ln1", "self_attn", "ln2", "cross_attn", "ln3", "mlp"}
        cross = block.submodules["cross_attn"]
        assert isinstance(cross, AttentionBridge)
        assert cross.name == "encoder_attn"
        assert cross.is_cross_attention is True


class TestM2M100Registration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["M2M100ForConditionalGeneration"] is M2M100ArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformers import M2M100Config

        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = M2M100Config(
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=256,
            decoder_ffn_dim=256,
        )
        assert determine_architecture_from_hf_config(cfg) == "M2M100ForConditionalGeneration"

    def test_loads_as_seq2seq(self):
        from transformers import AutoModelForSeq2SeqLM

        from transformer_lens.model_bridge.sources.transformers import (
            get_hf_model_class_for_architecture,
        )

        assert (
            get_hf_model_class_for_architecture("M2M100ForConditionalGeneration")
            is AutoModelForSeq2SeqLM
        )


class TestM2M100SymmetryGuards:
    def test_rejects_asymmetric_layers(self):
        cfg = _base_cfg()
        cfg.decoder_layers = 3
        with pytest.raises(ValueError, match="encoder_layers=2, decoder_layers=3"):
            M2M100ArchitectureAdapter(cfg)

    def test_rejects_asymmetric_ffn_dims(self):
        cfg = _base_cfg()
        cfg.decoder_ffn_dim = 512
        with pytest.raises(ValueError, match="encoder_ffn_dim=256, decoder_ffn_dim=512"):
            M2M100ArchitectureAdapter(cfg)
