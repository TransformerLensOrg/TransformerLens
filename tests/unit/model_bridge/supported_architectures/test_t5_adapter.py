"""Unit tests for the T5ArchitectureAdapter (download-free, tiny programmatic configs,
no real checkpoints).

Covered:
- Adapter config defaults (RMSNorm, relative position bias, fold-LN disabled, gated
  MLP default off; gated variant when `cfg.is_gated_act` is set).
- Component-mapping structure, bridge types, and HF module paths for both the encoder
  and decoder trees, including the relative-attention-bias wiring.
- Encoder block: self-attention with q/k/v/o + relative position bias, plain MLP with
  wi/wo (or gated variant in Flan-T5).
- Decoder block: self-attention + cross-attention + three layer norms; cross-attention
  is flagged with `is_cross_attention=True`.
- Gated MLP variant: GatedMLPBridge with wi_0/wi_1/wo for both encoder and decoder
  when `cfg.is_gated_act` is True.
- Factory registration: T5 is dual-registered under both `T5ForConditionalGeneration`
  and `MT5ForConditionalGeneration`; both dispatch back to the same adapter.
- Architecture guards: encoder has no cross_attn / no ln3; weight conversions stay
  empty.
"""

from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MLPBridge,
    PosEmbedBridge,
    RMSNormalizationBridge,
    T5BlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.t5 import (
    T5ArchitectureAdapter,
)


def _base_cfg(*, architecture: str = "T5ForConditionalGeneration") -> TransformerBridgeConfig:
    # Keep dimensions tiny so adapter tests need no HF downloads or real checkpoints.
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_vocab=256,
        architecture=architecture,
    )


def _gated_cfg() -> TransformerBridgeConfig:
    """Config for the Flan-T5 (gated FFN) variant: the adapter reads `cfg.is_gated_act`."""
    cfg = _base_cfg()
    cfg.is_gated_act = True
    return cfg


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _base_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> T5ArchitectureAdapter:
    return T5ArchitectureAdapter(cfg)


@pytest.fixture(scope="class")
def gated_adapter() -> T5ArchitectureAdapter:
    return T5ArchitectureAdapter(_gated_cfg())


def _mapping(adapter: T5ArchitectureAdapter) -> dict:
    """Narrow component_mapping (Optional on the base class) to a non-None dict.

    Factored into a helper so each test stays a one-liner instead of repeating the
    `assert ... is not None` prelude in every method.
    """
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


def _conversions(adapter: T5ArchitectureAdapter) -> dict:
    """weight_processing_conversions is Optional on the base class; assert it is populated.

    For T5 it is populated as an empty dict (no QKVO rearranges, since T5 stores
    Q/K/V/O per-head). That emptiness is itself a load-bearing invariant; see
    TestT5ArchitectureGuards.
    """
    conversions = adapter.weight_processing_conversions
    assert conversions is not None
    return conversions


def _encoder_block(adapter: T5ArchitectureAdapter) -> Any:
    return _mapping(adapter)["encoder_blocks"]


def _decoder_block(adapter: T5ArchitectureAdapter) -> Any:
    return _mapping(adapter)["decoder_blocks"]


class TestT5ComponentMapping:
    """Top-level structure of the encoder-decoder component mapping."""

    EXPECTED_TOP_LEVEL_KEYS = {
        "embed",
        "pos_embed",
        "encoder_blocks",
        "encoder_ln_final",
        "decoder_pos_embed",
        "decoder_blocks",
        "decoder_ln_final",
        "unembed",
    }

    def test_top_level_keys_are_exactly_expected(self, adapter: T5ArchitectureAdapter) -> None:
        assert set(_mapping(adapter).keys()) == self.EXPECTED_TOP_LEVEL_KEYS

    def test_encoder_block_submodules(self, adapter: T5ArchitectureAdapter) -> None:
        """Encoder block: pre-norm, single self-attention, FFN (two layer norms total)."""
        assert set(_encoder_block(adapter).submodules.keys()) == {"ln1", "attn", "ln2", "mlp"}

    def test_decoder_block_submodules(self, adapter: T5ArchitectureAdapter) -> None:
        """Decoder block: self-attention, cross-attention, FFN (three layer norms total)."""
        assert set(_decoder_block(adapter).submodules.keys()) == {
            "ln1",
            "self_attn",
            "ln2",
            "cross_attn",
            "ln3",
            "mlp",
        }


class TestT5TopLevelComponentTypes:
    """Top-level bridge classes, guarding against silent type substitution."""

    def test_embed_is_embedding_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["embed"], EmbeddingBridge)

    def test_pos_embed_is_pos_embed_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["pos_embed"], PosEmbedBridge)

    def test_encoder_blocks_is_t5_block_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_blocks"], T5BlockBridge)

    def test_encoder_ln_final_is_rms_norm(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_ln_final"], RMSNormalizationBridge)

    def test_decoder_pos_embed_is_pos_embed_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["decoder_pos_embed"], PosEmbedBridge)

    def test_decoder_blocks_is_t5_block_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["decoder_blocks"], T5BlockBridge)

    def test_decoder_ln_final_is_rms_norm(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["decoder_ln_final"], RMSNormalizationBridge)

    def test_unembed_is_unembedding_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["unembed"], UnembeddingBridge)


class TestT5HFModulePaths:
    """Top-level HF module paths (where each bridge attaches into the HF model tree)."""

    def test_top_level_paths(self, adapter: T5ArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert mapping["embed"].name == "shared"
        assert mapping["encoder_blocks"].name == "encoder.block"
        assert mapping["encoder_ln_final"].name == "encoder.final_layer_norm"
        assert mapping["decoder_blocks"].name == "decoder.block"
        assert mapping["decoder_ln_final"].name == "decoder.final_layer_norm"
        assert mapping["unembed"].name == "lm_head"

    def test_relative_attention_bias_paths(self, adapter: T5ArchitectureAdapter) -> None:
        """Position information lives on `block.0.layer.0.SelfAttention.relative_attention_bias`
        in each stack: HF stores the learned bias on the first block of each stack."""
        mapping = _mapping(adapter)
        assert (
            mapping["pos_embed"].name
            == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias"
        )
        assert (
            mapping["decoder_pos_embed"].name
            == "decoder.block.0.layer.0.SelfAttention.relative_attention_bias"
        )


class TestT5EncoderBlock:
    """Encoder block submodules: types, names, and per-attention contract flags."""

    def test_is_decoder_flag_is_false(self, adapter: T5ArchitectureAdapter) -> None:
        assert _encoder_block(adapter).is_decoder is False

    def test_ln1_ln2_are_rms_norm(self, adapter: T5ArchitectureAdapter) -> None:
        subs = _encoder_block(adapter).submodules
        assert isinstance(subs["ln1"], RMSNormalizationBridge)
        assert isinstance(subs["ln2"], RMSNormalizationBridge)

    def test_ln_hf_paths(self, adapter: T5ArchitectureAdapter) -> None:
        subs = _encoder_block(adapter).submodules
        assert subs["ln1"].name == "layer.0.layer_norm"
        assert subs["ln2"].name == "layer.1.layer_norm"

    def test_self_attn_is_plain_attention_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        """Encoder uses AttentionBridge, not PositionEmbeddingsAttentionBridge, because
        T5 supplies position information via relative bias rather than RoPE."""
        attn = _encoder_block(adapter).submodules["attn"]
        assert isinstance(attn, AttentionBridge)
        assert attn.name == "layer.0.SelfAttention"

    def test_self_attn_requires_relative_position_bias(
        self, adapter: T5ArchitectureAdapter
    ) -> None:
        attn = _encoder_block(adapter).submodules["attn"]
        assert attn.requires_relative_position_bias is True

    def test_self_attn_is_not_cross_attention(self, adapter: T5ArchitectureAdapter) -> None:
        """Encoder self-attention must not be flagged as cross-attention."""
        attn = _encoder_block(adapter).submodules["attn"]
        assert getattr(attn, "is_cross_attention", False) is False

    def test_self_attn_qkvo_submodule_types_and_paths(self, adapter: T5ArchitectureAdapter) -> None:
        """T5 uses single-letter projection names (q, k, v, o)."""
        attn = _encoder_block(adapter).submodules["attn"]
        for sub_name, expected_path in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path

    def test_mlp_is_plain_mlp_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        """Without `is_gated_act`, the encoder FFN is T5DenseReluDense with wi/wo."""
        mlp = _encoder_block(adapter).submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "layer.1.DenseReluDense"
        assert set(mlp.submodules.keys()) == {"in", "out"}
        assert mlp.submodules["in"].name == "wi"
        assert mlp.submodules["out"].name == "wo"


class TestT5DecoderBlock:
    """Decoder block submodules: self-attn + cross-attn + three layer norms."""

    def test_is_decoder_flag_is_true(self, adapter: T5ArchitectureAdapter) -> None:
        assert _decoder_block(adapter).is_decoder is True

    def test_ln1_ln2_ln3_are_rms_norm(self, adapter: T5ArchitectureAdapter) -> None:
        subs = _decoder_block(adapter).submodules
        assert isinstance(subs["ln1"], RMSNormalizationBridge)
        assert isinstance(subs["ln2"], RMSNormalizationBridge)
        assert isinstance(subs["ln3"], RMSNormalizationBridge)

    def test_ln_hf_paths(self, adapter: T5ArchitectureAdapter) -> None:
        subs = _decoder_block(adapter).submodules
        assert subs["ln1"].name == "layer.0.layer_norm"
        assert subs["ln2"].name == "layer.1.layer_norm"
        assert subs["ln3"].name == "layer.2.layer_norm"

    def test_self_attn_is_attention_bridge_requiring_relative_bias(
        self, adapter: T5ArchitectureAdapter
    ) -> None:
        self_attn = _decoder_block(adapter).submodules["self_attn"]
        assert isinstance(self_attn, AttentionBridge)
        assert self_attn.name == "layer.0.SelfAttention"
        assert self_attn.requires_relative_position_bias is True
        assert getattr(self_attn, "is_cross_attention", False) is False

    def test_self_attn_qkvo_submodule_types_and_paths(self, adapter: T5ArchitectureAdapter) -> None:
        self_attn = _decoder_block(adapter).submodules["self_attn"]
        for sub_name, expected_path in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")):
            sub = self_attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path

    def test_cross_attn_is_attention_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        cross_attn = _decoder_block(adapter).submodules["cross_attn"]
        assert isinstance(cross_attn, AttentionBridge)
        assert cross_attn.name == "layer.1.EncDecAttention"

    def test_cross_attn_is_flagged_as_cross_attention(self, adapter: T5ArchitectureAdapter) -> None:
        """The cross-attention bridge must be flagged so the bridge forward routes
        encoder hidden states into K/V instead of the residual stream."""
        cross_attn = _decoder_block(adapter).submodules["cross_attn"]
        assert cross_attn.is_cross_attention is True

    def test_cross_attn_requires_relative_position_bias(
        self, adapter: T5ArchitectureAdapter
    ) -> None:
        cross_attn = _decoder_block(adapter).submodules["cross_attn"]
        assert cross_attn.requires_relative_position_bias is True

    def test_cross_attn_qkvo_submodule_types_and_paths(
        self, adapter: T5ArchitectureAdapter
    ) -> None:
        cross_attn = _decoder_block(adapter).submodules["cross_attn"]
        for sub_name, expected_path in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")):
            sub = cross_attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path

    def test_mlp_is_plain_mlp_bridge(self, adapter: T5ArchitectureAdapter) -> None:
        """Without `is_gated_act`, the decoder FFN is T5DenseReluDense with wi/wo."""
        mlp = _decoder_block(adapter).submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "layer.2.DenseReluDense"
        assert set(mlp.submodules.keys()) == {"in", "out"}
        assert mlp.submodules["in"].name == "wi"
        assert mlp.submodules["out"].name == "wo"


class TestT5GatedMLPVariant:
    """Flan-T5 (`cfg.is_gated_act = True`) swaps both encoder and decoder MLPs to the
    gated variant: GatedMLPBridge with `wi_0` (gate), `wi_1` (in), `wo` (out)."""

    def test_encoder_mlp_is_gated(self, gated_adapter: T5ArchitectureAdapter) -> None:
        mlp = _encoder_block(gated_adapter).submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "layer.1.DenseReluDense"

    def test_encoder_mlp_submodules_and_paths(self, gated_adapter: T5ArchitectureAdapter) -> None:
        mlp = _encoder_block(gated_adapter).submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}
        assert mlp.submodules["gate"].name == "wi_0"
        assert mlp.submodules["in"].name == "wi_1"
        assert mlp.submodules["out"].name == "wo"

    def test_decoder_mlp_is_gated(self, gated_adapter: T5ArchitectureAdapter) -> None:
        mlp = _decoder_block(gated_adapter).submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "layer.2.DenseReluDense"

    def test_decoder_mlp_submodules_and_paths(self, gated_adapter: T5ArchitectureAdapter) -> None:
        mlp = _decoder_block(gated_adapter).submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}
        assert mlp.submodules["gate"].name == "wi_0"
        assert mlp.submodules["in"].name == "wi_1"
        assert mlp.submodules["out"].name == "wo"


class TestT5FactoryRegistration:
    """T5 is dual-registered under both T5ForConditionalGeneration and
    MT5ForConditionalGeneration; both dispatch back to the same adapter class."""

    def test_mt5_factory_lookup_returns_adapter_class(self) -> None:
        """MT5 shares the T5 architecture wiring, so the factory reuses the adapter."""
        assert SUPPORTED_ARCHITECTURES["MT5ForConditionalGeneration"] is T5ArchitectureAdapter

    def test_t5_with_lm_head_alias_returns_adapter_class(self) -> None:
        """Old google-t5 checkpoints (t5-3b, t5-11b) carry the legacy class name."""
        assert SUPPORTED_ARCHITECTURES["T5WithLMHeadModel"] is T5ArchitectureAdapter

    def test_t5_with_lm_head_alias_loads_as_seq2seq(self) -> None:
        from transformers import AutoModelForSeq2SeqLM

        from transformer_lens.model_bridge.sources.transformers import (
            get_hf_model_class_for_architecture,
        )

        assert get_hf_model_class_for_architecture("T5WithLMHeadModel") is AutoModelForSeq2SeqLM


class TestT5ArchitectureGuards:
    """Guards against drift from the T5 encoder-decoder contract."""

    def test_weight_conversions_stay_empty(self, adapter: T5ArchitectureAdapter) -> None:
        """T5 stores Q/K/V/O per-head, so no rearrange conversions should ever be added."""
        assert _conversions(adapter) == {}
