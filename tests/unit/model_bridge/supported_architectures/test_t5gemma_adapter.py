"""Unit tests for T5GemmaArchitectureAdapter.

Covered:
- Config flags: normalization_type, positional_embedding_type, gated_mlp, rmsnorm_uses_offset,
  fold_ln disabled.
- Component-mapping top-level keys: encoder/decoder embed, rotary_emb, encoder_blocks /
  decoder_blocks (standard Bridge names), final norms, unembed (lm_head.out_proj).
- Encoder block (encoder_blocks): BlockBridge with Gemma-style pre/post norms, RoPE attention,
  gated MLP; no cross_attn.
- Decoder block (decoder_blocks): T5GemmaDecoderBlockBridge with self_attn, cross_attn (flagged
  is_cross_attention=True), three ln pairs, gated MLP; intermediate hook points present.
- Attention projections: q_proj/k_proj/v_proj/o_proj names.
- MLP projections: gate_proj/up_proj/down_proj names.
- lm_head path resolves to lm_head.out_proj (not lm_head directly).
- RMSNorm offset flag: rmsnorm_uses_offset is True.
- Factory registration: T5GemmaForConditionalGeneration in SUPPORTED_ARCHITECTURES.
- Architecture classification: T5Gemma is in SEQ2SEQ_ARCHITECTURES.
- Model registry: T5GemmaForConditionalGeneration in HF_SUPPORTED_ARCHITECTURES.
- Config mapping: nested decoder sub-config extracts d_model, n_heads, n_layers correctly.
- Key translation: convert_hf_key_to_tl_key maps representative HF keys to canonical TL keys.
- Conversion table: weight_processing_conversions contains the exact keys that key translation
  emits (with {i} placeholder), verifying conversions will actually fire.
"""

from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.t5gemma_decoder_block import (
    T5GemmaDecoderBlockBridge,
)
from transformer_lens.model_bridge.supported_architectures.t5gemma import (
    T5GemmaArchitectureAdapter,
)
from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES
from transformer_lens.utilities.architectures import SEQ2SEQ_ARCHITECTURES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_cfg(*, architecture: str = "T5GemmaForConditionalGeneration") -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=2,
        d_vocab=256,
        architecture=architecture,
    )


@pytest.fixture(scope="module")
def cfg() -> TransformerBridgeConfig:
    return _base_cfg()


@pytest.fixture(scope="module")
def adapter(cfg: TransformerBridgeConfig) -> T5GemmaArchitectureAdapter:
    return T5GemmaArchitectureAdapter(cfg)


def _mapping(adapter: T5GemmaArchitectureAdapter) -> dict:
    m = adapter.component_mapping
    assert m is not None
    return m


def _enc_block(adapter: T5GemmaArchitectureAdapter) -> Any:
    return _mapping(adapter)["encoder_blocks"]


def _dec_block(adapter: T5GemmaArchitectureAdapter) -> Any:
    return _mapping(adapter)["decoder_blocks"]


# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------


class TestT5GemmaAdapterConfig:
    def test_normalization_type_is_rms(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp_is_true(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_rmsnorm_uses_offset_is_true(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert adapter.cfg.rmsnorm_uses_offset is True

    def test_fold_ln_disabled(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert adapter.supports_fold_ln is False

    def test_final_rms_is_true(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True


# ---------------------------------------------------------------------------
# Top-level mapping structure
# ---------------------------------------------------------------------------


class TestT5GemmaComponentMapping:
    EXPECTED_KEYS = {
        "encoder_embed",
        "encoder_rotary_emb",
        "encoder_blocks",
        "encoder_ln_final",
        "decoder_embed",
        "decoder_rotary_emb",
        "decoder_blocks",
        "decoder_ln_final",
        "unembed",
    }

    def test_top_level_keys(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert set(_mapping(adapter).keys()) == self.EXPECTED_KEYS

    def test_standard_block_names_used(self, adapter: T5GemmaArchitectureAdapter) -> None:
        # Bridge _setup_hook_compatibility only visits encoder_blocks / decoder_blocks.
        # Using enc_blocks / dec_blocks would silently skip hook setup.
        keys = set(_mapping(adapter).keys())
        assert "encoder_blocks" in keys, "must use 'encoder_blocks', not 'enc_blocks'"
        assert "decoder_blocks" in keys, "must use 'decoder_blocks', not 'dec_blocks'"
        assert "enc_blocks" not in keys
        assert "dec_blocks" not in keys

    def test_encoder_embed_is_embedding_bridge(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_embed"], EmbeddingBridge)

    def test_encoder_embed_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_embed"].name == "model.encoder.embed_tokens"

    def test_encoder_rotary_emb_is_rotary_bridge(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_rotary_emb"], RotaryEmbeddingBridge)

    def test_encoder_rotary_emb_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_rotary_emb"].name == "model.encoder.rotary_emb"

    def test_encoder_blocks_is_block_bridge(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_blocks"], BlockBridge)

    def test_encoder_blocks_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_blocks"].name == "model.encoder.layers"

    def test_encoder_ln_final_is_rms_norm(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_ln_final"], RMSNormalizationBridge)

    def test_encoder_ln_final_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_ln_final"].name == "model.encoder.norm"

    def test_decoder_embed_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_embed"].name == "model.decoder.embed_tokens"

    def test_decoder_rotary_emb_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_rotary_emb"].name == "model.decoder.rotary_emb"

    def test_decoder_blocks_is_t5gemma_decoder_block_bridge(
        self, adapter: T5GemmaArchitectureAdapter
    ) -> None:
        assert isinstance(_mapping(adapter)["decoder_blocks"], T5GemmaDecoderBlockBridge)

    def test_decoder_blocks_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_blocks"].name == "model.decoder.layers"

    def test_decoder_ln_final_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_ln_final"].name == "model.decoder.norm"

    def test_unembed_is_unembedding_bridge(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["unembed"], UnembeddingBridge)

    def test_unembed_points_to_lm_head_out_proj(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _mapping(adapter)["unembed"].name == "lm_head.out_proj"


# ---------------------------------------------------------------------------
# Encoder block submodules
# ---------------------------------------------------------------------------


class TestT5GemmaEncoderBlock:
    EXPECTED_ENC_KEYS = {"ln1", "ln1_post", "attn", "ln2", "ln2_post", "mlp"}

    def test_encoder_submodule_keys(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert set(_enc_block(adapter).submodules.keys()) == self.EXPECTED_ENC_KEYS

    def test_encoder_has_no_cross_attn(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert "cross_attn" not in _enc_block(adapter).submodules

    def test_encoder_ln1_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _enc_block(adapter).submodules["ln1"].name == "pre_self_attn_layernorm"

    def test_encoder_ln1_post_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _enc_block(adapter).submodules["ln1_post"].name == "post_self_attn_layernorm"

    def test_encoder_ln2_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _enc_block(adapter).submodules["ln2"].name == "pre_feedforward_layernorm"

    def test_encoder_ln2_post_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _enc_block(adapter).submodules["ln2_post"].name == "post_feedforward_layernorm"

    def test_encoder_attn_is_position_embeddings_bridge(
        self, adapter: T5GemmaArchitectureAdapter
    ) -> None:
        assert isinstance(_enc_block(adapter).submodules["attn"], PositionEmbeddingsAttentionBridge)

    def test_encoder_attn_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _enc_block(adapter).submodules["attn"].name == "self_attn"

    def test_encoder_attn_qkvo_paths(self, adapter: T5GemmaArchitectureAdapter) -> None:
        subs = _enc_block(adapter).submodules["attn"].submodules
        assert subs["q"].name == "q_proj"
        assert subs["k"].name == "k_proj"
        assert subs["v"].name == "v_proj"
        assert subs["o"].name == "o_proj"

    def test_encoder_mlp_is_gated(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert isinstance(_enc_block(adapter).submodules["mlp"], GatedMLPBridge)

    def test_encoder_mlp_proj_paths(self, adapter: T5GemmaArchitectureAdapter) -> None:
        subs = _enc_block(adapter).submodules["mlp"].submodules
        assert subs["gate"].name == "gate_proj"
        assert subs["in"].name == "up_proj"
        assert subs["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# Decoder block submodules
# ---------------------------------------------------------------------------


class TestT5GemmaDecoderBlock:
    EXPECTED_DEC_KEYS = {
        "ln1",
        "ln1_post",
        "self_attn",
        "ln2",
        "ln2_post",
        "cross_attn",
        "ln3",
        "ln3_post",
        "mlp",
    }

    def test_decoder_submodule_keys(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert set(_dec_block(adapter).submodules.keys()) == self.EXPECTED_DEC_KEYS

    def test_decoder_self_attn_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["self_attn"].name == "self_attn"

    def test_decoder_cross_attn_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["cross_attn"].name == "cross_attn"

    def test_decoder_cross_attn_is_flagged(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["cross_attn"].is_cross_attention is True

    def test_decoder_self_attn_is_not_cross_attention(
        self, adapter: T5GemmaArchitectureAdapter
    ) -> None:
        assert (
            getattr(_dec_block(adapter).submodules["self_attn"], "is_cross_attention", False)
            is False
        )

    def test_decoder_ln1_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["ln1"].name == "pre_self_attn_layernorm"

    def test_decoder_ln2_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["ln2"].name == "pre_cross_attn_layernorm"

    def test_decoder_ln3_hf_path(self, adapter: T5GemmaArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["ln3"].name == "pre_feedforward_layernorm"

    def test_decoder_self_attn_qkvo_paths(self, adapter: T5GemmaArchitectureAdapter) -> None:
        subs = _dec_block(adapter).submodules["self_attn"].submodules
        assert subs["q"].name == "q_proj"
        assert subs["k"].name == "k_proj"
        assert subs["v"].name == "v_proj"
        assert subs["o"].name == "o_proj"

    def test_decoder_cross_attn_qkvo_paths(self, adapter: T5GemmaArchitectureAdapter) -> None:
        subs = _dec_block(adapter).submodules["cross_attn"].submodules
        assert subs["q"].name == "q_proj"
        assert subs["k"].name == "k_proj"
        assert subs["v"].name == "v_proj"
        assert subs["o"].name == "o_proj"

    def test_decoder_mlp_proj_paths(self, adapter: T5GemmaArchitectureAdapter) -> None:
        subs = _dec_block(adapter).submodules["mlp"].submodules
        assert subs["gate"].name == "gate_proj"
        assert subs["in"].name == "up_proj"
        assert subs["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# T5GemmaDecoderBlockBridge intermediate hook points
# ---------------------------------------------------------------------------


class TestT5GemmaDecoderBlockHookPoints:
    def test_hook_resid_mid_exists(self, adapter: T5GemmaArchitectureAdapter) -> None:
        dec = _dec_block(adapter)
        assert hasattr(dec, "hook_resid_mid")

    def test_hook_resid_mid2_exists(self, adapter: T5GemmaArchitectureAdapter) -> None:
        dec = _dec_block(adapter)
        assert hasattr(dec, "hook_resid_mid2")

    def test_encoder_block_has_no_hook_resid_mid(self, adapter: T5GemmaArchitectureAdapter) -> None:
        enc = _enc_block(adapter)
        assert not hasattr(enc, "hook_resid_mid")


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------


class TestT5GemmaFactoryRegistration:
    def test_registered_in_supported_architectures(self) -> None:
        assert "T5GemmaForConditionalGeneration" in SUPPORTED_ARCHITECTURES

    def test_maps_to_t5gemma_adapter(self) -> None:
        assert (
            SUPPORTED_ARCHITECTURES["T5GemmaForConditionalGeneration"] is T5GemmaArchitectureAdapter
        )


# ---------------------------------------------------------------------------
# Architecture classification
# ---------------------------------------------------------------------------


class TestT5GemmaArchitectureClassification:
    def test_is_in_seq2seq_architectures(self) -> None:
        assert "T5GemmaForConditionalGeneration" in SEQ2SEQ_ARCHITECTURES


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestT5GemmaModelRegistry:
    def test_is_in_hf_supported_architectures(self) -> None:
        assert "T5GemmaForConditionalGeneration" in HF_SUPPORTED_ARCHITECTURES


# ---------------------------------------------------------------------------
# Config mapping from nested HF T5GemmaConfig
# ---------------------------------------------------------------------------


class TestT5GemmaConfigMapping:
    def test_config_mapping_extracts_decoder_dims(self) -> None:
        """map_default_transformer_lens_config must read from the nested decoder sub-config."""
        try:
            from transformers import T5GemmaConfig

            from transformer_lens.model_bridge.sources.transformers import (
                map_default_transformer_lens_config,
            )
        except ImportError:
            pytest.skip("transformers not installed")

        enc_cfg = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "rms_norm_eps": 1e-6,
            "dropout_rate": 0.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "layer_types": ["full_attention", "full_attention"],
            "head_dim": 16,
            "rope_theta": 10000,
            "rope_scaling": None,
            "max_position_embeddings": 128,
            "attention_bias": False,
            "vocab_size": 256,
            "pad_token_id": 0,
        }
        dec_cfg = dict(enc_cfg, is_decoder=True)
        hf_config = T5GemmaConfig(encoder=enc_cfg, decoder=dec_cfg)
        mapped = map_default_transformer_lens_config(hf_config)

        assert mapped.d_model == 64
        assert mapped.n_heads == 4
        assert mapped.n_layers == 2
        assert mapped.d_vocab == 256

    def test_architecture_detected_from_model_type(self) -> None:
        """determine_architecture_from_hf_config must map model_type 't5gemma'."""
        try:
            from transformers import T5GemmaConfig

            from transformer_lens.model_bridge.sources.transformers import (
                determine_architecture_from_hf_config,
            )
        except ImportError:
            pytest.skip("transformers not installed")

        enc_cfg = {
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "rms_norm_eps": 1e-6,
            "dropout_rate": 0.0,
            "hidden_activation": "gelu_pytorch_tanh",
            "layer_types": ["full_attention", "full_attention"],
            "head_dim": 16,
            "rope_theta": 10000,
            "rope_scaling": None,
            "max_position_embeddings": 128,
            "attention_bias": False,
            "vocab_size": 256,
            "pad_token_id": 0,
        }
        dec_cfg = dict(enc_cfg, is_decoder=True)
        hf_config = T5GemmaConfig(encoder=enc_cfg, decoder=dec_cfg)
        arch = determine_architecture_from_hf_config(hf_config)
        assert arch == "T5GemmaForConditionalGeneration"


# ---------------------------------------------------------------------------
# Key translation: HF key → TL canonical key
# ---------------------------------------------------------------------------


class TestT5GemmaKeyTranslation:
    """Verify convert_hf_key_to_tl_key emits the exact keys that weight_processing_conversions
    is indexed by (after the weight_processing regex replaces digits with {i}).

    These tests use no HF model download — only the adapter's component_mapping.
    """

    @pytest.mark.parametrize(
        "hf_key,expected_tl_key",
        [
            # Encoder self-attn
            (
                "model.encoder.layers.0.self_attn.q_proj.weight",
                "encoder_blocks.0.self_attn.q_proj.weight",
            ),
            (
                "model.encoder.layers.0.self_attn.k_proj.weight",
                "encoder_blocks.0.self_attn.k_proj.weight",
            ),
            (
                "model.encoder.layers.0.self_attn.v_proj.weight",
                "encoder_blocks.0.self_attn.v_proj.weight",
            ),
            (
                "model.encoder.layers.0.self_attn.o_proj.weight",
                "encoder_blocks.0.self_attn.o_proj.weight",
            ),
            # Encoder RMSNorm
            (
                "model.encoder.layers.0.pre_self_attn_layernorm.weight",
                "encoder_blocks.0.pre_self_attn_layernorm.weight",
            ),
            (
                "model.encoder.layers.0.post_feedforward_layernorm.weight",
                "encoder_blocks.0.post_feedforward_layernorm.weight",
            ),
            # Encoder MLP
            (
                "model.encoder.layers.0.mlp.gate_proj.weight",
                "encoder_blocks.0.mlp.gate_proj.weight",
            ),
            (
                "model.encoder.layers.0.mlp.down_proj.weight",
                "encoder_blocks.0.mlp.down_proj.weight",
            ),
            # Decoder self-attn
            (
                "model.decoder.layers.0.self_attn.q_proj.weight",
                "decoder_blocks.0.self_attn.q_proj.weight",
            ),
            (
                "model.decoder.layers.0.self_attn.o_proj.weight",
                "decoder_blocks.0.self_attn.o_proj.weight",
            ),
            # Decoder cross-attn
            (
                "model.decoder.layers.0.cross_attn.q_proj.weight",
                "decoder_blocks.0.cross_attn.q_proj.weight",
            ),
            (
                "model.decoder.layers.0.cross_attn.k_proj.weight",
                "decoder_blocks.0.cross_attn.k_proj.weight",
            ),
            (
                "model.decoder.layers.0.cross_attn.v_proj.weight",
                "decoder_blocks.0.cross_attn.v_proj.weight",
            ),
            (
                "model.decoder.layers.0.cross_attn.o_proj.weight",
                "decoder_blocks.0.cross_attn.o_proj.weight",
            ),
            # Decoder RMSNorm
            (
                "model.decoder.layers.0.pre_cross_attn_layernorm.weight",
                "decoder_blocks.0.pre_cross_attn_layernorm.weight",
            ),
            (
                "model.decoder.layers.0.post_cross_attn_layernorm.weight",
                "decoder_blocks.0.post_cross_attn_layernorm.weight",
            ),
            # Decoder MLP
            (
                "model.decoder.layers.0.mlp.gate_proj.weight",
                "decoder_blocks.0.mlp.gate_proj.weight",
            ),
            (
                "model.decoder.layers.0.mlp.up_proj.weight",
                "decoder_blocks.0.mlp.up_proj.weight",
            ),
            (
                "model.decoder.layers.0.mlp.down_proj.weight",
                "decoder_blocks.0.mlp.down_proj.weight",
            ),
            # lm_head
            ("lm_head.out_proj.weight", "unembed.weight"),
        ],
    )
    def test_hf_key_translates_to_tl_key(
        self,
        adapter: T5GemmaArchitectureAdapter,
        hf_key: str,
        expected_tl_key: str,
    ) -> None:
        tl_key = adapter.convert_hf_key_to_tl_key(hf_key)
        assert (
            tl_key == expected_tl_key
        ), f"HF key {hf_key!r} → {tl_key!r}, expected {expected_tl_key!r}"


# ---------------------------------------------------------------------------
# Conversion table keys match what key translation emits
# ---------------------------------------------------------------------------


class TestT5GemmaConversionTableAlignment:
    """Verify the weight_processing_conversions dict is indexed by the exact keys that
    convert_hf_key_to_tl_key emits (with index replaced by {i}).

    This ensures conversions actually fire during weight loading.
    """

    @pytest.mark.parametrize(
        "hf_key",
        [
            # Encoder self-attn q/k/v/o
            "model.encoder.layers.0.self_attn.q_proj.weight",
            "model.encoder.layers.0.self_attn.k_proj.weight",
            "model.encoder.layers.0.self_attn.v_proj.weight",
            "model.encoder.layers.0.self_attn.o_proj.weight",
            # Encoder RMSNorm
            "model.encoder.layers.0.pre_self_attn_layernorm.weight",
            "model.encoder.layers.0.post_self_attn_layernorm.weight",
            "model.encoder.layers.0.pre_feedforward_layernorm.weight",
            "model.encoder.layers.0.post_feedforward_layernorm.weight",
            # Encoder MLP
            "model.encoder.layers.0.mlp.gate_proj.weight",
            "model.encoder.layers.0.mlp.up_proj.weight",
            "model.encoder.layers.0.mlp.down_proj.weight",
            # Decoder self-attn q/k/v/o
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.self_attn.k_proj.weight",
            "model.decoder.layers.0.self_attn.v_proj.weight",
            "model.decoder.layers.0.self_attn.o_proj.weight",
            # Decoder cross-attn q/k/v/o
            "model.decoder.layers.0.cross_attn.q_proj.weight",
            "model.decoder.layers.0.cross_attn.k_proj.weight",
            "model.decoder.layers.0.cross_attn.v_proj.weight",
            "model.decoder.layers.0.cross_attn.o_proj.weight",
            # Decoder RMSNorm
            "model.decoder.layers.0.pre_self_attn_layernorm.weight",
            "model.decoder.layers.0.post_self_attn_layernorm.weight",
            "model.decoder.layers.0.pre_cross_attn_layernorm.weight",
            "model.decoder.layers.0.post_cross_attn_layernorm.weight",
            "model.decoder.layers.0.pre_feedforward_layernorm.weight",
            "model.decoder.layers.0.post_feedforward_layernorm.weight",
            # Decoder MLP
            "model.decoder.layers.0.mlp.gate_proj.weight",
            "model.decoder.layers.0.mlp.up_proj.weight",
            "model.decoder.layers.0.mlp.down_proj.weight",
        ],
    )
    def test_conversion_table_has_key_for_hf_param(
        self,
        adapter: T5GemmaArchitectureAdapter,
        hf_key: str,
    ) -> None:
        import re

        tl_key = adapter.convert_hf_key_to_tl_key(hf_key)
        placeholder_key = re.sub(r"blocks\.(\d+)\.", "blocks.{i}.", tl_key)
        assert adapter.weight_processing_conversions is not None
        in_table = (
            placeholder_key in adapter.weight_processing_conversions
            or placeholder_key.removesuffix(".weight") in adapter.weight_processing_conversions
        )
        assert in_table, (
            f"No conversion entry for HF key {hf_key!r}. "
            f"Translated TL key: {tl_key!r}, placeholder: {placeholder_key!r}"
        )
