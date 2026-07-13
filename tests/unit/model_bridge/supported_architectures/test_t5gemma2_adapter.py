"""Unit tests for T5Gemma2ArchitectureAdapter (text-only).

Covered:
- Config flags: normalization_type, positional_embedding_type, gated_mlp, rmsnorm_uses_offset,
  fold_ln disabled, final_rms.
- Component-mapping top-level keys: encoder/decoder embed, rotary_emb, encoder_blocks /
  decoder_blocks, final norms, unembed (lm_head.out_proj).
- Encoder text stack paths live under model.encoder.text_model.*.
- Encoder block: BlockBridge with Gemma-style pre/post norms, RoPE attention with QK-norm,
  gated MLP; no cross_attn.
- Decoder block: T5Gemma2DecoderBlockBridge with a single merged self_attn (base AttentionBridge,
  delegates to native T5Gemma2MergedAttention), QK-norm, two ln pairs, gated MLP; NO cross_attn
  and NO cross-attn layernorms; hook_resid_mid present, hook_resid_mid2 absent.
- Attention projections: q_proj/k_proj/v_proj/o_proj + q_norm/k_norm names.
- MLP projections: gate_proj/up_proj/down_proj names.
- lm_head path resolves to lm_head.out_proj.
- Factory registration: T5Gemma2ForConditionalGeneration in SUPPORTED_ARCHITECTURES.
- Architecture classification: in SEQ2SEQ_ARCHITECTURES.
- Model registry: in HF_SUPPORTED_ARCHITECTURES.
- Architecture detection: architectures list and model_type 't5gemma2' both resolve.
- Key translation: convert_hf_key_to_tl_key maps representative HF keys to canonical TL keys.
- Conversion table: weight_processing_conversions contains the exact keys key translation emits.
"""

import types
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
    GatedMLPBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.t5gemma2_decoder_block import (
    T5Gemma2DecoderBlockBridge,
)
from transformer_lens.model_bridge.generalized_components.t5gemma2_merged_attention import (
    T5Gemma2MergedAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.t5gemma2 import (
    T5Gemma2ArchitectureAdapter,
)
from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES
from transformer_lens.utilities.architectures import SEQ2SEQ_ARCHITECTURES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_cfg(*, architecture: str = "T5Gemma2ForConditionalGeneration") -> TransformerBridgeConfig:
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
def adapter(cfg: TransformerBridgeConfig) -> T5Gemma2ArchitectureAdapter:
    return T5Gemma2ArchitectureAdapter(cfg)


def _mapping(adapter: T5Gemma2ArchitectureAdapter) -> dict:
    m = adapter.component_mapping
    assert m is not None
    return m


def _enc_block(adapter: T5Gemma2ArchitectureAdapter) -> Any:
    return _mapping(adapter)["encoder_blocks"]


def _dec_block(adapter: T5Gemma2ArchitectureAdapter) -> Any:
    return _mapping(adapter)["decoder_blocks"]


# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------


class TestT5Gemma2AdapterConfig:
    def test_encoder_attn_has_rotary_hooks(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        attn = _enc_block(adapter).submodules["attn"]
        assert hasattr(attn, "hook_rot_q")
        assert hasattr(attn, "hook_rot_k")

    def test_gated_mlp_is_true(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_rmsnorm_uses_offset_is_true(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert adapter.cfg.rmsnorm_uses_offset is True

    def test_fold_ln_disabled(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert adapter.supports_fold_ln is False

    def test_final_rms_is_true(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_act_fn_is_gelu_pytorch_tanh(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert adapter.cfg.act_fn == "gelu_pytorch_tanh"


# ---------------------------------------------------------------------------
# Top-level mapping structure
# ---------------------------------------------------------------------------


class TestT5Gemma2ComponentMapping:
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

    def test_top_level_keys(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert set(_mapping(adapter).keys()) == self.EXPECTED_KEYS

    def test_encoder_embed_is_embedding_bridge(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_embed"], EmbeddingBridge)

    def test_encoder_embed_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_embed"].name == "model.encoder.text_model.embed_tokens"

    def test_encoder_rotary_emb_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_rotary_emb"].name == "model.encoder.text_model.rotary_emb"
        assert isinstance(_mapping(adapter)["encoder_rotary_emb"], RotaryEmbeddingBridge)

    def test_encoder_blocks_is_block_bridge(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["encoder_blocks"], BlockBridge)

    def test_encoder_blocks_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_blocks"].name == "model.encoder.text_model.layers"

    def test_encoder_ln_final_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["encoder_ln_final"].name == "model.encoder.text_model.norm"
        assert isinstance(_mapping(adapter)["encoder_ln_final"], RMSNormalizationBridge)

    def test_decoder_embed_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_embed"].name == "model.decoder.embed_tokens"

    def test_decoder_rotary_emb_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_rotary_emb"].name == "model.decoder.rotary_emb"

    def test_decoder_blocks_is_t5gemma2_decoder_block_bridge(
        self, adapter: T5Gemma2ArchitectureAdapter
    ) -> None:
        assert isinstance(_mapping(adapter)["decoder_blocks"], T5Gemma2DecoderBlockBridge)

    def test_decoder_blocks_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_blocks"].name == "model.decoder.layers"

    def test_decoder_ln_final_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["decoder_ln_final"].name == "model.decoder.norm"

    def test_unembed_points_to_lm_head_out_proj(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _mapping(adapter)["unembed"].name == "lm_head.out_proj"
        assert isinstance(_mapping(adapter)["unembed"], UnembeddingBridge)


# ---------------------------------------------------------------------------
# Encoder block submodules
# ---------------------------------------------------------------------------


class TestT5Gemma2EncoderBlock:
    EXPECTED_ENC_KEYS = {"ln1", "ln1_post", "attn", "ln2", "ln2_post", "mlp"}

    def test_encoder_submodule_keys(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert set(_enc_block(adapter).submodules.keys()) == self.EXPECTED_ENC_KEYS

    def test_encoder_has_no_cross_attn(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert "cross_attn" not in _enc_block(adapter).submodules

    def test_encoder_attn_is_native_attention_bridge(
        self, adapter: T5Gemma2ArchitectureAdapter
    ) -> None:
        """Encoder attention delegates to native so per-layer sliding windows apply."""
        attn = _enc_block(adapter).submodules["attn"]
        assert isinstance(attn, AttentionBridge)
        assert not isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_encoder_attn_requires_position_embeddings(
        self, adapter: T5Gemma2ArchitectureAdapter
    ) -> None:
        """HF's T5Gemma2SelfAttention unpacks position_embeddings unconditionally,
        so component testing must always supply a (cos, sin) tuple."""
        assert _enc_block(adapter).submodules["attn"].requires_position_embeddings

    def test_encoder_attn_qkvo_paths(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        subs = _enc_block(adapter).submodules["attn"].submodules
        assert subs["q"].name == "q_proj"
        assert subs["k"].name == "k_proj"
        assert subs["v"].name == "v_proj"
        assert subs["o"].name == "o_proj"

    def test_encoder_attn_has_qk_norm(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        subs = _enc_block(adapter).submodules["attn"].submodules
        assert subs["q_norm"].name == "q_norm"
        assert subs["k_norm"].name == "k_norm"

    def test_encoder_mlp_proj_paths(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert isinstance(_enc_block(adapter).submodules["mlp"], GatedMLPBridge)
        subs = _enc_block(adapter).submodules["mlp"].submodules
        assert subs["gate"].name == "gate_proj"
        assert subs["in"].name == "up_proj"
        assert subs["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# Decoder block submodules (merged attention)
# ---------------------------------------------------------------------------


class TestT5Gemma2DecoderBlock:
    EXPECTED_DEC_KEYS = {"ln1", "ln1_post", "self_attn", "ln2", "ln2_post", "mlp"}

    def test_decoder_submodule_keys(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert set(_dec_block(adapter).submodules.keys()) == self.EXPECTED_DEC_KEYS

    def test_decoder_has_no_cross_attn(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert "cross_attn" not in _dec_block(adapter).submodules

    def test_decoder_self_attn_is_merged_attention_bridge(
        self, adapter: T5Gemma2ArchitectureAdapter
    ) -> None:
        """Merged attention delegates to native (via AttentionBridge), not reimplemented."""
        self_attn = _dec_block(adapter).submodules["self_attn"]
        assert isinstance(self_attn, T5Gemma2MergedAttentionBridge)
        assert isinstance(self_attn, AttentionBridge)
        assert not isinstance(self_attn, PositionEmbeddingsAttentionBridge)

    def test_decoder_self_attn_exposes_cross_pattern_hook(
        self, adapter: T5Gemma2ArchitectureAdapter
    ) -> None:
        """The merged bridge must expose the cross-attention pattern slice as a hook."""
        self_attn = _dec_block(adapter).submodules["self_attn"]
        assert hasattr(self_attn, "hook_cross_pattern")
        assert hasattr(self_attn, "hook_pattern")

    def test_decoder_self_attn_random_inputs_cover_native_signature(
        self, adapter: T5Gemma2ArchitectureAdapter
    ) -> None:
        """Component testing calls the native T5Gemma2MergedAttention with these
        kwargs; it requires position_embeddings, merged_attention_mask, and
        encoder_hidden_states positionally."""
        self_attn = _dec_block(adapter).submodules["self_attn"]
        inputs = self_attn.get_random_inputs(batch_size=2, seq_len=4)
        assert "position_embeddings" in inputs
        assert "merged_attention_mask" in inputs
        assert inputs["encoder_hidden_states"].shape == inputs["hidden_states"].shape

    def test_decoder_self_attn_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["self_attn"].name == "self_attn"

    def test_decoder_self_attn_qkvo_paths(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        subs = _dec_block(adapter).submodules["self_attn"].submodules
        assert subs["q"].name == "q_proj"
        assert subs["k"].name == "k_proj"
        assert subs["v"].name == "v_proj"
        assert subs["o"].name == "o_proj"

    def test_decoder_self_attn_has_qk_norm(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        subs = _dec_block(adapter).submodules["self_attn"].submodules
        assert subs["q_norm"].name == "q_norm"
        assert subs["k_norm"].name == "k_norm"

    def test_decoder_ln1_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["ln1"].name == "pre_self_attn_layernorm"

    def test_decoder_ln2_hf_path(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert _dec_block(adapter).submodules["ln2"].name == "pre_feedforward_layernorm"

    def test_decoder_mlp_proj_paths(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        subs = _dec_block(adapter).submodules["mlp"].submodules
        assert subs["gate"].name == "gate_proj"
        assert subs["in"].name == "up_proj"
        assert subs["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# T5Gemma2DecoderBlockBridge intermediate hook points
# ---------------------------------------------------------------------------


class TestT5Gemma2DecoderBlockHookPoints:
    def test_hook_resid_mid_exists(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        assert hasattr(_dec_block(adapter), "hook_resid_mid")

    def test_no_hook_resid_mid2(self, adapter: T5Gemma2ArchitectureAdapter) -> None:
        """Merged attention collapses self+cross into one sub-layer — no second mid hook."""
        assert not hasattr(_dec_block(adapter), "hook_resid_mid2")

    def test_encoder_block_has_no_hook_resid_mid(
        self, adapter: T5Gemma2ArchitectureAdapter
    ) -> None:
        assert not hasattr(_enc_block(adapter), "hook_resid_mid")


# ---------------------------------------------------------------------------
# Registration / classification
# ---------------------------------------------------------------------------


class TestT5Gemma2Registration:
    def test_registered_in_supported_architectures(self) -> None:
        assert "T5Gemma2ForConditionalGeneration" in SUPPORTED_ARCHITECTURES

    def test_maps_to_t5gemma2_adapter(self) -> None:
        assert (
            SUPPORTED_ARCHITECTURES["T5Gemma2ForConditionalGeneration"]
            is T5Gemma2ArchitectureAdapter
        )

    def test_is_in_seq2seq_architectures(self) -> None:
        assert "T5Gemma2ForConditionalGeneration" in SEQ2SEQ_ARCHITECTURES

    def test_is_in_hf_supported_architectures(self) -> None:
        assert "T5Gemma2ForConditionalGeneration" in HF_SUPPORTED_ARCHITECTURES


# ---------------------------------------------------------------------------
# Architecture detection (no model download)
# ---------------------------------------------------------------------------


class TestT5Gemma2ArchitectureDetection:
    def test_detected_from_architectures_list(self) -> None:
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        stub = types.SimpleNamespace(
            architectures=["T5Gemma2ForConditionalGeneration"], model_type="t5gemma2"
        )
        assert determine_architecture_from_hf_config(stub) == "T5Gemma2ForConditionalGeneration"

    def test_detected_from_model_type(self) -> None:
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        stub = types.SimpleNamespace(architectures=[], model_type="t5gemma2")
        assert determine_architecture_from_hf_config(stub) == "T5Gemma2ForConditionalGeneration"


# ---------------------------------------------------------------------------
# Key translation: HF key → TL canonical key
# ---------------------------------------------------------------------------


class TestT5Gemma2KeyTranslation:
    @pytest.mark.parametrize(
        "hf_key,expected_tl_key",
        [
            # Encoder self-attn (under text_model)
            (
                "model.encoder.text_model.layers.0.self_attn.q_proj.weight",
                "encoder_blocks.0.self_attn.q_proj.weight",
            ),
            (
                "model.encoder.text_model.layers.0.self_attn.q_norm.weight",
                "encoder_blocks.0.self_attn.q_norm.weight",
            ),
            (
                "model.encoder.text_model.layers.0.pre_self_attn_layernorm.weight",
                "encoder_blocks.0.pre_self_attn_layernorm.weight",
            ),
            (
                "model.encoder.text_model.layers.0.mlp.gate_proj.weight",
                "encoder_blocks.0.mlp.gate_proj.weight",
            ),
            # Decoder merged self-attn
            (
                "model.decoder.layers.0.self_attn.q_proj.weight",
                "decoder_blocks.0.self_attn.q_proj.weight",
            ),
            (
                "model.decoder.layers.0.self_attn.k_norm.weight",
                "decoder_blocks.0.self_attn.k_norm.weight",
            ),
            (
                "model.decoder.layers.0.pre_feedforward_layernorm.weight",
                "decoder_blocks.0.pre_feedforward_layernorm.weight",
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
        adapter: T5Gemma2ArchitectureAdapter,
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


class TestT5Gemma2ConversionTableAlignment:
    @pytest.mark.parametrize(
        "hf_key",
        [
            # Encoder self-attn q/k/v/o + qk-norm
            "model.encoder.text_model.layers.0.self_attn.q_proj.weight",
            "model.encoder.text_model.layers.0.self_attn.k_proj.weight",
            "model.encoder.text_model.layers.0.self_attn.v_proj.weight",
            "model.encoder.text_model.layers.0.self_attn.o_proj.weight",
            "model.encoder.text_model.layers.0.self_attn.q_norm.weight",
            "model.encoder.text_model.layers.0.self_attn.k_norm.weight",
            # Encoder RMSNorm
            "model.encoder.text_model.layers.0.pre_self_attn_layernorm.weight",
            "model.encoder.text_model.layers.0.post_self_attn_layernorm.weight",
            "model.encoder.text_model.layers.0.pre_feedforward_layernorm.weight",
            "model.encoder.text_model.layers.0.post_feedforward_layernorm.weight",
            # Encoder MLP
            "model.encoder.text_model.layers.0.mlp.gate_proj.weight",
            "model.encoder.text_model.layers.0.mlp.up_proj.weight",
            "model.encoder.text_model.layers.0.mlp.down_proj.weight",
            # Decoder merged self-attn q/k/v/o + qk-norm
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "model.decoder.layers.0.self_attn.k_proj.weight",
            "model.decoder.layers.0.self_attn.v_proj.weight",
            "model.decoder.layers.0.self_attn.o_proj.weight",
            "model.decoder.layers.0.self_attn.q_norm.weight",
            "model.decoder.layers.0.self_attn.k_norm.weight",
            # Decoder RMSNorm
            "model.decoder.layers.0.pre_self_attn_layernorm.weight",
            "model.decoder.layers.0.post_self_attn_layernorm.weight",
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
        adapter: T5Gemma2ArchitectureAdapter,
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
