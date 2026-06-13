"""Unit tests for LlavaOnevisionArchitectureAdapter.

LlavaOnevisionArchitectureAdapter inherits its config, component mapping, and
weight conversions from LlavaArchitectureAdapter (covered by test_llava_adapter.py).
This suite pins the subclass contract and the prepare_model weight-tying override.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CLIPVisionEncoderBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SiglipVisionEncoderBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llava_onevision import (
    LlavaOnevisionArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(vision_model_type: str = "clip_vision_model") -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for LLaVA-OneVision tests."""
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        n_layers=2,
        n_ctx=512,
        d_vocab=1000,
        architecture="LlavaOnevisionForConditionalGeneration",
    )
    cfg.vision_config = SimpleNamespace(
        model_type=vision_model_type,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
    )
    return cfg


@pytest.fixture
def adapter() -> LlavaOnevisionArchitectureAdapter:
    return LlavaOnevisionArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------


class TestLlavaOnevisionInheritance:
    """LlavaOnevisionArchitectureAdapter must be a LlavaArchitectureAdapter subclass."""

    def test_subclass_of_llava(self) -> None:
        assert issubclass(LlavaOnevisionArchitectureAdapter, LlavaArchitectureAdapter)

    def test_instance_of_llava(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter, LlavaArchitectureAdapter)


# ---------------------------------------------------------------------------
# Config attribute tests (inherited from LlavaArchitectureAdapter)
# ---------------------------------------------------------------------------


class TestLlavaOnevisionConfig:
    """Config attributes inherited from LlavaArchitectureAdapter."""

    def test_is_multimodal(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.cfg.is_multimodal is True

    def test_normalization_type_is_rms(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_true(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_is_true(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_is_false(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_eps_attr(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.cfg.eps_attr == "variance_epsilon"


# ---------------------------------------------------------------------------
# Component mapping tests (inherited from LlavaArchitectureAdapter)
# ---------------------------------------------------------------------------


class TestLlavaOnevisionComponentMapping:
    """Component mapping inherited from LlavaArchitectureAdapter."""

    def test_vision_encoder_is_clip_by_default(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["vision_encoder"], CLIPVisionEncoderBridge)

    def test_vision_encoder_is_siglip_when_configured(self) -> None:
        siglip_adapter = LlavaOnevisionArchitectureAdapter(_make_cfg("siglip_vision_model"))
        assert isinstance(
            siglip_adapter.component_mapping["vision_encoder"], SiglipVisionEncoderBridge
        )

    def test_vision_projector_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["vision_projector"], VisionProjectionBridge)
        assert adapter.component_mapping["vision_projector"].name == "model.multi_modal_projector"

    def test_embed_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)
        assert adapter.component_mapping["embed"].name == "model.language_model.embed_tokens"

    def test_rotary_emb_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert adapter.component_mapping["rotary_emb"].name == "model.language_model.rotary_emb"

    def test_blocks_is_block_bridge(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.language_model.layers"

    def test_ln_final_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)
        assert adapter.component_mapping["ln_final"].name == "model.language_model.norm"

    def test_unembed_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_blocks_ln1_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        ln1 = adapter.component_mapping["blocks"].submodules["ln1"]
        assert isinstance(ln1, RMSNormalizationBridge)
        assert ln1.name == "input_layernorm"

    def test_blocks_ln2_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        ln2 = adapter.component_mapping["blocks"].submodules["ln2"]
        assert isinstance(ln2, RMSNormalizationBridge)
        assert ln2.name == "post_attention_layernorm"

    def test_attn_is_position_embeddings_bridge(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_attn_name(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["attn"].name == "self_attn"

    def test_attn_requires_attention_mask(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True

    def test_attn_q_k_v_o_names(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_is_gated_mlp_bridge(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

    def test_mlp_submodule_names(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# Weight conversion tests
# ---------------------------------------------------------------------------


class TestLlavaOnevisionWeightConversions:
    """Weight conversions inherited from LlavaArchitectureAdapter."""

    @pytest.mark.parametrize(
        "key",
        [
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        ],
    )
    def test_conversion_key_present(
        self, adapter: LlavaOnevisionArchitectureAdapter, key: str
    ) -> None:
        assert key in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4


# ---------------------------------------------------------------------------
# prepare_model weight-tying tests
# ---------------------------------------------------------------------------


class TestLlavaOnevisionPrepareModel:
    """prepare_model fixes weight tying when text_config and top-level config disagree."""

    def _make_hf_model(
        self, tie_word_embeddings_text: bool, tie_word_embeddings_top: bool
    ) -> MagicMock:
        """Build a minimal mock HF model for prepare_model testing."""
        embed = MagicMock()
        embed.weight = "original_weight"

        language_model = MagicMock()
        language_model.embed_tokens = embed

        model = MagicMock()
        model.language_model = language_model

        lm_head = MagicMock()
        lm_head.weight = "random_weight"

        text_config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings_text)
        config = SimpleNamespace(
            tie_word_embeddings=tie_word_embeddings_top,
            text_config=text_config,
        )

        hf_model = MagicMock()
        hf_model.model = model
        hf_model.lm_head = lm_head
        hf_model.config = config
        return hf_model

    def test_ties_weights_when_text_config_says_tied_but_top_level_says_not(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        """lm_head.weight should be set to embed.weight when text_config disagrees."""
        hf_model = self._make_hf_model(tie_word_embeddings_text=True, tie_word_embeddings_top=False)
        adapter.prepare_model(hf_model)
        assert hf_model.lm_head.weight == "original_weight"

    def test_no_tying_when_both_agree_tied(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        """No weight override when top-level config already says tied."""
        hf_model = self._make_hf_model(tie_word_embeddings_text=True, tie_word_embeddings_top=True)
        original_weight = hf_model.lm_head.weight
        adapter.prepare_model(hf_model)
        assert hf_model.lm_head.weight == original_weight

    def test_no_tying_when_text_config_says_not_tied(
        self, adapter: LlavaOnevisionArchitectureAdapter
    ) -> None:
        """No weight override when text_config says not tied."""
        hf_model = self._make_hf_model(
            tie_word_embeddings_text=False, tie_word_embeddings_top=False
        )
        original_weight = hf_model.lm_head.weight
        adapter.prepare_model(hf_model)
        assert hf_model.lm_head.weight == original_weight

    def test_no_op_when_no_lm_head(self, adapter: LlavaOnevisionArchitectureAdapter) -> None:
        """prepare_model is a no-op when lm_head is absent."""
        hf_model = MagicMock(spec=[])  # no attributes at all
        adapter.prepare_model(hf_model)  # must not raise
