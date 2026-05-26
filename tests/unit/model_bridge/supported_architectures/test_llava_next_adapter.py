"""Unit tests for LlavaNextArchitectureAdapter.

LlavaNext shares its module hierarchy with the base Llava adapter (HF's forward
handles high-res tiling internally), so these tests assert that the subclass
preserves the inherited config, component mapping, weight conversions, and
that the factory routes the LlavaNext architecture key to it.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CLIPVisionEncoderBridge,
    EmbeddingBridge,
    GatedMLPBridge,
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
from transformer_lens.model_bridge.supported_architectures.llava_next import (
    LlavaNextArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 8,
    n_key_value_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 100,
    n_ctx: int = 128,
    vision_model_type: str = "clip_vision_model",
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig with a vision sub-config attached."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        default_prepend_bos=True,
        architecture="LlavaNextForConditionalGeneration",
    )
    cfg.vision_config = SimpleNamespace(
        model_type=vision_model_type,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
    )
    return cfg


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> LlavaNextArchitectureAdapter:
    return LlavaNextArchitectureAdapter(cfg)


class TestLlavaNextInheritance:
    """
    Documentation for subclass relationship
    """

    def test_subclass_of_llava(self) -> None:
        assert issubclass(LlavaNextArchitectureAdapter, LlavaArchitectureAdapter)

    def test_instance_is_also_llava(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert isinstance(adapter, LlavaArchitectureAdapter)


class TestLlavaNextAdapterConfig:
    """
    Config attribute tests
    """

    def test_is_multimodal(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.is_multimodal is True

    def test_normalization_type(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_uses_rms_norm(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_attn_only(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_attn_implementation(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.attn_implementation == "eager"

    def test_eps_attr(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.eps_attr == "variance_epsilon"

    def test_n_key_value_heads_preserved(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 4

    def test_vision_config_propagated(self, adapter: LlavaNextArchitectureAdapter) -> None:
        assert adapter.cfg.vision_hidden_size == 128
        assert adapter.cfg.vision_num_layers == 4
        assert adapter.cfg.vision_num_heads == 8


class TestLlavaNextAdapterComponentMapping:
    """
    Testcases for setup component mapping
    """

    @staticmethod
    def _mapping(adapter: LlavaNextArchitectureAdapter) -> dict[str, Any]:
        mapping = adapter.component_mapping
        assert mapping is not None
        return mapping

    def test_vision_encoder_clip_default(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["vision_encoder"], CLIPVisionEncoderBridge)
        assert mapping["vision_encoder"].name == "model.vision_tower"

    def test_vision_encoder_siglip_when_configured(self) -> None:
        cfg = _make_cfg(vision_model_type="siglip_vision_model")
        adapter = LlavaNextArchitectureAdapter(cfg)
        mapping = adapter.component_mapping
        assert mapping is not None
        assert isinstance(mapping["vision_encoder"], SiglipVisionEncoderBridge)

    def test_vision_projector(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["vision_projector"], VisionProjectionBridge)
        assert mapping["vision_projector"].name == "model.multi_modal_projector"

    def test_embed(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.language_model.embed_tokens"

    def test_rotary_emb(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert mapping["rotary_emb"].name == "model.language_model.rotary_emb"

    def test_blocks(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert mapping["blocks"].name == "model.language_model.layers"

    def test_ln_final(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert mapping["ln_final"].name == "model.language_model.norm"

    def test_unembed(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_block_ln1(self, adapter: LlavaNextArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_block_ln2(self, adapter: LlavaNextArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"

    def test_block_attn(self, adapter: LlavaNextArchitectureAdapter) -> None:
        attn = self._mapping(adapter)["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_block_mlp(self, adapter: LlavaNextArchitectureAdapter) -> None:
        mlp = self._mapping(adapter)["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "mlp"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# Weight conversion tests
# ---------------------------------------------------------------------------


class TestLlavaNextAdapterWeightConversions:
    """
    Testcases for accurate weights conversions
    """

    def test_four_conversion_keys(self, adapter: LlavaNextArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert len(convs) == 4

    def test_qkvo_keys_present(self, adapter: LlavaNextArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        for key in [
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        ]:
            assert key in convs

    def test_q_uses_n_heads(self, adapter: LlavaNextArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_k_uses_n_key_value_heads(self, adapter: LlavaNextArchitectureAdapter) -> None:
        """GQA: K is split along n_key_value_heads."""
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.k.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_v_uses_n_key_value_heads(self, adapter: LlavaNextArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.v.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_k_falls_back_to_n_heads_when_no_gqa(self) -> None:
        """Without n_key_value_heads, K must use n_heads."""
        cfg = _make_cfg(n_key_value_heads=None)
        adapter = LlavaNextArchitectureAdapter(cfg)
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.k.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_o_pattern(self, adapter: LlavaNextArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


class TestLlavaNextFactoryRegistration:
    """
    Lllava Next factory Registration Tests
    """

    def test_factory_key_registered(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "LlavaNextForConditionalGeneration" in SUPPORTED_ARCHITECTURES

    def test_factory_returns_llava_next_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        cfg.architecture = "LlavaNextForConditionalGeneration"
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, LlavaNextArchitectureAdapter)

    def test_factory_key_distinct_from_base_llava(self) -> None:
        """LlavaNext must not be aliased to base Llava in the registry."""
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert (
            SUPPORTED_ARCHITECTURES["LlavaNextForConditionalGeneration"]
            is LlavaNextArchitectureAdapter
        )

    def test_import_from_init(self) -> None:
        from transformer_lens.model_bridge.supported_architectures import (
            LlavaNextArchitectureAdapter as FromInit,
        )

        assert FromInit is LlavaNextArchitectureAdapter
