"""Unit tests for LLava architecture adapter and configuration."""

from types import SimpleNamespace

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
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


def _make_llava_cfg(vision_model_type: str = "clip_vision_model", **overrides):
    """TransformerBridgeConfig for LLava 1.5 7B."""
    defaults = dict(
        d_model=4096,
        d_head=128,
        n_heads=32,
        n_layers=32,
        n_ctx=4096,
        d_vocab=32064,
        architecture="LlavaForConditionalGeneration",
    )
    defaults.update(overrides)
    cfg = TransformerBridgeConfig(**defaults)
    # vision_config matters for vision-bridge selection (CLIP vs SigLIP).
    cfg.vision_config = SimpleNamespace(
        model_type=vision_model_type,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
    )
    return cfg


class TestLlavaAdapterConfig:
    """LlavaArchitectureAdapter configuration."""

    @pytest.fixture(scope="class")
    def adapter(self):
        cfg = _make_llava_cfg()
        return LlavaArchitectureAdapter(cfg)

    def test_is_multimodal(self, adapter):
        assert adapter.cfg.is_multimodal is True

    def test_vision_config_extracted(self, adapter):
        assert adapter.cfg.vision_hidden_size == 1024
        assert adapter.cfg.vision_num_layers == 24
        assert adapter.cfg.vision_num_heads == 16


class TestLlavaComponentMappingPresence:
    """Required component slots exist."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return LlavaArchitectureAdapter(_make_llava_cfg())

    def test_has_vision_encoder_component(self, adapter):
        assert "vision_encoder" in adapter.component_mapping

    def test_has_vision_projector_component(self, adapter):
        assert "vision_projector" in adapter.component_mapping

    def test_has_language_model_components(self, adapter):
        for name in ("embed", "rotary_emb", "blocks", "ln_final", "unembed"):
            assert name in adapter.component_mapping


class TestLlavaComponentMappingPaths:
    """HF module path for each component slot."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return LlavaArchitectureAdapter(_make_llava_cfg())

    def test_vision_encoder_path(self, adapter):
        assert adapter.component_mapping["vision_encoder"].name == "model.vision_tower"

    def test_vision_projector_path(self, adapter):
        assert adapter.component_mapping["vision_projector"].name == "model.multi_modal_projector"

    def test_embed_path(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.language_model.embed_tokens"

    def test_rotary_emb_path(self, adapter):
        assert adapter.component_mapping["rotary_emb"].name == "model.language_model.rotary_emb"

    def test_blocks_path(self, adapter):
        assert adapter.component_mapping["blocks"].name == "model.language_model.layers"

    def test_ln_final_path(self, adapter):
        assert adapter.component_mapping["ln_final"].name == "model.language_model.norm"

    def test_unembed_path(self, adapter):
        assert adapter.component_mapping["unembed"].name == "lm_head"


class TestLlavaComponentTypes:
    """Component bridge classes — guards against silent type substitution."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return LlavaArchitectureAdapter(_make_llava_cfg())

    def test_vision_encoder_is_clip_bridge(self, adapter):
        # vision_model_type='clip_vision_model' must select CLIP, not SigLIP.
        assert isinstance(adapter.component_mapping["vision_encoder"], CLIPVisionEncoderBridge)

    def test_vision_projector_type(self, adapter):
        assert isinstance(adapter.component_mapping["vision_projector"], VisionProjectionBridge)

    def test_embed_type(self, adapter):
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_rotary_emb_type(self, adapter):
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_blocks_type(self, adapter):
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_ln_final_type(self, adapter):
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_unembed_type(self, adapter):
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)


class TestLlavaSiglipVisionVariant:
    """SigLIP vision-tower variants select the SigLIP bridge."""

    def test_siglip_selects_siglip_bridge(self):
        adapter = LlavaArchitectureAdapter(_make_llava_cfg(vision_model_type="siglip_vision_model"))
        assert isinstance(adapter.component_mapping["vision_encoder"], SiglipVisionEncoderBridge)
        assert not isinstance(adapter.component_mapping["vision_encoder"], CLIPVisionEncoderBridge)

    def test_siglip_short_alias_selects_siglip_bridge(self):
        adapter = LlavaArchitectureAdapter(_make_llava_cfg(vision_model_type="siglip"))
        assert isinstance(adapter.component_mapping["vision_encoder"], SiglipVisionEncoderBridge)


class TestLlavaBlockSubmodules:
    """Language-model BlockBridge wires LLaMA-pattern submodules."""

    @pytest.fixture(scope="class")
    def blocks(self):
        adapter = LlavaArchitectureAdapter(_make_llava_cfg())
        return adapter.component_mapping["blocks"]

    def test_block_has_required_submodules(self, blocks):
        for name in ("ln1", "ln2", "attn", "mlp"):
            assert name in blocks.submodules, f"BlockBridge missing submodule '{name}'"

    def test_ln1_is_rms_norm(self, blocks):
        ln1 = blocks.submodules["ln1"]
        assert isinstance(ln1, RMSNormalizationBridge)
        assert ln1.name == "input_layernorm"

    def test_ln2_is_rms_norm(self, blocks):
        ln2 = blocks.submodules["ln2"]
        assert isinstance(ln2, RMSNormalizationBridge)
        assert ln2.name == "post_attention_layernorm"

    def test_attn_is_position_embeddings_attention(self, blocks):
        """LLaMA-style RoPE attention requires both mask and position embeddings."""
        attn = blocks.submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

    def test_attn_qkv_submodule_paths(self, blocks):
        attn = blocks.submodules["attn"]
        for sub_name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "o_proj"),
        ):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path

    def test_mlp_is_gated(self, blocks):
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "mlp"

    def test_mlp_submodule_paths(self, blocks):
        mlp = blocks.submodules["mlp"]
        for sub_name, expected_path in (
            ("gate", "gate_proj"),
            ("in", "up_proj"),
            ("out", "down_proj"),
        ):
            sub = mlp.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path


class TestLlavaGQASupport:
    """GQA: n_key_value_heads affects K/V conversions only."""

    def test_no_gqa_when_not_set(self):
        """Unset n_key_value_heads falls back to n_heads."""
        cfg = _make_llava_cfg()
        adapter = LlavaArchitectureAdapter(cfg)
        kv_conv = adapter.weight_processing_conversions["blocks.{i}.attn.k.weight"]
        assert kv_conv.tensor_conversion.axes_lengths["n"] == 32

    def test_gqa_propagates_to_kv_conversions(self):
        cfg = _make_llava_cfg(n_key_value_heads=8)
        adapter = LlavaArchitectureAdapter(cfg)
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == 8

    def test_gqa_does_not_change_q_or_o_conversions(self):
        """Q and O always follow n_heads; GQA only affects K/V."""
        cfg = _make_llava_cfg(n_key_value_heads=8)
        adapter = LlavaArchitectureAdapter(cfg)
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 32
        assert o_conv.tensor_conversion.axes_lengths["n"] == 32


class TestLlavaWeightProcessingConversions:
    """Rearrange semantics for QKVO conversion entries."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return LlavaArchitectureAdapter(_make_llava_cfg())

    def test_all_qkvo_keys_exist(self, adapter):
        for slot in ("q", "k", "v", "o"):
            key = f"blocks.{{i}}.attn.{slot}.weight"
            assert key in adapter.weight_processing_conversions

    def test_qkv_conversions_use_split_heads_pattern(self, adapter):
        """'(n h) m -> n m h' splits (n_heads * d_head, d_model) into the bridge's (n, d_model, d_head)."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_o_conversion_uses_merge_heads_pattern(self, adapter):
        """'m (n h) -> n h m' splits the trailing (n_heads*d_head) dim and moves n to the front."""
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_no_norm_offset_conversions(self, adapter):
        """LLaMA-style RMSNorm — no +1 offset like Gemma."""
        for key in adapter.weight_processing_conversions:
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key
