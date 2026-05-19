"""Unit tests for XGLMArchitectureAdapter: cfg, components, weight conversions, hook compat, factory."""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.xglm import (
    XGLMArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for XGLM adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="XGLMForCausalLM",
    )


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> XGLMArchitectureAdapter:
    return XGLMArchitectureAdapter(cfg)


class TestXGLMAdapterConfig:
    """Adapter sets all required config attributes."""

    def test_normalization_type_is_ln(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_standard(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "standard"

    def test_final_rms_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is False


class TestXGLMAdapterWeightConversions:
    """Adapter defines exactly the four standard QKVO weight conversions."""

    def test_q_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: XGLMArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4


class TestXGLMAdapterComponentMapping:
    """component_mapping has correct bridge types and HF module paths."""

    def test_embed_is_embedding_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_no_pos_embed_in_mapping(self, adapter: XGLMArchitectureAdapter) -> None:
        # Sinusoidal embeddings have no weights — no bridge entry.
        assert "pos_embed" not in adapter.component_mapping

    def test_blocks_is_block_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.layer_norm"

    def test_unembed_is_unembedding_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_ln1_is_normalization_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_ln1_name(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "self_attn_layer_norm"

    def test_attn_is_attention_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_requires_attention_mask(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_attention_mask is True

    def test_attn_attention_mask_4d(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].attention_mask_4d is True

    def test_attn_q_name(self, adapter: XGLMArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"

    def test_attn_k_name(self, adapter: XGLMArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["k"].name == "k_proj"

    def test_attn_v_name(self, adapter: XGLMArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["v"].name == "v_proj"

    def test_attn_o_name_is_out_proj(self, adapter: XGLMArchitectureAdapter) -> None:
        # XGLM uses out_proj, not o_proj (common scaffold mistake).
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "out_proj"

    def test_ln2_is_normalization_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)

    def test_ln2_name(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln2"].name == "final_layer_norm"

    def test_mlp_is_symbolic_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], SymbolicBridge)

    def test_mlp_in_name(self, adapter: XGLMArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "fc1"

    def test_mlp_out_name(self, adapter: XGLMArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "fc2"


class TestXGLMAdapterHookCompatibility:
    """Adapter must not override setup_hook_compatibility — XGLMScaledWordEmbedding
    scales internally, so any override would double-scale embed.hook_out."""

    def test_adapter_does_not_override_setup_hook_compatibility(
        self, adapter: XGLMArchitectureAdapter
    ) -> None:
        # bridge.py:763 uses hasattr() to decide whether to call the override.
        assert "setup_hook_compatibility" not in vars(type(adapter))


class TestXGLMFactoryRegistration:
    """XGLMForCausalLM is registered in SUPPORTED_ARCHITECTURES and resolves correctly."""

    def test_factory_returns_xglm_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, XGLMArchitectureAdapter)

    def test_factory_key_is_xglm_for_causal_lm(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "XGLMForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_maps_to_correct_class(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert SUPPORTED_ARCHITECTURES["XGLMForCausalLM"] is XGLMArchitectureAdapter


class TestXGLMComponentMappingPresence:
    """Component slots exist (deletion guard)."""

    def test_has_embed(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "embed" in adapter.component_mapping

    def test_has_blocks(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks" in adapter.component_mapping

    def test_has_ln_final(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "ln_final" in adapter.component_mapping

    def test_has_unembed(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "unembed" in adapter.component_mapping

    def test_all_expected_top_level_keys_present(self, adapter: XGLMArchitectureAdapter) -> None:
        # No top-level rotary_emb (sinusoidal) and no pos_embed (non-persistent).
        expected = {"embed", "blocks", "ln_final", "unembed"}
        assert set(adapter.component_mapping.keys()) == expected


class TestXGLMBlockSubmodules:
    """Decoder BlockBridge wires XGLM-pattern submodules."""

    @pytest.fixture(scope="class")
    def blocks(self, adapter: XGLMArchitectureAdapter) -> BlockBridge:
        return adapter.component_mapping["blocks"]

    def test_block_has_required_submodules(self, blocks: BlockBridge) -> None:
        for name in ("ln1", "ln2", "attn", "mlp"):
            assert name in blocks.submodules, f"BlockBridge missing submodule '{name}'"

    def test_ln1_is_normalization_bridge(self, blocks: BlockBridge) -> None:
        ln1 = blocks.submodules["ln1"]
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.name == "self_attn_layer_norm"

    def test_ln2_is_normalization_bridge(self, blocks: BlockBridge) -> None:
        ln2 = blocks.submodules["ln2"]
        assert isinstance(ln2, NormalizationBridge)
        assert ln2.name == "final_layer_norm"

    def test_attn_is_attention_bridge(self, blocks: BlockBridge) -> None:
        attn = blocks.submodules["attn"]
        assert isinstance(attn, AttentionBridge)
        assert attn.name == "self_attn"
        # 4-D mask, no position embeddings (sinusoidal added pre-block).
        assert attn.requires_attention_mask is True
        assert attn.attention_mask_4d is True

    def test_attn_qkvo_submodules_are_linear_bridges(self, blocks: BlockBridge) -> None:
        attn = blocks.submodules["attn"]
        for sub_name, expected_path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "out_proj"),
        ):
            sub = attn.submodules[sub_name]
            assert isinstance(sub, LinearBridge), f"attn.{sub_name} must be LinearBridge"
            assert sub.name == expected_path

    def test_mlp_is_symbolic_bridge(self, blocks: BlockBridge) -> None:
        # fc1/fc2 live directly on the decoder layer — SymbolicBridge holds the TL shape.
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp, SymbolicBridge)

    def test_mlp_submodules_are_linear_bridges(self, blocks: BlockBridge) -> None:
        mlp = blocks.submodules["mlp"]
        for sub_name, expected_path in (("in", "fc1"), ("out", "fc2")):
            sub = mlp.submodules[sub_name]
            assert isinstance(sub, LinearBridge), f"mlp.{sub_name} must be LinearBridge"
            assert sub.name == expected_path

    def test_mlp_has_no_gate(self, blocks: BlockBridge) -> None:
        # Standard 2-layer MLP (fc1 -> gelu -> fc2), NOT gated.
        mlp = blocks.submodules["mlp"]
        assert "gate" not in mlp.submodules


class TestXGLMComponentTypes:
    """Component bridge classes — guard against silent type substitution."""

    def test_embed_type(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_blocks_type(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_ln_final_type(self, adapter: XGLMArchitectureAdapter) -> None:
        # XGLM uses LayerNorm (not RMS).
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_unembed_type(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)


class TestXGLMWeightConversionSemantics:
    """QKVO conversion entries use the expected types and patterns."""

    def test_q_conversion_type(self, adapter: XGLMArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)

    def test_qkv_split_heads_pattern(self, adapter: XGLMArchitectureAdapter) -> None:
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_o_merge_heads_pattern(self, adapter: XGLMArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_qkvo_n_axis_equals_n_heads(self, adapter: XGLMArchitectureAdapter) -> None:
        # MHA: K/V share n_heads with Q/O (no GQA on XGLM).
        for slot in ("q", "k", "v", "o"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


class TestXGLMArchitectureGuards:
    """Guards against drift toward neighbouring adapter patterns."""

    def test_no_gqa_setting(self, adapter: XGLMArchitectureAdapter) -> None:
        # All published XGLM sizes are MHA.
        assert getattr(adapter.cfg, "n_key_value_heads", None) is None

    def test_no_norm_offset_conversions(self, adapter: XGLMArchitectureAdapter) -> None:
        # XGLM is not Gemma — no +1 norm offset entries.
        for key in adapter.weight_processing_conversions:
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key

    def test_no_mlp_weight_conversions(self, adapter: XGLMArchitectureAdapter) -> None:
        for key in adapter.weight_processing_conversions:
            assert "mlp" not in key

    def test_center_writing_weights_disabled(self, adapter: XGLMArchitectureAdapter) -> None:
        # Sinusoidal pos_embed has no params → cannot center pos_embed.
        assert adapter.supports_center_writing_weights is False

    def test_no_rotary_in_blocks(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert "rotary_emb" not in blocks.submodules

    def test_no_top_level_rotary_emb(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "rotary_emb" not in adapter.component_mapping
