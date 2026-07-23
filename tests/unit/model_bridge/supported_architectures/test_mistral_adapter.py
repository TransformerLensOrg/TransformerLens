"""Unit tests for MistralArchitectureAdapter.

Tests cover:
- Component mapping structure (bridge types and HF module names)
- Weight conversion key set and rearrange patterns
- GQA: n_key_value_heads propagates to K/V conversions only
- Anti-drift config flags
"""

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 32,
    d_model: int = 4096,
    n_layers: int = 32,
    d_vocab: int = 32000,
    n_ctx: int = 4096,
    n_key_value_heads: int | None = 8,
    **overrides,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Mistral adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        n_key_value_heads=n_key_value_heads,
        architecture="MistralForCausalLM",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def adapter() -> MistralArchitectureAdapter:
    return MistralArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Component mapping — top-level key set and bridge types
# ---------------------------------------------------------------------------


class TestMistralComponentMapping:
    """Component mapping has the correct slots, bridge types, and HF module paths."""

    def test_top_level_keys(self, adapter: MistralArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_no_pos_embed_key(self, adapter: MistralArchitectureAdapter) -> None:
        """Mistral uses rotary embeddings — no learned positional embedding component."""
        assert "pos_embed" not in adapter.component_mapping

    def test_bridge_types(self, adapter: MistralArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: MistralArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: MistralArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_block_submodule_types(self, adapter: MistralArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], AttentionBridge)
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)

    def test_attn_is_position_embeddings_bridge(self, adapter: MistralArchitectureAdapter) -> None:
        """Mistral uses PositionEmbeddingsAttentionBridge, like Qwen2.

        That bridge is what gives Mistral hook_attn_in and the q/k/v forks.
        """
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)

    def test_block_submodule_hf_paths(self, adapter: MistralArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["attn"].name == "self_attn"
        assert blocks.submodules["mlp"].name == "mlp"

    def test_attn_requires_mask_and_position_embeddings(
        self, adapter: MistralArchitectureAdapter
    ) -> None:
        """Mistral RoPE attention requires both attention mask and position embeddings."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

    def test_attn_qkvo_submodule_paths(self, adapter: MistralArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attn_qkvo_are_linear_bridges(self, adapter: MistralArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for sub in attn.submodules.values():
            assert isinstance(sub, LinearBridge)

    def test_mlp_submodule_paths(self, adapter: MistralArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# Anti-drift config flags
# ---------------------------------------------------------------------------


class TestMistralAdapterConfig:
    """Anti-drift flags that must not silently regress."""

    def test_final_rms_is_false(self, adapter: MistralArchitectureAdapter) -> None:
        """Mistral does not use final RMSNorm — final_rms must remain False."""
        assert adapter.cfg.final_rms is False


# ---------------------------------------------------------------------------
# Weight processing conversions
# ---------------------------------------------------------------------------


class TestMistralWeightConversions:
    """weight_processing_conversions has exactly the expected QKVO keys."""

    def test_exact_conversion_key_set(self, adapter: MistralArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_qkv_conversions_use_split_heads_pattern(
        self, adapter: MistralArchitectureAdapter
    ) -> None:
        """'(n h) m -> n m h' splits [n_heads*d_head, d_model] → [n, d_model, d_head]."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_o_conversion_uses_merge_heads_pattern(
        self, adapter: MistralArchitectureAdapter
    ) -> None:
        """'m (n h) -> n h m' moves n to the front for the output projection."""
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"


# ---------------------------------------------------------------------------
# GQA support
# ---------------------------------------------------------------------------


class TestMistralGQASupport:
    """n_key_value_heads must propagate to K/V conversions and leave Q/O unchanged."""

    def test_no_kv_heads_falls_back_to_n_heads(self) -> None:
        """Without n_key_value_heads, K/V fall back to n_heads."""
        adapter = MistralArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=None))
        k_conv = adapter.weight_processing_conversions["blocks.{i}.attn.k.weight"]
        assert k_conv.tensor_conversion.axes_lengths["n"] == 32

    def test_gqa_propagates_to_kv_conversions(self) -> None:
        """With 8 KV heads, K/V conversions must use n=8."""
        adapter = MistralArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=8))
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == 8

    def test_gqa_does_not_affect_q_conversion(self) -> None:
        """Q always uses full n_heads regardless of GQA."""
        adapter = MistralArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=8))
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 32

    def test_gqa_does_not_affect_o_conversion(self) -> None:
        """O projection always uses n_heads; GQA only affects K/V."""
        adapter = MistralArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=8))
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert o_conv.tensor_conversion.axes_lengths["n"] == 32
