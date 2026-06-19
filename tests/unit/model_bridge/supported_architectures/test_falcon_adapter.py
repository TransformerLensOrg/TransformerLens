"""Unit tests for FalconArchitectureAdapter.

Tests cover:
- Component mapping for RoPE+parallel (default), ALiBi, and sequential variants
- Bridge types and HF module paths
- Weight conversion key set and rearrange patterns
- GQA: n_key_value_heads propagates to K/V; multi_query forces n_key_value_heads=1
- Anti-drift config flags
"""

import pytest

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    ALiBiJointQKVAttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.falcon import (
    FalconArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 32,
    d_model: int = 4096,
    n_layers: int = 32,
    d_vocab: int = 65024,
    n_ctx: int = 2048,
    n_key_value_heads: int | None = 8,
    alibi: bool = False,
    new_decoder_architecture: bool = False,
    multi_query: bool = False,
    parallel_attn: bool = True,
    **overrides,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Falcon adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        n_key_value_heads=n_key_value_heads,
        architecture="FalconForCausalLM",
    )
    setattr(cfg, "alibi", alibi)
    setattr(cfg, "new_decoder_architecture", new_decoder_architecture)
    setattr(cfg, "multi_query", multi_query)
    setattr(cfg, "parallel_attn", parallel_attn)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def adapter() -> FalconArchitectureAdapter:
    """Default RoPE + parallel attention adapter."""
    return FalconArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Component mapping — RoPE + parallel (default)
# ---------------------------------------------------------------------------


class TestFalconComponentMapping:
    """Component mapping has the correct slots, bridge types, and HF module paths."""

    def test_top_level_keys_rope(self, adapter: FalconArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_no_pos_embed_key(self, adapter: FalconArchitectureAdapter) -> None:
        """Falcon uses rotary or ALiBi — no learned positional embedding component."""
        assert "pos_embed" not in adapter.component_mapping

    def test_bridge_types(self, adapter: FalconArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: FalconArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "transformer.word_embeddings"
        assert mapping["rotary_emb"].name == "transformer.rotary_emb"
        assert mapping["blocks"].name == "transformer.h"
        assert mapping["ln_final"].name == "transformer.ln_f"
        assert mapping["unembed"].name == "lm_head"

    def test_blocks_is_parallel_block_bridge(self, adapter: FalconArchitectureAdapter) -> None:
        """Parallel attention mode uses ParallelBlockBridge."""
        assert isinstance(adapter.component_mapping["blocks"], ParallelBlockBridge)

    def test_parallel_block_has_no_ln2(self, adapter: FalconArchitectureAdapter) -> None:
        """Parallel attention shares one LN — no ln2 in default config."""
        assert "ln2" not in adapter.component_mapping["blocks"].submodules

    def test_block_submodule_keys_parallel(self, adapter: FalconArchitectureAdapter) -> None:
        assert set(adapter.component_mapping["blocks"].submodules.keys()) == {"ln1", "attn", "mlp"}

    def test_block_submodule_types(self, adapter: FalconArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert isinstance(blocks.submodules["attn"], JointQKVPositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_ln1_hf_path_standard(self, adapter: FalconArchitectureAdapter) -> None:
        """Standard (non-new-arch) uses input_layernorm."""
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "input_layernorm"

    def test_attn_hf_path(self, adapter: FalconArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["attn"].name == "self_attention"

    def test_attn_submodule_keys(self, adapter: FalconArchitectureAdapter) -> None:
        """Falcon fuses QKV — submodules are qkv and o, not separate q/k/v."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"qkv", "o"}

    def test_attn_submodule_hf_paths(self, adapter: FalconArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["qkv"].name == "query_key_value"
        assert attn.submodules["o"].name == "dense"

    def test_attn_submodules_are_linear_bridges(self, adapter: FalconArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for sub in attn.submodules.values():
            assert isinstance(sub, LinearBridge)

    def test_mlp_submodule_hf_paths(self, adapter: FalconArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "dense_h_to_4h"
        assert mlp.submodules["out"].name == "dense_4h_to_h"


# ---------------------------------------------------------------------------
# ALiBi variant
# ---------------------------------------------------------------------------


class TestFalconALiBiVariant:
    """ALiBi Falcon uses a different attention bridge and has no rotary_emb key."""

    def test_no_rotary_emb_key(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(alibi=True))
        assert "rotary_emb" not in adapter.component_mapping

    def test_alibi_top_level_keys(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(alibi=True))
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_attn_is_alibi_bridge(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(alibi=True))
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, ALiBiJointQKVAttentionBridge)

    def test_positional_embedding_type_is_alibi(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(alibi=True))
        assert adapter.cfg.positional_embedding_type == "alibi"


# ---------------------------------------------------------------------------
# Sequential (non-parallel) variant
# ---------------------------------------------------------------------------


class TestFalconSequentialVariant:
    """Non-parallel Falcon uses BlockBridge and adds ln2."""

    def test_block_is_block_bridge_not_parallel(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(parallel_attn=False))
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)
        assert not isinstance(adapter.component_mapping["blocks"], ParallelBlockBridge)

    def test_sequential_block_has_ln2(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(parallel_attn=False))
        submodules = adapter.component_mapping["blocks"].submodules
        assert "ln2" in submodules
        assert isinstance(submodules["ln2"], NormalizationBridge)

    def test_ln2_hf_path_sequential(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(parallel_attn=False))
        ln2 = adapter.component_mapping["blocks"].submodules["ln2"]
        assert ln2.name == "post_attention_layernorm"

    def test_parallel_attn_mlp_is_false(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(parallel_attn=False))
        assert adapter.cfg.parallel_attn_mlp is False


# ---------------------------------------------------------------------------
# New-arch variant (Falcon 40B+)
# ---------------------------------------------------------------------------


class TestFalconNewArchVariant:
    """New decoder architecture uses ln_attn as the first layer norm name."""

    def test_ln1_name_is_ln_attn(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(new_decoder_architecture=True))
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "ln_attn"


# ---------------------------------------------------------------------------
# Anti-drift config flags
# ---------------------------------------------------------------------------


class TestFalconAdapterConfig:
    """Anti-drift flags that must not silently regress."""

    def test_normalization_type_is_ln(self, adapter: FalconArchitectureAdapter) -> None:
        """Falcon uses LayerNorm, not RMSNorm."""
        assert adapter.cfg.normalization_type == "LN"

    def test_gated_mlp_is_false(self, adapter: FalconArchitectureAdapter) -> None:
        """Falcon uses a standard (non-gated) MLP."""
        assert adapter.cfg.gated_mlp is False

    def test_parallel_attn_mlp_is_true(self, adapter: FalconArchitectureAdapter) -> None:
        """Default Falcon uses parallel attention+MLP."""
        assert adapter.cfg.parallel_attn_mlp is True

    def test_positional_embedding_type_is_rotary(self, adapter: FalconArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"


# ---------------------------------------------------------------------------
# Weight processing conversions
# ---------------------------------------------------------------------------


class TestFalconWeightConversions:
    """weight_processing_conversions has exactly the expected QKVO keys."""

    def test_exact_conversion_key_set(self, adapter: FalconArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q",
            "blocks.{i}.attn.k",
            "blocks.{i}.attn.v",
            "blocks.{i}.attn.o",
        }

    def test_qkv_conversions_use_split_heads_pattern(
        self, adapter: FalconArchitectureAdapter
    ) -> None:
        """'(n h) m -> n m h' splits [n_heads*d_head, d_model] → [n, d_model, d_head]."""
        for slot in ("q", "k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_o_conversion_uses_merge_heads_pattern(
        self, adapter: FalconArchitectureAdapter
    ) -> None:
        """'m (n h) -> n h m' moves n to the front for the output projection."""
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_q_uses_n_heads(self, adapter: FalconArchitectureAdapter) -> None:
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 32

    def test_kv_use_n_key_value_heads(self, adapter: FalconArchitectureAdapter) -> None:
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}"]
            assert conv.tensor_conversion.axes_lengths["n"] == 8

    def test_o_uses_n_heads(self, adapter: FalconArchitectureAdapter) -> None:
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o"]
        assert o_conv.tensor_conversion.axes_lengths["n"] == 32


# ---------------------------------------------------------------------------
# GQA / multi-query support
# ---------------------------------------------------------------------------


class TestFalconGQASupport:
    """n_key_value_heads propagates to K/V; multi_query forces it to 1."""

    def test_gqa_propagates_to_kv(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=8))
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}"]
            assert conv.tensor_conversion.axes_lengths["n"] == 8

    def test_gqa_does_not_affect_q(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=8))
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 32

    def test_gqa_does_not_affect_o(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=8))
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o"]
        assert o_conv.tensor_conversion.axes_lengths["n"] == 32

    def test_no_kv_heads_falls_back_to_n_heads(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(n_heads=32, n_key_value_heads=None))
        k_conv = adapter.weight_processing_conversions["blocks.{i}.attn.k"]
        assert k_conv.tensor_conversion.axes_lengths["n"] == 32

    def test_multi_query_sets_kv_heads_to_1(self) -> None:
        """multi_query=True overrides n_key_value_heads to 1 on the config."""
        adapter = FalconArchitectureAdapter(_make_cfg(n_heads=32, multi_query=True))
        assert adapter.cfg.n_key_value_heads == 1

    def test_multi_query_kv_conversion_uses_1_head(self) -> None:
        adapter = FalconArchitectureAdapter(_make_cfg(n_heads=32, multi_query=True))
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}"]
            assert conv.tensor_conversion.axes_lengths["n"] == 1
