"""Unit tests for NeoxArchitectureAdapter.

Tests cover:
- Config attribute validation (all required attributes are set correctly)
- Component mapping structure (correct bridge types and HF module names)
- Weight conversion keys and count
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.neox import (
    NeoxArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for NeoX adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=False,
        architecture="GPTNeoXForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> NeoxArchitectureAdapter:
    return NeoxArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestNeoxAdapterConfig:
    """Tests that the adapter sets required config attributes correctly."""

    def test_normalization_type_is_ln(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_rotary(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_false(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_parallel_attn_mlp_is_true(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.cfg.parallel_attn_mlp is True

    def test_default_prepend_bos_is_false(self, adapter: NeoxArchitectureAdapter) -> None:
        """GPT-NeoX/Pythia models were not trained with BOS tokens."""
        assert adapter.cfg.default_prepend_bos is False


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestNeoxAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "gpt_neox.embed_in"

    def test_rotary_emb_is_rotary_embedding_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        """NeoX uses rotary embeddings instead of learned positional embeddings."""
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_rotary_emb_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.component_mapping["rotary_emb"].name == "gpt_neox.rotary_emb"

    def test_no_pos_embed_key(self, adapter: NeoxArchitectureAdapter) -> None:
        """NeoX has no learned positional embedding — uses rotary instead."""
        assert "pos_embed" not in adapter.component_mapping

    def test_blocks_is_parallel_block_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        """NeoX runs attention and MLP in parallel (ParallelBlockBridge)."""
        assert isinstance(adapter.component_mapping["blocks"], ParallelBlockBridge)

    def test_blocks_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "gpt_neox.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "gpt_neox.final_layer_norm"

    def test_unembed_is_unembedding_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "embed_out"

    # -- Block submodules --

    def test_blocks_ln1_is_normalization_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln1"], NormalizationBridge
        )

    def test_blocks_ln1_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "input_layernorm"

    def test_blocks_ln2_is_normalization_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln2"], NormalizationBridge
        )

    def test_blocks_ln2_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert (
            adapter.component_mapping["blocks"].submodules["ln2"].name == "post_attention_layernorm"
        )

    def test_attn_is_joint_qkv_position_embeddings_bridge(
        self, adapter: NeoxArchitectureAdapter
    ) -> None:
        """NeoX uses a combined QKV matrix with rotary embeddings."""
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], JointQKVPositionEmbeddingsAttentionBridge)

    def test_attn_name(self, adapter: NeoxArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attention"

    def test_attn_requires_attention_mask(self, adapter: NeoxArchitectureAdapter) -> None:
        """GPTNeoX/StableLM requires an explicit attention mask."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True

    def test_attn_qkv_is_linear_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["qkv"], LinearBridge)

    def test_attn_qkv_name(self, adapter: NeoxArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["qkv"].name == "query_key_value"

    def test_attn_o_is_linear_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["o"], LinearBridge)

    def test_attn_o_name(self, adapter: NeoxArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "dense"

    def test_mlp_is_mlp_bridge(self, adapter: NeoxArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: NeoxArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["mlp"].name == "mlp"

    def test_mlp_in_name(self, adapter: NeoxArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "dense_h_to_4h"

    def test_mlp_out_name(self, adapter: NeoxArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "dense_4h_to_h"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestNeoxAdapterWeightConversions:
    """Tests that weight_processing_conversions has exactly the expected keys."""

    @pytest.mark.parametrize(
        "key",
        [
            "blocks.{i}.attn.q",
            "blocks.{i}.attn.k",
            "blocks.{i}.attn.v",
            "blocks.{i}.attn.b_Q",
            "blocks.{i}.attn.b_K",
            "blocks.{i}.attn.b_V",
            "blocks.{i}.attn.o",
        ],
    )
    def test_conversion_key_present(self, adapter: NeoxArchitectureAdapter, key: str) -> None:
        assert key in adapter.weight_processing_conversions

    def test_exactly_seven_conversion_keys(self, adapter: NeoxArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 7

    def test_qkv_conversions_share_source_key(self, adapter: NeoxArchitectureAdapter) -> None:
        """Q, K, V weights all come from the same combined QKV matrix in HuggingFace."""
        expected_source = "gpt_neox.layers.{i}.attention.query_key_value.weight"
        for key in ("blocks.{i}.attn.q", "blocks.{i}.attn.k", "blocks.{i}.attn.v"):
            assert adapter.weight_processing_conversions[key].source_key == expected_source

    def test_bias_conversions_share_source_key(self, adapter: NeoxArchitectureAdapter) -> None:
        """Q, K, V biases all come from the same combined QKV bias vector."""
        expected_source = "gpt_neox.layers.{i}.attention.query_key_value.bias"
        for key in ("blocks.{i}.attn.b_Q", "blocks.{i}.attn.b_K", "blocks.{i}.attn.b_V"):
            assert adapter.weight_processing_conversions[key].source_key == expected_source

    def test_o_conversion_source_key(self, adapter: NeoxArchitectureAdapter) -> None:
        expected_source = "gpt_neox.layers.{i}.attention.dense.weight"
        assert (
            adapter.weight_processing_conversions["blocks.{i}.attn.o"].source_key == expected_source
        )
