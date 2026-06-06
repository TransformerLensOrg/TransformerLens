"""Unit tests for Gpt2LmHeadCustomArchitectureAdapter.

Tests cover:
- Component mapping structure (correct bridge types and HF module names)
- Weight conversion keys and count

Note: unlike GPT2ArchitectureAdapter, this adapter sets no cfg.* attributes
and uses a plain AttentionBridge (no combined-QKV split), so it has no
config-attribute tests.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gpt2_lm_head_custom import (
    Gpt2LmHeadCustomArchitectureAdapter,
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
    """Return a minimal TransformerBridgeConfig for custom GPT-2 adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="GPT2LMHeadCustomModel",
    )


@pytest.fixture
def adapter() -> Gpt2LmHeadCustomArchitectureAdapter:
    return Gpt2LmHeadCustomArchitectureAdapter(_make_cfg())


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestCustomAdapterComponentMapping:
    """Component mapping must have the correct bridge types and HF module names."""

    def test_embed_is_embedding_bridge(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_pos_embed_is_pos_embed_bridge(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["pos_embed"], PosEmbedBridge)

    def test_pos_embed_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["pos_embed"].name == "transformer.wpe"

    def test_blocks_is_block_bridge(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.h"

    def test_ln_final_is_normalization_bridge(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_ln1_is_normalization_bridge(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter
    ) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln1"], NormalizationBridge
        )

    def test_ln1_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "ln_1"

    def test_attn_is_attention_bridge(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        """The custom adapter uses a plain AttentionBridge, not a JointQKVAttentionBridge."""
        assert isinstance(adapter.component_mapping["blocks"].submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["attn"].name == "attn"

    def test_ln2_is_normalization_bridge(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter
    ) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln2"], NormalizationBridge
        )

    def test_ln2_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln2"].name == "ln_2"

    def test_mlp_is_mlp_bridge(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"].submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: Gpt2LmHeadCustomArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["mlp"].name == "mlp"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestCustomAdapterWeightConversions:
    """Adapter must define exactly the expected QKVO weight conversion keys."""

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
    def test_conversion_key_present(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter, key: str
    ) -> None:
        assert key in adapter.weight_processing_conversions

    def test_exactly_seven_conversion_keys(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter
    ) -> None:
        assert len(adapter.weight_processing_conversions) == 7

    def test_qkv_conversions_source_from_combined_c_attn(
        self, adapter: Gpt2LmHeadCustomArchitectureAdapter
    ) -> None:
        """Q/K/V weights are all fetched from the single combined c_attn.weight."""
        for key in ("blocks.{i}.attn.q", "blocks.{i}.attn.k", "blocks.{i}.attn.v"):
            conversion = adapter.weight_processing_conversions[key]
            assert conversion.source_key == "transformer.h.{i}.attn.c_attn.weight"
