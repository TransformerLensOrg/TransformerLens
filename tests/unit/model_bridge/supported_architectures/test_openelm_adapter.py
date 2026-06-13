"""Unit tests for OpenElmArchitectureAdapter.

Tests cover:
- Config attribute validation (all required attributes are set correctly)
- Component mapping structure (correct bridge types and HF module names)
- Weight conversion keys (empty for OpenELM — native attention handles all variants)
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.openelm import (
    OpenElmArchitectureAdapter,
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
    """Return a minimal TransformerBridgeConfig for OpenELM adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="OpenELMForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> OpenElmArchitectureAdapter:
    return OpenElmArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestOpenElmAdapterConfig:
    """Tests that the adapter sets required config attributes correctly."""

    def test_normalization_type_is_rms(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_true(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_is_true(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_is_false(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm_is_true(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_tokenizer_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM has no bundled tokenizer — uses LLaMA-2 tokenizer as proxy."""
        assert adapter.cfg.tokenizer_name == "NousResearch/Llama-2-7b-hf"


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestOpenElmAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.token_embeddings"

    def test_no_pos_embed_key(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM uses per-layer rotary embeddings — no shared positional embedding."""
        assert "pos_embed" not in adapter.component_mapping

    def test_no_rotary_emb_key(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM RoPE is embedded per-layer in attention, not a top-level component."""
        assert "rotary_emb" not in adapter.component_mapping

    def test_blocks_is_block_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.layers"

    def test_ln_final_is_rms_normalization_bridge(
        self, adapter: OpenElmArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_ln_final_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.norm"

    def test_unembed_is_unembedding_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_ln1_is_rms_normalization_bridge(
        self, adapter: OpenElmArchitectureAdapter
    ) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln1"], RMSNormalizationBridge
        )

    def test_blocks_ln1_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "attn_norm"

    def test_blocks_ln2_is_rms_normalization_bridge(
        self, adapter: OpenElmArchitectureAdapter
    ) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln2"], RMSNormalizationBridge
        )

    def test_blocks_ln2_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln2"].name == "ffn_norm"

    def test_attn_is_attention_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attn"

    def test_attn_requires_attention_mask(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True

    def test_attn_qkv_is_linear_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM uses a combined QKV projection (not separate q/k/v)."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["qkv"], LinearBridge)

    def test_attn_qkv_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["qkv"].name == "qkv_proj"

    def test_attn_o_is_linear_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["o"], LinearBridge)

    def test_attn_o_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "out_proj"

    def test_mlp_is_mlp_bridge(self, adapter: OpenElmArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        """OpenELM names its MLP submodule 'ffn' (feedforward network)."""
        assert adapter.component_mapping["blocks"].submodules["mlp"].name == "ffn"

    def test_mlp_in_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "proj_1"

    def test_mlp_out_name(self, adapter: OpenElmArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "proj_2"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestOpenElmAdapterWeightConversions:
    """Tests that weight_processing_conversions is empty for OpenELM.

    OpenELM uses per-layer varying head counts and FFN dimensions handled
    entirely by native HuggingFace attention — no static weight rearrangements
    are needed at the bridge level.
    """

    def test_no_weight_processing_conversions(self, adapter: OpenElmArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 0
