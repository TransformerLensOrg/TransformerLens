"""Unit tests for NeoArchitectureAdapter.

Tests cover:
- Config attribute validation (all required attributes are set correctly)
- Component mapping structure (correct bridge types and HF module names)
- Weight conversion keys and count
- NeoLinearTransposeConversion numerical correctness
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.neo import (
    NeoArchitectureAdapter,
    NeoLinearTransposeConversion,
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
    """Return a minimal TransformerBridgeConfig for Neo adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="GPTNeoForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> NeoArchitectureAdapter:
    return NeoArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestNeoAdapterConfig:
    """Tests that the adapter sets required config attributes correctly."""

    def test_normalization_type_is_ln(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_standard(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "standard"

    def test_final_rms_is_false(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestNeoAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_pos_embed_is_pos_embed_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["pos_embed"], PosEmbedBridge)

    def test_pos_embed_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["pos_embed"].name == "transformer.wpe"

    def test_blocks_is_block_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.h"

    def test_ln_final_is_normalization_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_ln1_is_normalization_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln1"], NormalizationBridge
        )

    def test_blocks_ln1_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "ln_1"

    def test_blocks_ln2_is_normalization_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln2"], NormalizationBridge
        )

    def test_blocks_ln2_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln2"].name == "ln_2"

    def test_attn_is_attention_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        """Neo uses separate Q/K/V projections (AttentionBridge), unlike GPT-2's combined QKV."""
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: NeoArchitectureAdapter) -> None:
        """Neo's attention submodule is nested as attn.attention in HuggingFace."""
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attn.attention"

    def test_attn_q_is_linear_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["q"], LinearBridge)

    def test_attn_q_name(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"

    def test_attn_k_is_linear_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["k"], LinearBridge)

    def test_attn_k_name(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["k"].name == "k_proj"

    def test_attn_v_is_linear_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["v"], LinearBridge)

    def test_attn_v_name(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["v"].name == "v_proj"

    def test_attn_o_is_linear_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["o"], LinearBridge)

    def test_attn_o_name(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "out_proj"

    def test_mlp_is_mlp_bridge(self, adapter: NeoArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: NeoArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["mlp"].name == "mlp"

    def test_mlp_in_name(self, adapter: NeoArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "c_fc"

    def test_mlp_out_name(self, adapter: NeoArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "c_proj"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestNeoAdapterWeightConversions:
    """Tests that weight_processing_conversions has exactly the expected keys."""

    @pytest.mark.parametrize(
        "key",
        [
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
            "blocks.{i}.mlp.in.weight",
            "blocks.{i}.mlp.out.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
        ],
    )
    def test_conversion_key_present(self, adapter: NeoArchitectureAdapter, key: str) -> None:
        assert key in adapter.weight_processing_conversions

    def test_exactly_nine_conversion_keys(self, adapter: NeoArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 9


# ---------------------------------------------------------------------------
# NeoLinearTransposeConversion — numerical correctness tests
# ---------------------------------------------------------------------------


class TestNeoLinearTransposeConversion:
    """Numerical correctness of Neo's Linear weight transposition."""

    D_MODEL, N_HEADS, D_HEAD = 64, 4, 16  # D_MODEL = N_HEADS * D_HEAD

    def test_transpose_only_roundtrips(self) -> None:
        """A weight transposed and reverted should recover the original."""
        torch.manual_seed(0)
        conv = NeoLinearTransposeConversion()
        original = torch.randn(self.D_MODEL, self.D_MODEL)
        reverted = conv.revert(conv.handle_conversion(original))
        assert reverted.shape == original.shape
        assert torch.allclose(original, reverted)

    def test_transpose_changes_shape(self) -> None:
        """handle_conversion transposes [out, in] -> [in, out]."""
        w = torch.zeros(128, 64)  # [out_features, in_features]
        out = NeoLinearTransposeConversion().handle_conversion(w)
        assert out.shape == (64, 128)

    def test_transpose_with_rearrange_q_weight(self) -> None:
        """Q/K/V weight: [d_model, n*d_head] -> transpose -> rearrange to [n, d_model, d_head]."""
        conv = NeoLinearTransposeConversion("d_model (n h) -> n d_model h", n=self.N_HEADS)
        w = torch.randn(self.D_MODEL, self.N_HEADS * self.D_HEAD)
        out = conv.handle_conversion(w)
        assert out.shape == (self.N_HEADS, self.D_MODEL, self.D_HEAD)

    def test_transpose_with_rearrange_o_weight(self) -> None:
        """O weight: [n*d_head, d_model] -> transpose -> rearrange to [n, d_head, d_model]."""
        conv = NeoLinearTransposeConversion("(n h) d_model -> n h d_model", n=self.N_HEADS)
        w = torch.randn(self.N_HEADS * self.D_HEAD, self.D_MODEL)
        out = conv.handle_conversion(w)
        assert out.shape == (self.N_HEADS, self.D_HEAD, self.D_MODEL)

    def test_rearrange_roundtrip(self) -> None:
        """handle_conversion -> revert recovers the original weight for Q projection."""
        torch.manual_seed(1)
        conv = NeoLinearTransposeConversion("d_model (n h) -> n d_model h", n=self.N_HEADS)
        original = torch.randn(self.D_MODEL, self.N_HEADS * self.D_HEAD)
        recovered = conv.revert(conv.handle_conversion(original))
        assert recovered.shape == original.shape
        assert torch.allclose(original, recovered, atol=1e-6)

    def test_values_preserved_after_transpose(self) -> None:
        """Values should be identical after transpose (not just shape)."""
        w = torch.arange(12, dtype=torch.float).reshape(3, 4)
        out = NeoLinearTransposeConversion().handle_conversion(w)
        assert torch.allclose(out, w.T)
