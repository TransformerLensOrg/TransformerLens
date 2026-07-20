"""Unit tests for NeoArchitectureAdapter.

Tests cover:
- Component mapping structure (bridge types and HF module names)
- Weight conversion key set
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
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestNeoAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    def test_top_level_keys(self, adapter: NeoArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "pos_embed",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_bridge_types(self, adapter: NeoArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["pos_embed"], PosEmbedBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: NeoArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "transformer.wte"
        assert mapping["pos_embed"].name == "transformer.wpe"
        assert mapping["blocks"].name == "transformer.h"
        assert mapping["ln_final"].name == "transformer.ln_f"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: NeoArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "attn", "ln2", "mlp"}

    def test_attention_submodule_keys(self, adapter: NeoArchitectureAdapter) -> None:
        """Neo uses separate Q, K, V, O projections (no combined QKV matrix)."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_mlp_submodule_keys(self, adapter: NeoArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"in", "out"}

    def test_block_bridge_types(self, adapter: NeoArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert isinstance(blocks.submodules["attn"], AttentionBridge)
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_attention_hf_paths(self, adapter: NeoArchitectureAdapter) -> None:
        """Neo's attention is nested as attn.attention in HuggingFace."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.name == "attn.attention"
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "out_proj"

    def test_block_hf_paths(self, adapter: NeoArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "ln_1"
        assert blocks.submodules["ln2"].name == "ln_2"
        assert blocks.submodules["mlp"].name == "mlp"
        assert blocks.submodules["mlp"].submodules["in"].name == "c_fc"
        assert blocks.submodules["mlp"].submodules["out"].name == "c_proj"

    def test_linear_submodule_bridge_types(self, adapter: NeoArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        for submodule in [*attn.submodules.values(), *mlp.submodules.values()]:
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestNeoAdapterWeightConversions:
    """Tests that weight_processing_conversions has exactly the expected keys.

    Neo uses standard PyTorch Linear layers whose weights are stored as
    [out_features, in_features], requiring a transpose to Conv1D format for
    attention heads and an optional einops rearrangement for head dimensions.
    """

    def test_exact_conversion_key_set(self, adapter: NeoArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
            "blocks.{i}.mlp.in.weight",
            "blocks.{i}.mlp.out.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
        }


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
