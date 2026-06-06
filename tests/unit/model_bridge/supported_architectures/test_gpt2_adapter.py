"""Unit tests for GPT2ArchitectureAdapter.

Tests cover:
- Config attribute validation (all required attributes are set correctly)
- Component mapping structure (correct bridge types and HF module names)
- Weight conversion keys and count
- QKVSplitRearrangeConversion numerical correctness
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointQKVAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gpt2 import (
    GPT2ArchitectureAdapter,
    QKVSplitRearrangeConversion,
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
    """Return a minimal TransformerBridgeConfig for GPT2 adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="GPT2LMHeadModel",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> GPT2ArchitectureAdapter:
    return GPT2ArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestGPT2AdapterConfig:
    """Tests that the adapter sets required config attributes correctly."""

    def test_normalization_type_is_ln(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_standard(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "standard"

    def test_split_attention_weights_is_true(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.cfg.split_attention_weights is True

    def test_uses_combined_qkv_is_true(self, adapter: GPT2ArchitectureAdapter) -> None:
        """GPT-2 stores Q, K, V in a single combined c_attn matrix."""
        assert adapter.uses_combined_qkv is True

    def test_default_cfg_uses_split_attention(self, adapter: GPT2ArchitectureAdapter) -> None:
        """default_cfg flags that GPT-2's combined QKV must be split."""
        assert adapter.default_cfg["uses_split_attention"] is True


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestGPT2AdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_pos_embed_is_pos_embed_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["pos_embed"], PosEmbedBridge)

    def test_pos_embed_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["pos_embed"].name == "transformer.wpe"

    def test_blocks_is_block_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.h"

    def test_ln_final_is_normalization_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_ln1_is_normalization_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln1"], NormalizationBridge
        )

    def test_blocks_ln1_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln1"].name == "ln_1"

    def test_blocks_ln2_is_normalization_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        """GPT-2 has a second layer norm before the MLP (no parallel attn/MLP)."""
        assert isinstance(
            adapter.component_mapping["blocks"].submodules["ln2"], NormalizationBridge
        )

    def test_blocks_ln2_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].submodules["ln2"].name == "ln_2"

    def test_attn_is_joint_qkv_attention_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], JointQKVAttentionBridge)

    def test_attn_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attn"

    def test_attn_does_not_require_attention_mask(self, adapter: GPT2ArchitectureAdapter) -> None:
        """GPT-2 attention applies a causal mask internally, so no external mask is needed."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is False

    def test_attn_qkv_is_linear_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        """The combined QKV projection is a single LinearBridge wrapping c_attn."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["qkv"], LinearBridge)

    def test_attn_qkv_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["qkv"].name == "c_attn"

    def test_attn_o_is_linear_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["o"], LinearBridge)

    def test_attn_o_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "c_proj"

    def test_mlp_is_mlp_bridge(self, adapter: GPT2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_in_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "c_fc"

    def test_mlp_out_name(self, adapter: GPT2ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "c_proj"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestGPT2AdapterWeightConversions:
    """Tests that weight_processing_conversions has exactly the expected keys."""

    @pytest.mark.parametrize(
        "key",
        [
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
            "blocks.{i}.attn.o.weight",
            "unembed.weight",
        ],
    )
    def test_conversion_key_present(self, adapter: GPT2ArchitectureAdapter, key: str) -> None:
        assert key in adapter.weight_processing_conversions

    def test_exactly_eight_conversion_keys(self, adapter: GPT2ArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 8


# ---------------------------------------------------------------------------
# QKVSplitRearrangeConversion — numerical correctness tests
# ---------------------------------------------------------------------------


class TestQKVSplitRearrangeConversion:
    """Numerical correctness of GPT-2's combined-QKV (c_attn) split."""

    N_HEADS, D_HEAD, D_MODEL = 4, 16, 64  # D_MODEL = N_HEADS * D_HEAD

    def _make_conv(self, qkv_index: int, n_heads: int = 4) -> QKVSplitRearrangeConversion:
        """Helper: build a QKVSplitRearrangeConversion for weight tensors."""
        return QKVSplitRearrangeConversion(
            qkv_index=qkv_index,
            rearrange_pattern="d_model (n h) -> n d_model h",
            n=n_heads,
        )

    @pytest.mark.parametrize(
        "shape, expected",
        [((64, 192), True), ((192, 64), True), ((64, 64), False), ((64, 128), False)],
    )
    def test_combined_detection(self, shape, expected) -> None:
        assert self._make_conv(0)._is_combined_qkv(torch.zeros(*shape)) is expected

    def test_q_k_v_extracted_from_correct_thirds(self) -> None:
        """Q/K/V split from the first/second/third third of the combined weight."""
        blocks = [torch.full((self.D_MODEL, self.D_MODEL), float(v)) for v in (1, 2, 3)]
        combined = torch.cat(blocks, dim=1)
        for idx, const in enumerate((1.0, 2.0, 3.0)):
            out = self._make_conv(idx).handle_conversion(combined)
            assert out.shape == (self.N_HEADS, self.D_MODEL, self.D_HEAD)
            assert torch.all(out == const)

    def test_already_split_weight_roundtrips(self) -> None:
        """handle_conversion -> revert recovers an already-split nn.Linear weight."""
        torch.manual_seed(2)
        conv = self._make_conv(0)
        original = torch.randn(self.N_HEADS * self.D_HEAD, self.D_MODEL)
        recovered = conv.revert(conv.handle_conversion(original))
        assert recovered.shape == original.shape
        assert torch.allclose(original, recovered)
