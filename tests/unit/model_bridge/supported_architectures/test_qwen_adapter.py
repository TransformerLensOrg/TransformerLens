"""Unit tests for QwenArchitectureAdapter

Tests cover:
- Config attributes the adapter sets (RMSNorm, rotary, gated MLP)
- Component mapping: TL canonical names, Qwen HF module paths, and bridge types
- The weight processing conversion keys {q/k/v/o} and their HF source weights
- _split_qkv_matrix: Qwen fuses Q, K, V into one `c_attn` matrix, this splits it back
- Factory registration: "QwenForCausalLM" resolves to this adapter

These are pure unit tests. We build the adapter from a tiny mock config and inspect
the Python object. Nothing is downloaded and no real Qwen checkpoint is loaded
"""

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    JointQKVAttentionBridge,
    LinearBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen import (
    QwenArchitectureAdapter,
)


# Helpers
def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Qwen adapter tests

    Dimensions are kept tiny so the tests stay fast and need no HF download.
    d_head is derived so that d_model == n_heads * d_head.
    """
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=False,
        architecture="QwenForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> QwenArchitectureAdapter:
    return QwenArchitectureAdapter(cfg)


# Config attribute tests
class TestQwenAdapterConfig:
    """The adapter sets these flags so downstream weight processing behaves correctly."""

    # Qwen uses RMSNorm, not LayerNorm
    def test_normalization_type_is_rms(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_true(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_is_true(self, adapter: QwenArchitectureAdapter) -> None:
        """Qwen's MLP has a gate branch, so the adapter flags gated_mlp."""
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_is_false(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False


# Component mapping tests
class TestQwenComponentMapping:
    """The adapter contract: TL canonical names mapped to Qwen HF module paths."""

    # Top-level keys

    def test_top_level_keys(self, adapter: QwenArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_embed_is_embedding_bridge(self, adapter: QwenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_blocks_is_block_bridge(self, adapter: QwenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.h"

    def test_ln_final_is_normalization_bridge(self, adapter: QwenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(self, adapter: QwenArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: QwenArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # Block submodules

    def test_block_submodule_keys(self, adapter: QwenArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "attn", "ln2", "mlp"}

    def test_ln1_is_normalization_bridge_named_ln_1(self, adapter: QwenArchitectureAdapter) -> None:
        ln1 = adapter.component_mapping["blocks"].submodules["ln1"]
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.name == "ln_1"

    def test_ln2_is_normalization_bridge_named_ln_2(self, adapter: QwenArchitectureAdapter) -> None:
        ln2 = adapter.component_mapping["blocks"].submodules["ln2"]
        assert isinstance(ln2, NormalizationBridge)
        assert ln2.name == "ln_2"

    # Attention submodules

    def test_attn_is_joint_qkv_attention_bridge(self, adapter: QwenArchitectureAdapter) -> None:
        """Qwen fuses Q/K/V into one matrix, so attention uses a JointQKVAttentionBridge."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, JointQKVAttentionBridge)

    def test_attn_name(self, adapter: QwenArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.name == "attn"

    def test_attn_submodule_keys(self, adapter: QwenArchitectureAdapter) -> None:
        """Adapter only defines qkv and o, but JointQKVAttentionBridge already splits
        the fused qkv back into q/k/v, so we include those in the adapter test here."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"qkv", "o", "q", "k", "v"}

    def test_attn_qkv_is_linear_bridge_named_c_attn(self, adapter: QwenArchitectureAdapter) -> None:
        qkv = adapter.component_mapping["blocks"].submodules["attn"].submodules["qkv"]
        assert isinstance(qkv, LinearBridge)
        assert qkv.name == "c_attn"

    def test_attn_o_is_linear_bridge_named_c_proj(self, adapter: QwenArchitectureAdapter) -> None:
        o = adapter.component_mapping["blocks"].submodules["attn"].submodules["o"]
        assert isinstance(o, LinearBridge)
        assert o.name == "c_proj"

    # MLP submodule

    def test_mlp_is_gated_mlp_bridge(self, adapter: QwenArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

    def test_mlp_name(self, adapter: QwenArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.name == "mlp"

    def test_mlp_submodule_keys(self, adapter: QwenArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_mlp_hf_paths(self, adapter: QwenArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "w1"
        assert mlp.submodules["in"].name == "w2"
        assert mlp.submodules["out"].name == "c_proj"

    def test_all_linear_submodules_are_linear_bridges(
        self, adapter: QwenArchitectureAdapter
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        for submodule in [*attn.submodules.values(), *mlp.submodules.values()]:
            assert isinstance(submodule, LinearBridge)


# Weight processing conversion tests


class TestQwenWeightConversions:
    """weight_processing_conversions reshape Qwen's fused weights into TL head-split form."""

    @pytest.mark.parametrize(
        "key",
        [
            "blocks.{i}.attn.q",
            "blocks.{i}.attn.k",
            "blocks.{i}.attn.v",
            "blocks.{i}.attn.o",
        ],
    )
    def test_conversion_key_present(self, adapter: QwenArchitectureAdapter, key: str) -> None:
        assert key in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: QwenArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4

    @pytest.mark.parametrize("key", ["q", "k", "v"])
    def test_qkv_conversions_read_from_c_attn(
        self, adapter: QwenArchitectureAdapter, key: str
    ) -> None:
        """Q, K, V all come from the single fused c_attn weight."""
        conversion = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{key}"]
        assert conversion.source_key == "transformer.h.{i}.attn.c_attn.weight"

    def test_o_conversion_reads_from_c_proj(self, adapter: QwenArchitectureAdapter) -> None:
        conversion = adapter.weight_processing_conversions["blocks.{i}.attn.o"]
        assert conversion.source_key == "transformer.h.{i}.attn.c_proj.weight"


# _split_qkv_matrix — numerical correctness tests


class MockQwenAttention(nn.Module):
    """Stand-in for Qwen's HF attention module.

    _split_qkv_matrix only looks at a single attribute, `c_attn`, so this is all
    we need. c_attn is a fused linear that produces Q, K, V stacked together (3x wide).
    """

    def __init__(self, c_attn: nn.Linear) -> None:
        super().__init__()
        self.c_attn = c_attn


class TestQwenQKVSplit:
    """Qwen stores Q/K/V in one c_attn matrix; _split_qkv_matrix slices it back into three for use."""

    D_MODEL = 64

    def test_returns_three_linears_of_right_shape(self, adapter: QwenArchitectureAdapter) -> None:
        d_model = self.D_MODEL
        mock = MockQwenAttention(nn.Linear(d_model, 3 * d_model, bias=True))

        q, k, v = adapter._split_qkv_matrix(mock)

        for proj in (q, k, v):
            assert isinstance(proj, nn.Linear)
            assert proj.weight.shape == (d_model, d_model)
            assert proj.bias.shape == (d_model,)

    def test_thirds_land_in_correct_projection(self, adapter: QwenArchitectureAdapter) -> None:
        """Standard Linear layout: rows [0:d], [d:2d], [2d:3d] become Q, K, V."""
        d_model = self.D_MODEL
        c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        with torch.no_grad():
            # Fill each third with a recognizable constant: Q=1, K=2, V=3
            c_attn.weight[:d_model] = 1.0
            c_attn.weight[d_model : 2 * d_model] = 2.0
            c_attn.weight[2 * d_model :] = 3.0

        q, k, v = adapter._split_qkv_matrix(MockQwenAttention(c_attn))

        assert torch.all(q.weight == 1.0)
        assert torch.all(k.weight == 2.0)
        assert torch.all(v.weight == 3.0)

    def test_split_matches_fused_output(self, adapter: QwenArchitectureAdapter) -> None:
        """Guarantee that the split projections reproduce the fused layer's output."""
        torch.manual_seed(0)
        d_model = self.D_MODEL
        c_attn = nn.Linear(d_model, 3 * d_model, bias=True)  # random weights and biases for testing
        mock = MockQwenAttention(c_attn)

        q, k, v = adapter._split_qkv_matrix(mock)

        x = torch.randn(2, 8, d_model)
        fused = c_attn(x)
        assert torch.allclose(q(x), fused[..., :d_model], atol=1e-6)
        assert torch.allclose(k(x), fused[..., d_model : 2 * d_model], atol=1e-6)
        assert torch.allclose(v(x), fused[..., 2 * d_model :], atol=1e-6)

    def test_no_bias_gives_zero_biases(self, adapter: QwenArchitectureAdapter) -> None:
        """When c_attn has no bias, the split projections get zero biases."""
        d_model = self.D_MODEL
        mock = MockQwenAttention(nn.Linear(d_model, 3 * d_model, bias=False))

        q, k, v = adapter._split_qkv_matrix(mock)

        for proj in (q, k, v):
            assert torch.all(proj.bias == 0.0)

    def test_conv1d_style_layout_is_transposed(self, adapter: QwenArchitectureAdapter) -> None:
        """Conv1D-style storage has weight shape (d_model, 3*d_model), it is split on dim=1."""
        d_model = self.D_MODEL
        c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        # Overwrite with a (d_model, 3*d_model) weight: columns are the Q/K/V thirds.
        thirds = [torch.full((d_model, d_model), float(c)) for c in (1, 2, 3)]
        c_attn.weight = nn.Parameter(torch.cat(thirds, dim=1))

        q, k, v = adapter._split_qkv_matrix(MockQwenAttention(c_attn))

        # Transposing a constant block keeps it constant and shapes stay (d_model, d_model).
        assert q.weight.shape == (d_model, d_model)
        assert torch.all(q.weight == 1.0)
        assert torch.all(k.weight == 2.0)
        assert torch.all(v.weight == 3.0)

    def test_unexpected_shape_raises(self, adapter: QwenArchitectureAdapter) -> None:
        """A c_attn whose shape is neither layout should fail loudly, not quietly misbehave."""
        mock = MockQwenAttention(nn.Linear(8, 8))
        with pytest.raises(ValueError, match="Unexpected c_attn weight shape"):
            adapter._split_qkv_matrix(mock)


# Factory registration test


class TestQwenFactoryRegistration:
    """The factory must resolve Qwen's HF architecture string to this adapter."""

    def test_factory_returns_qwen_adapter(self, cfg: TransformerBridgeConfig) -> None:
        built = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(built, QwenArchitectureAdapter)
