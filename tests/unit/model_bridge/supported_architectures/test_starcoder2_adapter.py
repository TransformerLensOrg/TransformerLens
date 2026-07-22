"""Unit tests for Starcoder2ArchitectureAdapter.

Tests cover:
- Config flags set by the adapter (LayerNorm, non-gated MLP, rotary)
- Component mapping structure (bridge types and HF module names)
- The GQA-aware weight-conversion key set, including the per-head bias
  rearrangements that distinguish Starcoder2 from the bias-free Llama family

Behavioural coverage (forward pass vs HuggingFace) lives in
``tests/integration/model_bridge/test_starcoder2_adapter.py``.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.starcoder2 import (
    Starcoder2ArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N_HEADS = 8
N_KV_HEADS = 2
D_MODEL = 64
D_MLP = 256
N_LAYERS = 2
N_CTX = 256
D_VOCAB = 1000


def _make_cfg(n_kv_heads: int | None = N_KV_HEADS) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Starcoder2 adapter tests."""
    return TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        n_heads=N_HEADS,
        d_vocab=D_VOCAB,
        d_mlp=D_MLP,
        n_key_value_heads=n_kv_heads,
        architecture="Starcoder2ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> Starcoder2ArchitectureAdapter:
    return Starcoder2ArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestStarcoder2AdapterConfig:
    """Tests that the adapter sets the correct config flags."""

    def test_normalization_type_is_layernorm(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        """Starcoder2 uses LayerNorm (with bias), not RMSNorm."""
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_not_final_rms(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_mlp_is_not_gated(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        """Starcoder2 uses a plain c_fc -> c_proj MLP, not a gated MLP."""
        assert adapter.cfg.gated_mlp is False

    def test_not_attn_only(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_n_key_value_heads_propagated(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == N_KV_HEADS


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestStarcoder2AdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    def test_top_level_keys(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_top_level_bridge_types(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        """Sequential pre-norm block: two LayerNorms, not the single norm of a parallel block."""
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_block_bridge_types(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_block_hf_paths(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["attn"].name == "self_attn"
        assert blocks.submodules["mlp"].name == "mlp"


# ---------------------------------------------------------------------------
# Attention mapping tests
# ---------------------------------------------------------------------------


class TestStarcoder2AdapterAttention:
    """Tests the separate-QKV attention mapping."""

    def test_attention_submodule_keys(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        """Unlike GPTBigCode's combined c_attn, Starcoder2 uses separate q/k/v projections."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_attention_hf_paths(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attention_linear_bridge_types(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for submodule in attn.submodules.values():
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# MLP mapping tests
# ---------------------------------------------------------------------------


class TestStarcoder2AdapterMLP:
    """Tests the non-gated c_fc -> c_proj MLP mapping."""

    def test_mlp_submodule_keys(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        """Non-gated MLP has only in/out, no gate."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"in", "out"}

    def test_mlp_hf_paths(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "c_fc"
        assert mlp.submodules["out"].name == "c_proj"

    def test_mlp_linear_bridge_types(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        for submodule in mlp.submodules.values():
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# Weight conversion key tests
# ---------------------------------------------------------------------------


class TestStarcoder2AdapterWeightConversions:
    """Tests the GQA-aware Q/K/V/O weight and bias conversions."""

    def test_conversion_key_set(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        """Starcoder2 has biases, so q/k/v get both a weight and a per-head bias conversion.

        The output projection ``o`` has no per-head bias (its bias stays ``[d_model]``),
        so there is no ``blocks.{i}.attn.o.bias`` conversion.
        """
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
        }

    def test_no_output_bias_conversion(self, adapter: Starcoder2ArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.bias" not in adapter.weight_processing_conversions

    def test_missing_kv_heads_falls_back_to_n_heads(self) -> None:
        """Without n_key_value_heads the adapter still builds a full conversion set (MHA)."""
        adapter = Starcoder2ArchitectureAdapter(_make_cfg(n_kv_heads=None))
        assert "blocks.{i}.attn.k.bias" in adapter.weight_processing_conversions
