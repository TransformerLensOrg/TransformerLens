"""Unit tests for LlamaArchitectureAdapter.

Tests cover:
- Config flags set by the adapter
- Component mapping structure (bridge types and HF module names)
- Weight conversion key set
- GQA support via n_key_value_heads
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
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


def _make_cfg(
    n_heads: int = N_HEADS,
    n_kv_heads: int | None = None,
    d_model: int = D_MODEL,
    n_layers: int = N_LAYERS,
    d_mlp: int = D_MLP,
    d_vocab: int = D_VOCAB,
    n_ctx: int = N_CTX,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Llama adapter tests."""
    kwargs = dict(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=False,
        architecture="LlamaForCausalLM",
    )
    if n_kv_heads is not None:
        kwargs["n_key_value_heads"] = n_kv_heads
    return TransformerBridgeConfig(**kwargs)


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> LlamaArchitectureAdapter:
    return LlamaArchitectureAdapter(cfg)


@pytest.fixture
def gqa_cfg() -> TransformerBridgeConfig:
    return _make_cfg(n_kv_heads=N_KV_HEADS)


@pytest.fixture
def gqa_adapter(gqa_cfg: TransformerBridgeConfig) -> LlamaArchitectureAdapter:
    return LlamaArchitectureAdapter(gqa_cfg)


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestLlamaAdapterConfig:
    """Tests that the adapter sets the correct config flags."""

    def test_normalization_type(self, adapter: LlamaArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: LlamaArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter: LlamaArchitectureAdapter) -> None:
        """Llama uses RMSNorm as the final norm (final_rms=True)."""
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter: LlamaArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_false(self, adapter: LlamaArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter: LlamaArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True


# ---------------------------------------------------------------------------
# GQA config tests
# ---------------------------------------------------------------------------


class TestLlamaAdapterGQA:
    """Tests for GQA (Grouped Query Attention) via n_key_value_heads."""

    def test_n_key_value_heads_propagated_to_cfg(
        self, gqa_adapter: LlamaArchitectureAdapter
    ) -> None:
        assert gqa_adapter.cfg.n_key_value_heads == N_KV_HEADS

    def test_n_key_value_heads_in_default_config(
        self, gqa_adapter: LlamaArchitectureAdapter
    ) -> None:
        assert gqa_adapter.default_config["n_key_value_heads"] == N_KV_HEADS

    def test_no_n_key_value_heads_without_gqa(self, adapter: LlamaArchitectureAdapter) -> None:
        """When n_key_value_heads is not set, default_config should not include it."""
        assert "n_key_value_heads" not in adapter.default_config


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestLlamaAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    def test_top_level_keys(self, adapter: LlamaArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_bridge_types(self, adapter: LlamaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: LlamaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: LlamaArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_block_bridge_types(self, adapter: LlamaArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)

    def test_block_hf_paths(self, adapter: LlamaArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["attn"].name == "self_attn"
        assert blocks.submodules["mlp"].name == "mlp"

    def test_attention_submodule_keys(self, adapter: LlamaArchitectureAdapter) -> None:
        """Llama uses separate Q, K, V, O projections."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_attention_hf_paths(self, adapter: LlamaArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_submodule_keys(self, adapter: LlamaArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_mlp_hf_paths(self, adapter: LlamaArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_attention_linear_bridge_types(self, adapter: LlamaArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for submodule in attn.submodules.values():
            assert isinstance(submodule, LinearBridge)

    def test_mlp_linear_bridge_types(self, adapter: LlamaArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        for submodule in mlp.submodules.values():
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# Weight conversion key tests
# ---------------------------------------------------------------------------


class TestLlamaAdapterWeightConversions:
    """Tests that weight_processing_conversions has exactly the expected keys."""

    def test_exact_conversion_key_set(self, adapter: LlamaArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }
