"""Unit tests for GraniteMoeArchitectureAdapter.

GraniteMoe is dense Granite with the gated MLP replaced by a Sparse Mixture
of Experts block (``block_sparse_moe``); everything else (embed, rotary
embeddings, attention, final norm, unembed, and the common Granite config
flags) is inherited unchanged from ``GraniteArchitectureAdapter``.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.granite import (
    GraniteArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.granite_moe import (
    GraniteMoeArchitectureAdapter,
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


def _make_dense_cfg() -> TransformerBridgeConfig:
    """Same shape as the MoE config, but for dense Granite (comparison baseline)."""
    return TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        n_heads=N_HEADS,
        d_vocab=D_VOCAB,
        d_mlp=D_MLP,
        n_key_value_heads=N_KV_HEADS,
        default_prepend_bos=False,
        architecture="GraniteForCausalLM",
    )


def _make_moe_cfg() -> TransformerBridgeConfig:
    """Return a minimal synthetic config for GraniteMoe adapter tests."""
    return TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        n_heads=N_HEADS,
        d_vocab=D_VOCAB,
        d_mlp=D_MLP,
        n_key_value_heads=N_KV_HEADS,
        num_experts=4,
        experts_per_token=2,
        default_prepend_bos=False,
        architecture="GraniteMoeForCausalLM",
    )


@pytest.fixture
def dense_adapter() -> GraniteArchitectureAdapter:
    return GraniteArchitectureAdapter(_make_dense_cfg())


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_moe_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> GraniteMoeArchitectureAdapter:
    return GraniteMoeArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestGraniteMoeAdapterConfig:
    """GraniteMoe doesn't override config setup, so it should inherit the same
    flags as dense Granite -- checked explicitly here rather than only
    implied by inheritance."""

    def test_normalization_type(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_false(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_default_prepend_bos_false(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False

    def test_n_key_value_heads_propagated(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == N_KV_HEADS


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestGraniteMoeAdapterComponentMapping:
    """GraniteMoe replaces the dense MLP with MoE; everything else is
    identical to dense Granite."""

    def test_top_level_keys(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_top_level_bridge_types(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_mlp_is_moe_bridge(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)

    def test_moe_hf_path(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.name == "block_sparse_moe"

    def test_attention_unchanged_from_dense(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"
        for submodule in attn.submodules.values():
            assert isinstance(submodule, LinearBridge)

    def test_non_mlp_components_match_dense(
        self,
        dense_adapter: GraniteArchitectureAdapter,
        adapter: GraniteMoeArchitectureAdapter,
    ) -> None:
        """Embed, rotary_emb, ln_final, and unembed are shared with dense Granite."""
        for key in ("embed", "rotary_emb", "ln_final", "unembed"):
            assert type(adapter.component_mapping[key]) is type(
                dense_adapter.component_mapping[key]
            )
            assert adapter.component_mapping[key].name == dense_adapter.component_mapping[key].name


# ---------------------------------------------------------------------------
# Weight conversion key tests
# ---------------------------------------------------------------------------


class TestGraniteMoeAdapterWeightConversions:
    """GraniteMoe doesn't override weight_processing_conversions, so it should
    carry the same qkvo conversion set as dense Granite (there is no
    MoE-specific weight remapping)."""

    def test_exact_conversion_key_set(self, adapter: GraniteMoeArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }
