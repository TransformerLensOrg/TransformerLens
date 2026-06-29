"""Unit tests for GraniteMoeHybridArchitectureAdapter."""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SSM2MixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.granite_moe_hybrid import (
    GraniteMoeHybridArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N_HEADS = 8
N_KV_HEADS = 2
D_MODEL = 64
D_MLP = 256
N_LAYERS = 3
N_CTX = 256
D_VOCAB = 1000
LAYER_TYPES = ["mamba", "attention", "mamba"]


def _make_cfg(
    *,
    num_experts: int | None = 4,
    num_local_experts: int | None = None,
    position_embedding_type: str = "rope",
) -> TransformerBridgeConfig:
    """Return a minimal synthetic config for Granite MoE Hybrid adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        n_heads=N_HEADS,
        d_vocab=D_VOCAB,
        d_mlp=D_MLP,
        n_key_value_heads=N_KV_HEADS,
        num_experts=num_experts,
        experts_per_token=2,
        default_prepend_bos=True,
        architecture="GraniteMoeHybridForCausalLM",
    )
    cfg.layer_types = LAYER_TYPES
    cfg.position_embedding_type = position_embedding_type
    if num_local_experts is not None:
        cfg.num_local_experts = num_local_experts
    return cfg


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> GraniteMoeHybridArchitectureAdapter:
    return GraniteMoeHybridArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestGraniteMoeHybridAdapterConfig:
    """Tests that the adapter sets Granite and hybrid-specific config flags."""

    def test_synthetic_config_fields_are_preserved(
        self, adapter: GraniteMoeHybridArchitectureAdapter
    ) -> None:
        assert adapter.cfg.layer_types == LAYER_TYPES
        assert adapter.cfg.num_experts == 4
        assert adapter.cfg.experts_per_token == 2
        assert adapter.cfg.n_key_value_heads == N_KV_HEADS

    def test_common_granite_flags(self, adapter: GraniteMoeHybridArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_only is False
        assert adapter.cfg.uses_rms_norm is True
        assert adapter.cfg.default_prepend_bos is False

    def test_non_rope_position_embedding_disables_rotary_mapping(self) -> None:
        adapter = GraniteMoeHybridArchitectureAdapter(
            _make_cfg(position_embedding_type="nope", num_experts=4)
        )

        assert adapter.cfg.positional_embedding_type == "none"
        assert "rotary_emb" not in adapter.component_mapping

    def test_num_local_experts_enables_moe_when_num_experts_missing(self) -> None:
        adapter = GraniteMoeHybridArchitectureAdapter(
            _make_cfg(num_experts=None, num_local_experts=4)
        )

        blocks = adapter.component_mapping["blocks"]
        assert "moe" in blocks.submodules

    def test_no_experts_omits_moe_mapping(self) -> None:
        adapter = GraniteMoeHybridArchitectureAdapter(_make_cfg(num_experts=0))

        blocks = adapter.component_mapping["blocks"]
        assert "moe" not in blocks.submodules


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestGraniteMoeHybridAdapterComponentMapping:
    """Tests component bridge types and HuggingFace module paths."""

    def test_top_level_mapping(self, adapter: GraniteMoeHybridArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert set(mapping.keys()) == {"embed", "rotary_emb", "blocks", "ln_final", "unembed"}

        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_mapping(self, adapter: GraniteMoeHybridArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {
            "ln1",
            "ln2",
            "attn",
            "mamba",
            "shared_mlp",
            "moe",
        }

        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mamba"], SSM2MixerBridge)
        assert isinstance(blocks.submodules["shared_mlp"], MLPBridge)
        assert isinstance(blocks.submodules["moe"], MoEBridge)

        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["attn"].name == "self_attn"
        assert blocks.submodules["mamba"].name == "mamba"
        assert blocks.submodules["shared_mlp"].name == "shared_mlp"
        assert blocks.submodules["moe"].name == "block_sparse_moe"

    def test_attention_mapping(self, adapter: GraniteMoeHybridArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.optional is True
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"
        for submodule in attn.submodules.values():
            assert isinstance(submodule, LinearBridge)

    def test_mamba_mapping(self, adapter: GraniteMoeHybridArchitectureAdapter) -> None:
        mamba = adapter.component_mapping["blocks"].submodules["mamba"]
        assert mamba.optional is True
        assert set(mamba.submodules.keys()) == {"in_proj", "conv1d", "inner_norm"}

        assert isinstance(mamba.submodules["in_proj"], LinearBridge)
        assert isinstance(mamba.submodules["conv1d"], DepthwiseConv1DBridge)
        assert isinstance(mamba.submodules["inner_norm"], LinearBridge)

        assert mamba.submodules["in_proj"].name == "in_proj"
        assert mamba.submodules["conv1d"].name == "conv1d"
        assert mamba.submodules["inner_norm"].name == "norm"

    def test_shared_mlp_mapping(self, adapter: GraniteMoeHybridArchitectureAdapter) -> None:
        shared_mlp = adapter.component_mapping["blocks"].submodules["shared_mlp"]
        assert set(shared_mlp.submodules.keys()) == {"in", "out"}

        assert isinstance(shared_mlp.submodules["in"], LinearBridge)
        assert isinstance(shared_mlp.submodules["out"], LinearBridge)

        assert shared_mlp.submodules["in"].name == "input_linear"
        assert shared_mlp.submodules["out"].name == "output_linear"


# ---------------------------------------------------------------------------
# Weight conversion key tests
# ---------------------------------------------------------------------------


class TestGraniteMoeHybridAdapterWeightConversions:
    """Tests weight-processing conversion declarations."""

    def test_weight_processing_conversions_are_empty(
        self, adapter: GraniteMoeHybridArchitectureAdapter
    ) -> None:
        assert adapter.weight_processing_conversions == {}
        assert adapter.supports_fold_ln is False
