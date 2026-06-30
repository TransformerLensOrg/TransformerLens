"""Unit tests for OptArchitectureAdapter.

Tests cover:
- Config attribute validation
- Post-norm support flags
- Weight conversion keys
- Component mapping structure
- OPT-350m projection mapping
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    PosEmbedBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.opt import (
    OptArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
    do_layer_norm_before: bool = True,
    word_embed_proj_dim: int | None = None,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for OPT adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="OPTForCausalLM",
    )
    cfg.do_layer_norm_before = do_layer_norm_before
    if word_embed_proj_dim is not None:
        cfg.word_embed_proj_dim = word_embed_proj_dim
    return cfg


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> OptArchitectureAdapter:
    return OptArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestOptAdapterConfig:
    """Adapter must set all required config attributes to the correct values."""

    def test_normalization_type_is_ln(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_standard_pos_embedding_omits_rotary_attn_hooks(
        self, adapter: OptArchitectureAdapter
    ) -> None:
        """Standard (learned) positional embeddings build attention WITHOUT rotary hooks.

        AttentionBridge only creates hook_rot_q/hook_rot_k when
        positional_embedding_type == "rotary"; OPT's "standard" setting must leave
        them off. Flipping OPT to rotary would add these hooks and fail this test.
        """
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert not hasattr(attn, "hook_rot_q")
        assert not hasattr(attn, "hook_rot_k")


class TestOptAdapterPostNorm:
    """Post-norm OPT disables transforms that require pre-norm semantics."""

    def test_post_norm_disables_fold_ln(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(do_layer_norm_before=False))
        assert adapter.supports_fold_ln is False

    def test_post_norm_disables_center_writing_weights(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(do_layer_norm_before=False))
        assert adapter.supports_center_writing_weights is False


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestOptAdapterWeightConversions:
    """Adapter must define exactly the four standard QKVO weight conversions."""

    def test_q_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: OptArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: OptArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestOptAdapterComponentMapping:
    """Component mapping must have the correct bridge types and HF module paths."""

    def test_embed_is_embedding_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.decoder.embed_tokens"

    def test_pos_embed_is_pos_embed_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["pos_embed"], PosEmbedBridge)

    def test_pos_embed_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["pos_embed"].name == "model.decoder.embed_positions"

    def test_blocks_is_block_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.decoder.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.decoder.final_layer_norm"

    def test_unembed_is_unembedding_bridge(self, adapter: OptArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: OptArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_ln1_is_normalization_bridge(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_ln1_name(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "self_attn_layer_norm"

    def test_attn_is_attention_bridge(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_requires_attention_mask(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_attention_mask is True

    def test_attn_attention_mask_4d(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].attention_mask_4d is True

    def test_attn_q_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"

    def test_attn_k_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["k"].name == "k_proj"

    def test_attn_v_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["v"].name == "v_proj"

    def test_attn_o_name(self, adapter: OptArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "out_proj"

    def test_ln2_is_normalization_bridge(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)

    def test_ln2_name(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln2"].name == "final_layer_norm"

    def test_mlp_is_symbolic_bridge(self, adapter: OptArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], SymbolicBridge)

    def test_mlp_fc1_name(self, adapter: OptArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "fc1"

    def test_mlp_fc2_name(self, adapter: OptArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "fc2"


# ---------------------------------------------------------------------------
# OPT-350m special path tests
# ---------------------------------------------------------------------------


class TestOpt350mProjectionMapping:
    """OPT-350m uses project_in/project_out instead of final_layer_norm."""

    def test_project_bridges_absent_in_standard_path(self, adapter: OptArchitectureAdapter) -> None:
        assert "project_in" not in adapter.component_mapping
        assert "project_out" not in adapter.component_mapping

    def test_ln_final_absent_when_word_embed_proj_dim_differs(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(d_model=64, word_embed_proj_dim=32))
        assert "ln_final" not in adapter.component_mapping

    def test_project_in_present_when_word_embed_proj_dim_differs(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(d_model=64, word_embed_proj_dim=32))
        assert isinstance(adapter.component_mapping["project_in"], LinearBridge)
        assert adapter.component_mapping["project_in"].name == "model.decoder.project_in"

    def test_project_out_present_when_word_embed_proj_dim_differs(self) -> None:
        adapter = OptArchitectureAdapter(_make_cfg(d_model=64, word_embed_proj_dim=32))
        assert isinstance(adapter.component_mapping["project_out"], LinearBridge)
        assert adapter.component_mapping["project_out"].name == "model.decoder.project_out"
