"""Unit tests for FalconH1ArchitectureAdapter.

Covers: config attribute propagation, component mapping bridge types and HF path
names, the parallel attn+mamba block shape, Mamba inner-norm optional flag,
applicable_phases, create_stateful_cache, factory + registry registration, and
guard tests against drift toward neighbouring (single-mixer) hybrid patterns.

Structural-only: no weight load, no HF Hub access.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    GatedRMSNormBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SSM2MixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.falcon_h1 import (
    FalconH1ArchitectureAdapter,
)


def _make_cfg(
    n_layers: int = 2,
    d_model: int = 64,
    d_head: int = 16,
    n_heads: int = 4,
    n_key_value_heads: int = 2,
    d_vocab: int = 100,
    n_ctx: int = 128,
    mamba_d_ssm: int = 32,
    mamba_n_heads: int = 4,
    mamba_d_head: int = 8,
    mamba_d_state: int = 4,
    mamba_n_groups: int = 1,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for Falcon-H1 adapter tests.

    Small Mamba-2 dimensions so tests run without loading any weights. The
    Falcon-H1-specific fields are injected via ``setattr`` (the adapter reads
    them with ``getattr``); they are not declared ``TransformerBridgeConfig``
    fields.
    """
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_head,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        default_prepend_bos=False,
        architecture="FalconH1ForCausalLM",
    )
    setattr(cfg, "mamba_d_ssm", mamba_d_ssm)
    setattr(cfg, "mamba_n_heads", mamba_n_heads)
    setattr(cfg, "mamba_d_head", mamba_d_head)
    setattr(cfg, "mamba_d_state", mamba_d_state)
    setattr(cfg, "mamba_n_groups", mamba_n_groups)
    return cfg


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> FalconH1ArchitectureAdapter:
    return FalconH1ArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attributes
# ---------------------------------------------------------------------------


class TestFalconH1AdapterConfig:
    """Adapter propagates all required config attributes."""

    def test_normalization_type_rms(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_uses_rms_norm(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_positional_embedding_type_rotary(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # Falcon-H1 has a model-level rotary module (unlike NemotronH).
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp_true(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # SwiGLU feed-forward.
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_false(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_final_rms_true(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_not_stateful(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # Falcon-H1 generates through the standard unified KV-cache path; the
        # pure-Mamba stateful loop would diverge from HF after the first decode.
        assert getattr(adapter.cfg, "is_stateful", False) is False

    def test_eps_attr_variance_epsilon(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert adapter.cfg.eps_attr == "variance_epsilon"

    def test_n_key_value_heads_propagated(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 2

    def test_mamba_intermediate_size_propagated(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # mamba_d_ssm is the inner SSM width directly.
        assert getattr(adapter.cfg, "mamba_intermediate_size", None) == 32

    def test_conv_dim_propagated(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # d_ssm=32, n_groups=1, d_state=4 → 32 + 2*1*4 = 40
        assert getattr(adapter.cfg, "conv_dim", None) == 40

    def test_expected_in_proj_out_features(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # 2*d_ssm + conv_dim + mamba_n_heads = 64 + 40 + 4 = 108
        assert getattr(adapter.cfg, "expected_in_proj_out_features", None) == 108

    def test_applicable_phases_empty(self) -> None:
        # verify_models is transformer-shaped; SSM hybrids skip it.
        assert FalconH1ArchitectureAdapter.applicable_phases == []


class TestFalconH1ConvDimFormula:
    """conv_dim and in_proj derivations track the Mamba-2 layout."""

    def test_conv_dim_formula(self) -> None:
        a = FalconH1ArchitectureAdapter(
            _make_cfg(mamba_d_ssm=48, mamba_n_groups=2, mamba_d_state=8)
        )
        assert getattr(a.cfg, "conv_dim") == 48 + 2 * 2 * 8

    def test_mamba_intermediate_equals_d_ssm(self) -> None:
        a = FalconH1ArchitectureAdapter(_make_cfg(mamba_d_ssm=96))
        assert getattr(a.cfg, "mamba_intermediate_size") == 96


# ---------------------------------------------------------------------------
# Top-level component mapping
# ---------------------------------------------------------------------------


class TestFalconH1TopLevelComponents:
    """component_mapping has exactly the expected top-level keys."""

    def test_required_keys(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_rotary_emb_present(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # Falcon-H1 exposes a model-level rotary module.
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_embed(self, adapter: FalconH1ArchitectureAdapter) -> None:
        embed = adapter.component_mapping["embed"]
        assert isinstance(embed, EmbeddingBridge)
        assert embed.name == "model.embed_tokens"

    def test_blocks_is_block_bridge(self, adapter: FalconH1ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final(self, adapter: FalconH1ArchitectureAdapter) -> None:
        ln_final = adapter.component_mapping["ln_final"]
        assert isinstance(ln_final, RMSNormalizationBridge)
        assert ln_final.name == "model.final_layernorm"

    def test_unembed(self, adapter: FalconH1ArchitectureAdapter) -> None:
        unembed = adapter.component_mapping["unembed"]
        assert isinstance(unembed, UnembeddingBridge)
        assert unembed.name == "lm_head"


# ---------------------------------------------------------------------------
# Block-level submodules (the parallel hybrid shape)
# ---------------------------------------------------------------------------


class TestFalconH1BlockSubmodules:
    """The block carries two norms plus parallel attn + mamba and a SwiGLU MLP."""

    @pytest.fixture(scope="class")
    def blocks(self, adapter: FalconH1ArchitectureAdapter) -> BlockBridge:
        return adapter.component_mapping["blocks"]

    def test_ln1_is_input_layernorm(self, blocks: BlockBridge) -> None:
        ln1 = blocks.submodules["ln1"]
        assert isinstance(ln1, RMSNormalizationBridge)
        assert ln1.name == "input_layernorm"

    def test_ln2_is_pre_ff_layernorm(self, blocks: BlockBridge) -> None:
        # Two-norm block: pre_ff_layernorm gives a real hook_resid_mid.
        ln2 = blocks.submodules["ln2"]
        assert isinstance(ln2, RMSNormalizationBridge)
        assert ln2.name == "pre_ff_layernorm"

    def test_attn_branch_present(self, blocks: BlockBridge) -> None:
        attn = blocks.submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"

    def test_mamba_branch_present(self, blocks: BlockBridge) -> None:
        mamba = blocks.submodules["mamba"]
        assert isinstance(mamba, SSM2MixerBridge)
        assert mamba.name == "mamba"

    def test_both_branches_present_in_every_block(self, blocks: BlockBridge) -> None:
        # The defining feature of Falcon-H1: attn AND mamba in parallel, not a
        # single per-layer mixer slot.
        assert "attn" in blocks.submodules
        assert "mamba" in blocks.submodules

    def test_mlp_is_swiglu(self, blocks: BlockBridge) -> None:
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "feed_forward"

    def test_block_submodule_keys(self, blocks: BlockBridge) -> None:
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mamba", "mlp"}


# ---------------------------------------------------------------------------
# Attention submodules
# ---------------------------------------------------------------------------


class TestFalconH1AttentionSubmodules:
    """Split q/k/v/o projections (GQA)."""

    @pytest.fixture(scope="class")
    def attn(self, adapter: FalconH1ArchitectureAdapter) -> PositionEmbeddingsAttentionBridge:
        return adapter.component_mapping["blocks"].submodules["attn"]

    @pytest.mark.parametrize(
        "key,hf_name",
        [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")],
    )
    def test_projection(
        self, attn: PositionEmbeddingsAttentionBridge, key: str, hf_name: str
    ) -> None:
        proj = attn.submodules[key]
        assert isinstance(proj, LinearBridge)
        assert proj.name == hf_name


# ---------------------------------------------------------------------------
# Mamba mixer submodules
# ---------------------------------------------------------------------------


class TestFalconH1MambaSubmodules:
    """SSM2MixerBridge submodules; inner_norm is optional (mamba_rms_norm=false)."""

    @pytest.fixture(scope="class")
    def mamba(self, adapter: FalconH1ArchitectureAdapter) -> SSM2MixerBridge:
        return adapter.component_mapping["blocks"].submodules["mamba"]

    def test_in_proj(self, mamba: SSM2MixerBridge) -> None:
        assert isinstance(mamba.submodules["in_proj"], LinearBridge)
        assert mamba.submodules["in_proj"].name == "in_proj"

    def test_conv1d(self, mamba: SSM2MixerBridge) -> None:
        assert isinstance(mamba.submodules["conv1d"], DepthwiseConv1DBridge)
        assert mamba.submodules["conv1d"].name == "conv1d"

    def test_out_proj(self, mamba: SSM2MixerBridge) -> None:
        assert isinstance(mamba.submodules["out_proj"], LinearBridge)
        assert mamba.submodules["out_proj"].name == "out_proj"

    def test_inner_norm_is_gated_and_optional(self, mamba: SSM2MixerBridge) -> None:
        inner_norm = mamba.submodules["inner_norm"]
        assert isinstance(inner_norm, GatedRMSNormBridge)
        # HF calls it "norm" inside the mixer; absent when mamba_rms_norm=false.
        assert inner_norm.name == "norm"
        assert inner_norm.optional is True

    def test_mamba_submodule_keys(self, mamba: SSM2MixerBridge) -> None:
        assert set(mamba.submodules.keys()) == {"in_proj", "conv1d", "inner_norm", "out_proj"}


# ---------------------------------------------------------------------------
# MLP submodules
# ---------------------------------------------------------------------------


class TestFalconH1MLPSubmodules:
    """SwiGLU feed-forward: gate_proj / up_proj / down_proj."""

    @pytest.fixture(scope="class")
    def mlp(self, adapter: FalconH1ArchitectureAdapter) -> GatedMLPBridge:
        return adapter.component_mapping["blocks"].submodules["mlp"]

    @pytest.mark.parametrize(
        "key,hf_name",
        [("gate", "gate_proj"), ("in", "up_proj"), ("out", "down_proj")],
    )
    def test_projection(self, mlp: GatedMLPBridge, key: str, hf_name: str) -> None:
        proj = mlp.submodules[key]
        assert isinstance(proj, LinearBridge)
        assert proj.name == hf_name


# ---------------------------------------------------------------------------
# Factory + registry registration
# ---------------------------------------------------------------------------


class TestFalconH1FactoryRegistration:
    def test_factory_returns_falcon_h1_adapter(self) -> None:
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(_make_cfg())
        assert isinstance(adapter, FalconH1ArchitectureAdapter)

    def test_architecture_key_present(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert SUPPORTED_ARCHITECTURES["FalconH1ForCausalLM"] is FalconH1ArchitectureAdapter


class TestFalconH1ModelRegistry:
    def test_in_hf_supported_architectures(self) -> None:
        from transformer_lens.tools.model_registry import HF_SUPPORTED_ARCHITECTURES

        assert "FalconH1ForCausalLM" in HF_SUPPORTED_ARCHITECTURES

    def test_canonical_author_is_tiiuae(self) -> None:
        from transformer_lens.tools.model_registry import CANONICAL_AUTHORS_BY_ARCH

        assert CANONICAL_AUTHORS_BY_ARCH.get("FalconH1ForCausalLM") == ["tiiuae"]


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


class TestFalconH1Guards:
    """Guards against drift toward neighbouring adapter patterns."""

    def test_uses_block_bridge_not_ssm_block_bridge(
        self, adapter: FalconH1ArchitectureAdapter
    ) -> None:
        # Two-norm parallel hybrid → BlockBridge (real resid_mid), unlike the
        # single-norm SSMBlockBridge used by NemotronH / Mamba2.
        from transformer_lens.model_bridge.generalized_components import SSMBlockBridge

        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, BlockBridge)
        assert not isinstance(blocks, SSMBlockBridge)

    def test_block_has_ln2(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # ln2 (pre_ff_layernorm) must exist or BlockBridge raises at construction.
        assert "ln2" in adapter.component_mapping["blocks"].submodules

    def test_no_weight_conversions_defined(self, adapter: FalconH1ArchitectureAdapter) -> None:
        # Passthrough mode → HF applies the ~12 multipliers; no folding/rearrange.
        assert len(adapter.weight_processing_conversions) == 0
