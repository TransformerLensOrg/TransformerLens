"""Unit tests for NemotronHArchitectureAdapter.

Covers: config attribute propagation, component mapping bridge types and HF
path names, Mamba-specific submodule optional flag, applicable_phases,
create_stateful_cache, factory registration, and guard tests.
"""

from unittest.mock import MagicMock

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedRMSNormBridge,
    LinearBridge,
    RMSNormalizationBridge,
    SSM2MixerBridge,
    SSMBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.nemotron_h import (
    NemotronHArchitectureAdapter,
)


def _make_cfg(
    n_layers: int = 3,
    d_model: int = 64,
    d_head: int = 8,
    n_heads: int = 8,
    d_vocab: int = 100,
    n_ctx: int = 128,
    mamba_num_heads: int = 4,
    mamba_head_dim: int = 8,
    n_groups: int = 2,
    ssm_state_size: int = 4,
    layers_block_type: list[str] | None = None,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for NemotronH adapter tests.

    Uses small Mamba-2 dimensions so tests run without loading any weights.
    """
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_head,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        default_prepend_bos=False,
        architecture="NemotronHForCausalLM",
    )
    # Inject NemotronH-specific fields the adapter reads via getattr.
    cfg.mamba_num_heads = mamba_num_heads  # type: ignore[attr-defined]
    cfg.mamba_head_dim = mamba_head_dim  # type: ignore[attr-defined]
    cfg.n_groups = n_groups  # type: ignore[attr-defined]
    cfg.ssm_state_size = ssm_state_size  # type: ignore[attr-defined]
    if layers_block_type is not None:
        cfg.layers_block_type = layers_block_type  # type: ignore[attr-defined]
    return cfg


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> NemotronHArchitectureAdapter:
    return NemotronHArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attributes
# ---------------------------------------------------------------------------


class TestNemotronHAdapterConfig:
    """Adapter propagates all required config attributes."""

    def test_normalization_type_rms(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_uses_rms_norm(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_positional_embedding_type_none(self, adapter: NemotronHArchitectureAdapter) -> None:
        # No model-level rotary module — attention handles RoPE internally.
        assert adapter.cfg.positional_embedding_type == "none"

    def test_gated_mlp_false(self, adapter: NemotronHArchitectureAdapter) -> None:
        # MLP layers use relu2, not SwiGLU.
        assert adapter.cfg.gated_mlp is False

    def test_final_rms_true(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_is_stateful_true(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.cfg.is_stateful is True

    def test_mamba_intermediate_size_propagated(
        self, adapter: NemotronHArchitectureAdapter
    ) -> None:
        # mamba_num_heads=4, mamba_head_dim=8 → 32
        assert getattr(adapter.cfg, "mamba_intermediate_size", None) == 32

    def test_conv_dim_propagated(self, adapter: NemotronHArchitectureAdapter) -> None:
        # intermediate=32, n_groups=2, ssm_state_size=4 → 32 + 2*2*4 = 48
        assert getattr(adapter.cfg, "conv_dim", None) == 48

    def test_applicable_phases_full(self) -> None:
        assert NemotronHArchitectureAdapter.applicable_phases == [1, 2, 3, 4]

    def test_weight_processing_conversions_empty(
        self, adapter: NemotronHArchitectureAdapter
    ) -> None:
        assert adapter.weight_processing_conversions == {}

    def test_layers_block_type_defaults_to_empty(self) -> None:
        cfg = _make_cfg()
        a = NemotronHArchitectureAdapter(cfg)
        assert getattr(a.cfg, "layers_block_type") == []


# ---------------------------------------------------------------------------
# Top-level component mapping
# ---------------------------------------------------------------------------


class TestNemotronHTopLevelComponents:
    """component_mapping has exactly the expected top-level keys."""

    def test_required_keys(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {"embed", "blocks", "ln_final", "unembed"}

    def test_embed_is_embedding_bridge(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embeddings"

    def test_blocks_is_ssm_block_bridge(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], SSMBlockBridge)

    def test_blocks_name(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_is_rms_normalization_bridge(
        self, adapter: NemotronHArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_ln_final_name(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.norm_f"

    def test_unembed_is_unembedding_bridge(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: NemotronHArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"


# ---------------------------------------------------------------------------
# Block-level submodules
# ---------------------------------------------------------------------------


class TestNemotronHBlockSubmodules:
    """SSMBlockBridge submodules have correct types and HF path names."""

    @pytest.fixture(scope="class")
    def blocks(self, adapter: NemotronHArchitectureAdapter) -> SSMBlockBridge:
        return adapter.component_mapping["blocks"]

    def test_norm_is_rms_normalization_bridge(self, blocks: SSMBlockBridge) -> None:
        assert isinstance(blocks.submodules["norm"], RMSNormalizationBridge)

    def test_norm_name(self, blocks: SSMBlockBridge) -> None:
        assert blocks.submodules["norm"].name == "norm"

    def test_mixer_is_ssm2_mixer_bridge(self, blocks: SSMBlockBridge) -> None:
        assert isinstance(blocks.submodules["mixer"], SSM2MixerBridge)

    def test_mixer_name(self, blocks: SSMBlockBridge) -> None:
        assert blocks.submodules["mixer"].name == "mixer"

    def test_block_has_no_ln2(self, blocks: SSMBlockBridge) -> None:
        # Single pre-norm architecture; no post-attention norm.
        assert "ln2" not in blocks.submodules


# ---------------------------------------------------------------------------
# Mixer submodules (Mamba-specific, optional)
# ---------------------------------------------------------------------------


class TestNemotronHMixerSubmodules:
    """SSM2MixerBridge submodules are Mamba-specific and optional."""

    @pytest.fixture(scope="class")
    def mixer(self, adapter: NemotronHArchitectureAdapter) -> SSM2MixerBridge:
        return adapter.component_mapping["blocks"].submodules["mixer"]

    def test_in_proj_is_linear_bridge(self, mixer: SSM2MixerBridge) -> None:
        assert isinstance(mixer.submodules["in_proj"], LinearBridge)

    def test_in_proj_name(self, mixer: SSM2MixerBridge) -> None:
        assert mixer.submodules["in_proj"].name == "in_proj"

    def test_in_proj_optional(self, mixer: SSM2MixerBridge) -> None:
        assert mixer.submodules["in_proj"].optional is True

    def test_conv1d_is_depthwise_bridge(self, mixer: SSM2MixerBridge) -> None:
        assert isinstance(mixer.submodules["conv1d"], DepthwiseConv1DBridge)

    def test_conv1d_name(self, mixer: SSM2MixerBridge) -> None:
        assert mixer.submodules["conv1d"].name == "conv1d"

    def test_conv1d_optional(self, mixer: SSM2MixerBridge) -> None:
        assert mixer.submodules["conv1d"].optional is True

    def test_inner_norm_is_gated_rms_norm_bridge(self, mixer: SSM2MixerBridge) -> None:
        assert isinstance(mixer.submodules["inner_norm"], GatedRMSNormBridge)

    def test_inner_norm_name(self, mixer: SSM2MixerBridge) -> None:
        # HF calls it "norm" inside the mixer; TL aliases to "inner_norm".
        assert mixer.submodules["inner_norm"].name == "norm"

    def test_inner_norm_optional(self, mixer: SSM2MixerBridge) -> None:
        assert mixer.submodules["inner_norm"].optional is True

    def test_out_proj_is_linear_bridge(self, mixer: SSM2MixerBridge) -> None:
        assert isinstance(mixer.submodules["out_proj"], LinearBridge)

    def test_out_proj_name(self, mixer: SSM2MixerBridge) -> None:
        assert mixer.submodules["out_proj"].name == "out_proj"

    def test_out_proj_optional(self, mixer: SSM2MixerBridge) -> None:
        assert mixer.submodules["out_proj"].optional is True

    def test_mixer_has_exactly_four_mamba_submodules(self, mixer: SSM2MixerBridge) -> None:
        assert set(mixer.submodules.keys()) == {"in_proj", "conv1d", "inner_norm", "out_proj"}


# ---------------------------------------------------------------------------
# create_stateful_cache
# ---------------------------------------------------------------------------


class TestNemotronHStatefulCache:
    """create_stateful_cache returns a DynamicCache instance."""

    def test_returns_dynamic_cache(self, adapter: NemotronHArchitectureAdapter) -> None:
        from transformers.cache_utils import DynamicCache

        hf_model = MagicMock()
        cache = adapter.create_stateful_cache(
            hf_model=hf_model, batch_size=1, device="cpu", dtype=None
        )
        assert isinstance(cache, DynamicCache)

    def test_cache_independent_per_call(self, adapter: NemotronHArchitectureAdapter) -> None:
        """Each call returns a fresh cache object."""
        hf_model = MagicMock()
        c1 = adapter.create_stateful_cache(hf_model, 1, "cpu", None)
        c2 = adapter.create_stateful_cache(hf_model, 1, "cpu", None)
        assert c1 is not c2


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------


class TestNemotronHFactoryRegistration:
    """NemotronHForCausalLM is registered in the adapter factory."""

    def test_factory_returns_nemotron_h_adapter(self) -> None:
        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, NemotronHArchitectureAdapter)

    def test_architecture_key_present(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "NemotronHForCausalLM" in SUPPORTED_ARCHITECTURES
        assert SUPPORTED_ARCHITECTURES["NemotronHForCausalLM"] is NemotronHArchitectureAdapter


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestNemotronHModelRegistry:
    """NemotronHForCausalLM is listed in the model registry constants."""

    def test_canonical_author_is_nvidia(self) -> None:
        from transformer_lens.tools.model_registry import CANONICAL_AUTHORS_BY_ARCH

        assert CANONICAL_AUTHORS_BY_ARCH.get("NemotronHForCausalLM") == ["nvidia"]


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


class TestNemotronHGuards:
    """Guards against drift toward neighbouring adapter patterns."""

    def test_mamba_intermediate_size_formula(self) -> None:
        """Verify formula: intermediate = mamba_num_heads * mamba_head_dim."""
        cfg = _make_cfg(mamba_num_heads=16, mamba_head_dim=32)
        a = NemotronHArchitectureAdapter(cfg)
        assert getattr(a.cfg, "mamba_intermediate_size") == 16 * 32

    def test_conv_dim_formula(self) -> None:
        """Verify formula: conv_dim = intermediate + 2 * n_groups * ssm_state_size."""
        cfg = _make_cfg(mamba_num_heads=8, mamba_head_dim=16, n_groups=4, ssm_state_size=8)
        a = NemotronHArchitectureAdapter(cfg)
        expected = 8 * 16 + 2 * 4 * 8  # 128 + 64 = 192
        assert getattr(a.cfg, "conv_dim") == expected
