"""Unit tests for Mamba2ArchitectureAdapter — programmatic configs only.

Tests cover:
- SSM-specific config (is_stateful marker)
- Computed config fields (intermediate_size, conv_dim, expected_in_proj_out_features)
- Component mapping structure and HF module paths
- SSM block structure with norm and mixer submodules
- SSM2 mixer submodule structure (in_proj, conv1d, inner_norm, out_proj — no x_proj/dt_proj)
- Weight conversions (empty for SSM)
"""

from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
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
from transformer_lens.model_bridge.supported_architectures.mamba2 import (
    Mamba2ArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 256,
    n_ctx: int = 128,
    expand: int = 2,
    state_size: int = 128,
    n_groups: int = 1,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Mamba2 adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        default_prepend_bos=True,
        architecture="Mamba2ForCausalLM",
    )
    setattr(cfg, "expand", expand)
    setattr(cfg, "state_size", state_size)
    setattr(cfg, "n_groups", n_groups)
    return cfg


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> Mamba2ArchitectureAdapter:
    return Mamba2ArchitectureAdapter(cfg)


def _mapping(adapter: Mamba2ArchitectureAdapter) -> dict[str, Any]:
    """Narrow component_mapping to non-None dict."""
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


class TestMamba2AdapterConfig:
    """SSM-specific config marker."""

    def test_is_stateful_is_true(self, adapter: Mamba2ArchitectureAdapter) -> None:
        assert adapter.cfg.is_stateful is True


class TestMamba2ComputedConfig:
    """Mamba2 computes additional config fields from base HF config."""

    def test_intermediate_size_computed(self, adapter: Mamba2ArchitectureAdapter) -> None:
        expand = getattr(adapter.cfg, "expand", 2)
        expected = expand * adapter.cfg.d_model
        assert getattr(adapter.cfg, "intermediate_size") == expected

    def test_conv_dim_computed(self, adapter: Mamba2ArchitectureAdapter) -> None:
        intermediate = getattr(adapter.cfg, "intermediate_size")
        n_groups = getattr(adapter.cfg, "n_groups")
        state_size = getattr(adapter.cfg, "state_size")
        expected = intermediate + 2 * n_groups * state_size
        assert getattr(adapter.cfg, "conv_dim") == expected

    def test_expected_in_proj_out_features_computed(
        self, adapter: Mamba2ArchitectureAdapter
    ) -> None:
        intermediate = getattr(adapter.cfg, "intermediate_size")
        conv_dim = getattr(adapter.cfg, "conv_dim")
        num_heads = adapter.cfg.n_heads
        expected = 2 * intermediate + conv_dim + num_heads
        assert getattr(adapter.cfg, "expected_in_proj_out_features") == expected

    def test_computed_values_with_custom_expand(self) -> None:
        cfg = _make_cfg(d_model=64, expand=4, n_groups=2, state_size=64, n_heads=8)
        adapter = Mamba2ArchitectureAdapter(cfg)
        assert getattr(adapter.cfg, "intermediate_size") == 256
        conv_dim = getattr(adapter.cfg, "conv_dim")
        assert conv_dim == 256 + 2 * 2 * 64
        expected_proj = 2 * 256 + conv_dim + 8
        assert getattr(adapter.cfg, "expected_in_proj_out_features") == expected_proj


class TestMamba2WeightConversions:
    """Mamba2 has no Q/K/V/O weight conversions (SSM architecture)."""

    def test_weight_conversions_is_empty(self, adapter: Mamba2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions == {}


class TestMamba2ApplicablePhases:
    """Mamba2 only applies to Phase 4 (generation + text quality)."""

    def test_applicable_phases_is_phase_4_only(self, adapter: Mamba2ArchitectureAdapter) -> None:
        assert adapter.applicable_phases == [4]


class TestMamba2ComponentMapping:
    """Component mapping must have correct bridge types and HF module paths."""

    def test_has_required_top_level_keys(self, adapter: Mamba2ArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        for key in ("embed", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_no_rotary_emb_key(self, adapter: Mamba2ArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert "rotary_emb" not in mapping

    def test_no_pos_embed_key(self, adapter: Mamba2ArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert "pos_embed" not in mapping

    def test_embed(self, adapter: Mamba2ArchitectureAdapter) -> None:
        embed = _mapping(adapter)["embed"]
        assert isinstance(embed, EmbeddingBridge)
        assert embed.name == "backbone.embeddings"

    def test_blocks(self, adapter: Mamba2ArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        assert isinstance(blocks, SSMBlockBridge)
        assert blocks.name == "backbone.layers"

    def test_ln_final(self, adapter: Mamba2ArchitectureAdapter) -> None:
        ln_final = _mapping(adapter)["ln_final"]
        assert isinstance(ln_final, RMSNormalizationBridge)
        assert ln_final.name == "backbone.norm_f"

    def test_unembed(self, adapter: Mamba2ArchitectureAdapter) -> None:
        unembed = _mapping(adapter)["unembed"]
        assert isinstance(unembed, UnembeddingBridge)
        assert unembed.name == "lm_head"


class TestMamba2BlockSubmodules:
    """SSM block submodules: norm and mixer."""

    def test_blocks_has_required_submodules(self, adapter: Mamba2ArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        for key in ("norm", "mixer"):
            assert key in blocks.submodules, f"Missing blocks submodule: {key!r}"

    def test_norm(self, adapter: Mamba2ArchitectureAdapter) -> None:
        norm = _mapping(adapter)["blocks"].submodules["norm"]
        assert isinstance(norm, RMSNormalizationBridge)
        assert norm.name == "norm"

    def test_mixer(self, adapter: Mamba2ArchitectureAdapter) -> None:
        mixer = _mapping(adapter)["blocks"].submodules["mixer"]
        assert isinstance(mixer, SSM2MixerBridge)
        assert mixer.name == "mixer"


class TestMamba2MixerSubmodules:
    """SSM2 mixer submodules: in_proj, conv1d, inner_norm, out_proj (no x_proj/dt_proj)."""

    @pytest.fixture(scope="class")
    def mixer(self, adapter: Mamba2ArchitectureAdapter) -> SSM2MixerBridge:
        blocks = _mapping(adapter)["blocks"]
        mixer = blocks.submodules["mixer"]
        assert isinstance(mixer, SSM2MixerBridge)
        return mixer

    def test_mixer_has_in_proj(self, mixer: SSM2MixerBridge) -> None:
        assert "in_proj" in mixer.submodules
        assert isinstance(mixer.submodules["in_proj"], LinearBridge)
        assert mixer.submodules["in_proj"].name == "in_proj"

    def test_mixer_has_conv1d(self, mixer: SSM2MixerBridge) -> None:
        assert "conv1d" in mixer.submodules
        assert isinstance(mixer.submodules["conv1d"], DepthwiseConv1DBridge)
        assert mixer.submodules["conv1d"].name == "conv1d"

    def test_mixer_has_inner_norm(self, mixer: SSM2MixerBridge) -> None:
        assert "inner_norm" in mixer.submodules
        assert isinstance(mixer.submodules["inner_norm"], GatedRMSNormBridge)
        assert mixer.submodules["inner_norm"].name == "norm"

    def test_mixer_has_out_proj(self, mixer: SSM2MixerBridge) -> None:
        assert "out_proj" in mixer.submodules
        assert isinstance(mixer.submodules["out_proj"], LinearBridge)
        assert mixer.submodules["out_proj"].name == "out_proj"

    def test_mixer_has_exactly_four_submodules(self, mixer: SSM2MixerBridge) -> None:
        expected = {"in_proj", "conv1d", "inner_norm", "out_proj"}
        assert set(mixer.submodules.keys()) == expected

    def test_mixer_no_x_proj(self, mixer: SSM2MixerBridge) -> None:
        assert "x_proj" not in mixer.submodules

    def test_mixer_no_dt_proj(self, mixer: SSM2MixerBridge) -> None:
        assert "dt_proj" not in mixer.submodules


class TestMamba2ArchitectureGuards:
    """Guards against drift from Mamba2 conventions."""

    def test_no_attention_submodules(self, adapter: Mamba2ArchitectureAdapter) -> None:
        """SSM blocks have no attention — guard against accidental transformer patterns."""
        blocks = _mapping(adapter)["blocks"]
        assert "attn" not in blocks.submodules
        assert "ln1" not in blocks.submodules
        assert "ln2" not in blocks.submodules

    def test_no_mlp_submodule(self, adapter: Mamba2ArchitectureAdapter) -> None:
        """SSM uses mixer instead of MLP."""
        blocks = _mapping(adapter)["blocks"]
        assert "mlp" not in blocks.submodules
