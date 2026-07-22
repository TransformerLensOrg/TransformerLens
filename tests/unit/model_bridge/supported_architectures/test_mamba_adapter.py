"""Unit tests for MambaArchitectureAdapter — programmatic configs only.

Tests cover:
- Config attribute validation (RMS norm, no positional embeddings, stateful)
- Component mapping structure and HF module paths
- SSM block structure with norm and mixer submodules
- SSM mixer submodule structure (in_proj, conv1d, x_proj, dt_proj, out_proj)
- Factory registration
- Weight conversions (empty for SSM)
"""

from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    SSMBlockBridge,
    SSMMixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.mamba import (
    MambaArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 256,
    n_ctx: int = 128,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Mamba adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        default_prepend_bos=True,
        architecture="MambaForCausalLM",
    )


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> MambaArchitectureAdapter:
    return MambaArchitectureAdapter(cfg)


def _mapping(adapter: MambaArchitectureAdapter) -> dict[str, Any]:
    """Narrow component_mapping to non-None dict."""
    mapping = adapter.component_mapping
    assert mapping is not None
    return mapping


class TestMambaAdapterConfig:
    """Adapter sets all required config attributes for Mamba-1 SSM models."""

    def test_normalization_type_is_rms(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_uses_rms_norm_is_true(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_positional_embedding_type_is_none(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "none"

    def test_gated_mlp_is_false(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_final_rms_is_true(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_is_stateful_is_true(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.cfg.is_stateful is True


class TestMambaWeightConversions:
    """Mamba has no Q/K/V/O weight conversions (SSM architecture)."""

    def test_weight_conversions_is_empty(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions == {}


class TestMambaApplicablePhases:
    """Mamba only applies to Phase 4 (generation + text quality)."""

    def test_applicable_phases_is_phase_4_only(self, adapter: MambaArchitectureAdapter) -> None:
        assert adapter.applicable_phases == [4]


class TestMambaComponentMapping:
    """Component mapping must have correct bridge types and HF module paths."""

    def test_has_required_top_level_keys(self, adapter: MambaArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        for key in ("embed", "blocks", "ln_final", "unembed"):
            assert key in mapping, f"Missing top-level key: {key!r}"

    def test_no_rotary_emb_key(self, adapter: MambaArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert "rotary_emb" not in mapping

    def test_no_pos_embed_key(self, adapter: MambaArchitectureAdapter) -> None:
        mapping = _mapping(adapter)
        assert "pos_embed" not in mapping

    def test_embed_is_embedding_bridge(self, adapter: MambaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: MambaArchitectureAdapter) -> None:
        assert _mapping(adapter)["embed"].name == "backbone.embeddings"

    def test_blocks_is_ssm_block_bridge(self, adapter: MambaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["blocks"], SSMBlockBridge)

    def test_blocks_name(self, adapter: MambaArchitectureAdapter) -> None:
        assert _mapping(adapter)["blocks"].name == "backbone.layers"

    def test_ln_final_is_rms_normalization_bridge(self, adapter: MambaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["ln_final"], RMSNormalizationBridge)

    def test_ln_final_name(self, adapter: MambaArchitectureAdapter) -> None:
        assert _mapping(adapter)["ln_final"].name == "backbone.norm_f"

    def test_unembed_is_unembedding_bridge(self, adapter: MambaArchitectureAdapter) -> None:
        assert isinstance(_mapping(adapter)["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: MambaArchitectureAdapter) -> None:
        assert _mapping(adapter)["unembed"].name == "lm_head"


class TestMambaBlockSubmodules:
    """SSM block submodules: norm and mixer."""

    def test_blocks_has_required_submodules(self, adapter: MambaArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        for key in ("norm", "mixer"):
            assert key in blocks.submodules, f"Missing blocks submodule: {key!r}"

    def test_norm_is_rms_normalization_bridge(self, adapter: MambaArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["norm"], RMSNormalizationBridge)

    def test_norm_name(self, adapter: MambaArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        assert blocks.submodules["norm"].name == "norm"

    def test_mixer_is_ssm_mixer_bridge(self, adapter: MambaArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["mixer"], SSMMixerBridge)

    def test_mixer_name(self, adapter: MambaArchitectureAdapter) -> None:
        blocks = _mapping(adapter)["blocks"]
        assert blocks.submodules["mixer"].name == "mixer"


class TestMambaMixerSubmodules:
    """SSM mixer submodules: in_proj, conv1d, x_proj, dt_proj, out_proj."""

    @pytest.fixture(scope="class")
    def mixer(self, adapter: MambaArchitectureAdapter) -> SSMMixerBridge:
        blocks = _mapping(adapter)["blocks"]
        mixer = blocks.submodules["mixer"]
        assert isinstance(mixer, SSMMixerBridge)
        return mixer

    def test_mixer_has_in_proj(self, mixer: SSMMixerBridge) -> None:
        assert "in_proj" in mixer.submodules
        assert isinstance(mixer.submodules["in_proj"], LinearBridge)
        assert mixer.submodules["in_proj"].name == "in_proj"

    def test_mixer_has_conv1d(self, mixer: SSMMixerBridge) -> None:
        assert "conv1d" in mixer.submodules
        assert isinstance(mixer.submodules["conv1d"], DepthwiseConv1DBridge)
        assert mixer.submodules["conv1d"].name == "conv1d"

    def test_mixer_has_x_proj(self, mixer: SSMMixerBridge) -> None:
        assert "x_proj" in mixer.submodules
        assert isinstance(mixer.submodules["x_proj"], LinearBridge)
        assert mixer.submodules["x_proj"].name == "x_proj"

    def test_mixer_has_dt_proj(self, mixer: SSMMixerBridge) -> None:
        assert "dt_proj" in mixer.submodules
        assert isinstance(mixer.submodules["dt_proj"], LinearBridge)
        assert mixer.submodules["dt_proj"].name == "dt_proj"

    def test_mixer_has_out_proj(self, mixer: SSMMixerBridge) -> None:
        assert "out_proj" in mixer.submodules
        assert isinstance(mixer.submodules["out_proj"], LinearBridge)
        assert mixer.submodules["out_proj"].name == "out_proj"

    def test_mixer_has_exactly_five_submodules(self, mixer: SSMMixerBridge) -> None:
        expected = {"in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"}
        assert set(mixer.submodules.keys()) == expected


class TestMambaFactoryRegistration:
    """MambaForCausalLM architecture dispatches to this adapter."""

    def test_architecture_registered_in_supported_architectures(self) -> None:
        assert "MambaForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_returns_correct_adapter_type(self) -> None:
        cfg = _make_cfg()
        adapter_cls = SUPPORTED_ARCHITECTURES["MambaForCausalLM"]
        adapter = adapter_cls(cfg)
        assert isinstance(adapter, MambaArchitectureAdapter)


class TestMambaArchitectureGuards:
    """Guards against drift from Mamba conventions."""

    def test_no_attention_submodules(self, adapter: MambaArchitectureAdapter) -> None:
        """SSM blocks have no attention — guard against accidental transformer patterns."""
        blocks = _mapping(adapter)["blocks"]
        assert "attn" not in blocks.submodules
        assert "ln1" not in blocks.submodules
        assert "ln2" not in blocks.submodules

    def test_no_mlp_submodule(self, adapter: MambaArchitectureAdapter) -> None:
        """SSM uses mixer instead of MLP."""
        blocks = _mapping(adapter)["blocks"]
        assert "mlp" not in blocks.submodules

    def test_blocks_use_ssm_block_bridge_not_block_bridge(
        self, adapter: MambaArchitectureAdapter
    ) -> None:
        """Mamba uses SSMBlockBridge, not the transformer BlockBridge."""
        blocks = _mapping(adapter)["blocks"]
        assert isinstance(blocks, SSMBlockBridge)
        assert type(blocks) is SSMBlockBridge
