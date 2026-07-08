"""Unit tests for GiddArchitectureAdapter.

Uniform-noise diffusion: bidirectional softcap attention delegates,
QK norms are optional (config.use_qk_norm), ScaledLinear projections
rule out folding, and generation is the model's own diffusion sampler.
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    MLPBridge,
)
from transformer_lens.model_bridge.supported_architectures.gidd import (
    GiddArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    return make_bridge_cfg("GiddForDiffusionLM", **overrides)


@pytest.fixture
def adapter() -> GiddArchitectureAdapter:
    return GiddArchitectureAdapter(_make_cfg())


class TestGiddPhases:
    def test_diffusion_treatment(self, adapter):
        assert adapter.applicable_phases == [1, 2, 3]
        assert adapter.supports_generation is False
        assert adapter.supports_fold_ln is False


class TestGiddMapping:
    def test_attention_delegated_with_optional_qk_norms(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert type(attn) is AttentionBridge
        assert attn.maintain_native_attention is True
        assert attn.submodules["q_norm"].optional is True
        assert attn.submodules["k_norm"].optional is True

    def test_gidd_norm_names(self, adapter):
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "attn_layernorm"
        assert blocks.submodules["ln2"].name == "mlp_layernorm"

    def test_ungated_mlp(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.submodules["in"].name == "up_proj"

    def test_no_rotary_module_entry(self, adapter):
        """Rotary lives as a model-level buffer, not a module."""
        assert "rotary_emb" not in adapter.component_mapping


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["GiddForDiffusionLM"] is GiddArchitectureAdapter
