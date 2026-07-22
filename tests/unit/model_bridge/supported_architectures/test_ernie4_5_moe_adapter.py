"""Unit tests for the Ernie4_5_MoeArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    MoEBridge,
)
from transformer_lens.model_bridge.supported_architectures.ernie4_5_moe import (
    Ernie4_5_MoeArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Ernie4_5_MoeForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> Ernie4_5_MoeArchitectureAdapter:
    return Ernie4_5_MoeArchitectureAdapter(_make_cfg())


class TestErnie4_5MoeAdapterConfig:
    def test_shares_dense_ernie_conventions(self, adapter):
        assert adapter.cfg.rotary_adjacent_pairs is True
        assert adapter.cfg.default_prepend_bos is False
        assert adapter.cfg.normalization_type == "RMS"


class TestErnie4_5MoeComponentMapping:
    def test_moe_with_optional_shared_experts(self, adapter):
        """Dense-prefix layers hold a plain gated MLP — shared experts and the
        router are absent there, so shared_experts is optional and the custom
        sigmoid router is fully delegated."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert set(mlp.submodules) == {"gate", "shared_experts"}
        assert mlp.submodules["gate"].optional is True
        shared = mlp.submodules["shared_experts"]
        assert isinstance(shared, GatedMLPBridge)
        assert shared.optional is True
        assert shared.submodules["gate"].name == "gate_proj"


class TestErnie4_5MoeRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Ernie4_5_MoeForCausalLM"] is Ernie4_5_MoeArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="ernie4_5_moe", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Ernie4_5_MoeForCausalLM"
