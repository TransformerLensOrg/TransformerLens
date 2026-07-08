"""Unit tests for SwitchTransformersArchitectureAdapter.

The foundational top-1 capacity-routed MoE on the T5 skeleton: the FF seam
swaps in a delegated MoEBridge whose router is optional (dense on even
layers, sparse on odd).
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.supported_architectures.switch_transformers import (
    SwitchTransformersArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.t5 import (
    T5ArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    return make_bridge_cfg("SwitchTransformersForConditionalGeneration", **overrides)


@pytest.fixture
def adapter() -> SwitchTransformersArchitectureAdapter:
    return SwitchTransformersArchitectureAdapter(_make_cfg())


class TestSwitchMapping:
    def test_is_t5_shaped(self, adapter):
        assert isinstance(adapter, T5ArchitectureAdapter)
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "shared"

    def test_ff_is_moe_with_optional_router(self, adapter):
        enc_mlp = adapter.component_mapping["encoder_blocks"].submodules["mlp"]
        dec_mlp = adapter.component_mapping["decoder_blocks"].submodules["mlp"]
        assert isinstance(enc_mlp, MoEBridge) and enc_mlp.name == "layer.1.mlp"
        assert isinstance(dec_mlp, MoEBridge) and dec_mlp.name == "layer.2.mlp"
        assert enc_mlp.submodules["gate"].optional is True
        assert enc_mlp.submodules["gate"].name == "router"


def test_factory_registration():
    assert (
        SUPPORTED_ARCHITECTURES["SwitchTransformersForConditionalGeneration"]
        is SwitchTransformersArchitectureAdapter
    )
