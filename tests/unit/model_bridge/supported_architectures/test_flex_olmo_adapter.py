"""Unit tests for FlexOlmoArchitectureAdapter.

FlexOlmo is the union of OLMo-2 (post-norm blocks, full-width q/k norms)
and OLMoE (batched-expert sparse MoE); only the MLP seam differs from
the OLMo-2 parent.
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    MoEBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.flex_olmo import (
    FlexOlmoArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    return make_bridge_cfg("FlexOlmoForCausalLM", **overrides)


@pytest.fixture
def adapter() -> FlexOlmoArchitectureAdapter:
    return FlexOlmoArchitectureAdapter(_make_cfg())


class TestFlexOlmoMapping:
    def test_moe_replaces_dense_mlp(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        # Router is a raw-parameter module, not nn.Linear — plain delegation.
        assert type(mlp.submodules["gate"]) is GeneralizedComponent
        assert mlp.submodules["gate"].name == "gate"

    def test_inherits_olmo2_post_norm_layout(self, adapter):
        assert isinstance(adapter, Olmo2ArchitectureAdapter)
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "post_attention_layernorm"
        assert blocks.submodules["ln2"].name == "post_feedforward_layernorm"
        attn = blocks.submodules["attn"]
        assert attn.submodules["q_norm"].name == "q_norm"


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["FlexOlmoForCausalLM"] is FlexOlmoArchitectureAdapter
