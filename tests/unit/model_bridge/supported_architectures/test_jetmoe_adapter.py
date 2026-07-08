"""Unit tests for JetMoeArchitectureAdapter.

Mixture-of-Attention-heads: per-expert Q/O live inside the delegated MoA
(no q/k/v/o Linears to alias — a fused-KV alias set replaces them), both
routers are hookable, and per-expert 3D projections rule out folding.
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
    MoEBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.jetmoe import (
    JetMoeArchitectureAdapter,
    _JetMoeAttentionBridge,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    overrides.setdefault("n_key_value_heads", 4)
    return make_bridge_cfg("JetMoeForCausalLM", **overrides)


@pytest.fixture
def adapter() -> JetMoeArchitectureAdapter:
    return JetMoeArchitectureAdapter(_make_cfg())


class TestJetMoeMapping:
    def test_moa_attention_delegated(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert type(attn) is _JetMoeAttentionBridge
        assert isinstance(attn, AttentionBridge)
        assert attn.maintain_native_attention is True
        assert attn.submodules["kv"].name == "kv_proj"
        assert attn.submodules["experts"].submodules["router"].name == "router"

    def test_fused_kv_alias_set(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert "hook_kv" in attn.hook_aliases
        assert "hook_q" not in attn.hook_aliases

    def test_moe_mlp_router_hookable(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert type(mlp.submodules["gate"]) is GeneralizedComponent
        assert mlp.submodules["gate"].name == "router"

    def test_no_fold_target(self, adapter):
        assert adapter.supports_fold_ln is False
        assert adapter.weight_processing_conversions == {}


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["JetMoeForCausalLM"] is JetMoeArchitectureAdapter
