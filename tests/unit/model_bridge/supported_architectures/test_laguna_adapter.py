"""Unit tests for LagunaArchitectureAdapter.

Heterogeneous per-layer head counts and softplus attention gating force
delegated attention with no uniform weight conversions; MoE follows the
optional gate/shared-experts pattern for mixed dense/sparse layers.
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
from transformer_lens.model_bridge.supported_architectures.laguna import (
    LagunaArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    overrides.setdefault("n_key_value_heads", 2)
    return make_bridge_cfg("LagunaForCausalLM", **overrides)


@pytest.fixture
def adapter() -> LagunaArchitectureAdapter:
    return LagunaArchitectureAdapter(_make_cfg())


class TestLagunaMapping:
    def test_no_uniform_conversions(self, adapter):
        """Per-layer head counts: no single (n h) reshape is correct."""
        assert adapter.supports_fold_ln is False
        assert adapter.weight_processing_conversions == {}

    def test_attention_delegated_with_softplus_gate(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert type(attn) is AttentionBridge
        assert attn.maintain_native_attention is True
        assert attn.submodules["gate"].name == "g_proj"
        assert attn.submodules["q"].name == "q_proj"

    def test_moe_optional_submodules(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert mlp.submodules["gate"].optional is True
        assert mlp.submodules["shared_experts"].optional is True


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["LagunaForCausalLM"] is LagunaArchitectureAdapter


def test_prepare_loading_registers_conversion_mapping(adapter):
    """Laguna ships remote code, so transformers skips its native per-expert->batched
    expert conversion for the custom-code class unless the model type is user-registered.
    Without this the batched experts.gate_up_proj/down_proj stay at random init (30,108
    unexpected + 234 missing checkpoint keys) and the experts are noise. prepare_loading
    must user-register the mapping so the conversion runs under remote code."""
    conv = pytest.importorskip("transformers.conversion_mapping")
    if conv.get_checkpoint_conversion_mapping("laguna") is None:
        pytest.skip("transformers build has no native laguna conversion mapping")

    conv.USER_REGISTERED_MAPPINGS.discard("laguna")  # clean precondition
    assert "laguna" not in conv.USER_REGISTERED_MAPPINGS

    adapter.prepare_loading("poolside/Laguna-XS.2", {})

    assert "laguna" in conv.USER_REGISTERED_MAPPINGS
