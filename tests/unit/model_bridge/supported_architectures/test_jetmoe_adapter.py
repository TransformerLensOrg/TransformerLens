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
    MoERouterBridge,
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

    def test_moa_exposes_hook_z_and_head_results(self, adapter):
        from transformer_lens.hook_points import HookPoint

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.hook_z, HookPoint)
        assert "hook_z" not in attn.hook_aliases  # no o_proj to alias it to
        assert attn.supports_attn_result is True
        assert callable(attn.head_results_from_cache)

    def test_slot_routing_inverts_hf_expert_sort(self):
        """HF's gating sorts (token, slot) pairs by expert; slot k must get top-k entry k."""
        import torch
        from transformers.models.jetmoe.modeling_jetmoe import JetMoeTopKGating

        torch.manual_seed(0)
        gating = JetMoeTopKGating(input_size=8, num_experts=5, top_k=2)
        x = torch.randn(6, 8)
        index_sorted, batch_index, batch_gates, expert_size, logits = gating(x)
        experts_idx, gates = _JetMoeAttentionBridge._slot_routing(
            (index_sorted, batch_index, batch_gates, expert_size), [2, 3], 2, 5
        )
        top_logits, top_idx = logits.topk(2, dim=-1)
        assert torch.equal(experts_idx, top_idx.reshape(2, 3, 2))
        torch.testing.assert_close(gates, torch.softmax(top_logits, -1).reshape(2, 3, 2))

    def test_moe_mlp_router_hookable(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert isinstance(mlp.submodules["gate"], MoERouterBridge)
        assert mlp.submodules["gate"].logits_index == -1
        assert mlp.submodules["gate"].name == "router"

    def test_no_fold_target(self, adapter):
        assert adapter.supports_fold_ln is False
        assert adapter.weight_processing_conversions == {}


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["JetMoeForCausalLM"] is JetMoeArchitectureAdapter
