"""Unit tests for LLaDA2MoeArchitectureAdapter.

Masked block-diffusion MoE: fused-QKV attention delegates (bidirectional),
generation phases are excluded, dense-first MoE submodules are optional,
and the Dream rope shim is registered at load.
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
from transformer_lens.model_bridge.supported_architectures.llada2_moe import (
    LLaDA2MoeArchitectureAdapter,
    _LLaDA2FusedAttentionBridge,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    overrides.setdefault("n_key_value_heads", 4)
    return make_bridge_cfg("LLaDA2MoeModelLM", **overrides)


@pytest.fixture
def adapter() -> LLaDA2MoeArchitectureAdapter:
    return LLaDA2MoeArchitectureAdapter(_make_cfg())


class TestLLaDA2MoePhases:
    def test_diffusion_treatment(self, adapter):
        assert adapter.applicable_phases == [1, 2, 3, 4]
        assert adapter.supports_generation is False
        assert adapter.supports_fold_ln is False


class TestLLaDA2MoeMapping:
    def test_fused_qkv_attention_delegated(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert type(attn) is _LLaDA2FusedAttentionBridge
        assert isinstance(attn, AttentionBridge)
        assert attn.maintain_native_attention is True
        assert attn.submodules["qkv"].name == "query_key_value"
        assert attn.submodules["o"].name == "dense"
        assert attn.submodules["q_norm"].name == "query_layernorm"

    def test_moe_with_optional_dense_first_submodules(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert type(mlp.submodules["gate"]) is GeneralizedComponent
        assert mlp.submodules["gate"].optional is True
        assert mlp.submodules["shared_experts"].optional is True

    def test_word_embeddings_path(self, adapter):
        assert adapter.component_mapping["embed"].name == "model.word_embeddings"


def test_prepare_loading_registers_rope_shim(adapter=None):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ROPE_INIT_FUNCTIONS.pop("default", None)
    LLaDA2MoeArchitectureAdapter(_make_cfg()).prepare_loading("inclusionAI/LLaDA2.0-mini", {})
    assert "default" in ROPE_INIT_FUNCTIONS


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["LLaDA2MoeModelLM"] is LLaDA2MoeArchitectureAdapter
