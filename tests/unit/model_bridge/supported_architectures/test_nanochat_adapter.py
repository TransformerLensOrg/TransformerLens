"""Unit tests for NanoChatArchitectureAdapter.

NanoChat's three deliberate simplifications drive the mapping: weightless
RMSNorms (nothing to fold), ungated relu^2 MLP, and tanh logit soft-cap.
q/k norms run after rope, so attention stays delegated.
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
    BlockBridge,
    MLPBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.nanochat import (
    NanoChatArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    soft_cap = overrides.pop("final_logit_softcapping", 15.0)
    cfg = make_bridge_cfg("NanoChatForCausalLM", **overrides)
    # Not a TransformerBridgeConfig field; arrives via HF-config passthrough.
    cfg.final_logit_softcapping = soft_cap
    return cfg


@pytest.fixture
def adapter() -> NanoChatArchitectureAdapter:
    return NanoChatArchitectureAdapter(_make_cfg())


class TestNanoChatConfig:
    def test_weightless_norms_disable_fold_ln(self, adapter):
        assert adapter.supports_fold_ln is False

    def test_ungated_mlp_flag(self, adapter):
        assert adapter.cfg.gated_mlp is False

    def test_logit_softcap_propagates(self, adapter):
        assert adapter.cfg.output_logits_soft_cap == 15.0


class TestNanoChatComponentMapping:
    def test_attention_delegated_with_post_rope_norms(self, adapter):
        """q/k norms apply AFTER rope — the reverse of the bridge's
        reimplementation order, so attention must delegate."""
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, BlockBridge)
        attn = blocks.submodules["attn"]
        assert type(attn) is AttentionBridge
        assert attn.maintain_native_attention is True
        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert isinstance(attn.submodules["k_norm"], RMSNormalizationBridge)

    def test_relu2_mlp_shape(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.submodules["in"].name == "fc1"
        assert mlp.submodules["out"].name == "fc2"

    def test_top_level_names(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["NanoChatForCausalLM"] is NanoChatArchitectureAdapter
