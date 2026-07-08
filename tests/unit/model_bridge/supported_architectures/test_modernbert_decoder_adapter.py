"""Unit tests for ModernBertDecoderArchitectureAdapter.

Ettin decoders: sliding/global attention mix delegates, layer-0 Identity
attn-norm disables folding, fused-GLU Wi/Wo MLP, embedding norm, and a
BERT-style prediction head before the untied decoder projection.
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
    NormalizationBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.modernbert_decoder import (
    ModernBertDecoderArchitectureAdapter,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    return make_bridge_cfg("ModernBertDecoderForCausalLM", **overrides)


@pytest.fixture
def adapter() -> ModernBertDecoderArchitectureAdapter:
    return ModernBertDecoderArchitectureAdapter(_make_cfg())


class TestModernBertDecoderMapping:
    def test_identity_layer0_norm_disables_folding(self, adapter):
        assert adapter.supports_fold_ln is False

    def test_embedding_norm_mapped(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embeddings.tok_embeddings"
        assert isinstance(mapping["embed_ln"], NormalizationBridge)
        assert mapping["embed_ln"].name == "model.embeddings.norm"

    def test_attention_delegated_for_sliding_mix(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert type(attn) is AttentionBridge
        assert attn.maintain_native_attention is True
        assert attn.submodules["o"].name == "Wo"

    def test_fused_glu_mlp(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.submodules["in"].name == "Wi"
        assert mlp.submodules["out"].name == "Wo"

    def test_prediction_head_then_decoder_unembed(self, adapter):
        mapping = adapter.component_mapping
        assert type(mapping["prediction_head"]) is GeneralizedComponent
        assert mapping["prediction_head"].name == "lm_head"
        assert mapping["unembed"].name == "decoder"


def test_factory_registration():
    assert (
        SUPPORTED_ARCHITECTURES["ModernBertDecoderForCausalLM"]
        is ModernBertDecoderArchitectureAdapter
    )
