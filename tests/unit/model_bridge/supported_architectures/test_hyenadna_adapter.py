"""Unit tests for HyenaDNAArchitectureAdapter.

Attention-free Hyena long-conv blocks: the mixer delegates wholesale,
generation is unavailable in the HF port, and the block wrapper must not
tuple-normalize outputs (the backbone's minimal layer(hidden_states) call
looks like a standalone block call).
"""
from typing import Any

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import MLPBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.hyenadna import (
    HyenaDNAArchitectureAdapter,
    _HyenaBlockBridge,
)


def _make_cfg(**overrides: Any) -> TransformerBridgeConfig:
    overrides.setdefault("d_vocab", 12)
    return make_bridge_cfg("HyenaDNAForCausalLM", **overrides)


@pytest.fixture
def adapter() -> HyenaDNAArchitectureAdapter:
    return HyenaDNAArchitectureAdapter(_make_cfg())


class TestHyenaDNAPhases:
    def test_no_generation(self, adapter):
        """The HF port ships no generate() (not a GenerationMixin)."""
        assert adapter.applicable_phases == [1, 2, 3]
        assert adapter.supports_generation is False


class TestHyenaDNAMapping:
    def test_paths_under_hyena_backbone(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "hyena.backbone.embeddings.word_embeddings"
        assert mapping["ln_final"].name == "hyena.backbone.ln_f"
        assert mapping["unembed"].name == "lm_head"
        assert mapping["blocks"].name == "hyena.backbone.layers"

    def test_blocks_never_tuple_normalize(self, adapter):
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks, _HyenaBlockBridge)
        assert blocks._is_standalone_hidden_state_call((object(),), {}) is False

    def test_mixer_delegated_with_hookable_projections(self, adapter):
        mixer = adapter.component_mapping["blocks"].submodules["mixer"]
        assert type(mixer) is GeneralizedComponent
        assert mixer.submodules["in_proj"].name == "in_proj"
        assert mixer.submodules["out_proj"].name == "out_proj"

    def test_gelu_mlp(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert mlp.submodules["in"].name == "fc1"
        assert mlp.submodules["out"].name == "fc2"


def test_factory_registration():
    assert SUPPORTED_ARCHITECTURES["HyenaDNAForCausalLM"] is HyenaDNAArchitectureAdapter
