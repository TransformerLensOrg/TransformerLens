"""Unit tests for the Qwen3VLMoeArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import MoEBridge
from transformer_lens.model_bridge.supported_architectures.qwen3_vl_moe import (
    Qwen3VLMoeArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "Qwen3VLMoeForConditionalGeneration",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> Qwen3VLMoeArchitectureAdapter:
    return Qwen3VLMoeArchitectureAdapter(_make_cfg())


class TestQwen3VLMoeComponentMapping:
    def test_moe_mlp(self, adapter):
        """Router logits hook via MoERouterBridge; gate/experts optional since
        dense mlp_only_layers share the name."""
        from transformer_lens.model_bridge.generalized_components import MoERouterBridge

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert set(mlp.submodules) == {"gate", "experts"}
        assert isinstance(mlp.submodules["gate"], MoERouterBridge)
        assert mlp.submodules["gate"].optional is True
        assert mlp.submodules["experts"].optional is True

    def test_inherits_qwen3_vl_layout(self, adapter):
        tower = adapter.component_mapping["vision_encoder"]
        assert tower.name == "model.visual"
        assert "deepstack_mergers" in tower.submodules
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True


class TestQwen3VLMoeRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Qwen3VLMoeForConditionalGeneration"]
            is Qwen3VLMoeArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen3_vl_moe", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen3VLMoeForConditionalGeneration"
