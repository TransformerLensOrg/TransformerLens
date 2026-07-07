"""Unit tests for the Glm4ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import JointGateUpMLPBridge
from transformer_lens.model_bridge.supported_architectures.glm4 import (
    Glm4ArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Glm4ForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> Glm4ArchitectureAdapter:
    return Glm4ArchitectureAdapter(_make_cfg())


class TestGlm4ComponentMapping:
    def test_sandwich_norms(self, adapter):
        """GLM-4-0414 keeps GLM naming: post_attention_layernorm is the
        pre-MLP norm; the sandwich norms sit on the sublayer outputs."""
        blocks = adapter.component_mapping["blocks"].submodules
        assert blocks["ln1"].name == "input_layernorm"
        assert blocks["ln1_post"].name == "post_self_attn_layernorm"
        assert blocks["ln2"].name == "post_attention_layernorm"
        assert blocks["ln2_post"].name == "post_mlp_layernorm"

    def test_inherits_glm_conventions(self, adapter):
        assert adapter.cfg.rotary_adjacent_pairs is True
        assert adapter.supports_fold_ln is False
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, JointGateUpMLPBridge)


class TestGlm4Registration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Glm4ForCausalLM"] is Glm4ArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="glm4", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Glm4ForCausalLM"
