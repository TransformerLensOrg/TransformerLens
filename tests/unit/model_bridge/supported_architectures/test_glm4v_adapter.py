"""Unit tests for the Glm4vArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import JointGateUpMLPBridge
from transformer_lens.model_bridge.supported_architectures.glm4v import (
    Glm4vArchitectureAdapter,
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
        architecture="Glm4vForConditionalGeneration",
    )


@pytest.fixture(scope="class")
def adapter() -> Glm4vArchitectureAdapter:
    return Glm4vArchitectureAdapter(_make_cfg())


class TestGlm4vComponentMapping:
    def test_glm4_sandwich_layout(self, adapter):
        """GLM-4-0414 text layout nested under model.language_model."""
        blocks = adapter.component_mapping["blocks"].submodules
        assert blocks["ln1"].name == "input_layernorm"
        assert blocks["ln1_post"].name == "post_self_attn_layernorm"
        assert blocks["ln2"].name == "post_attention_layernorm"
        assert blocks["ln2_post"].name == "post_mlp_layernorm"
        assert isinstance(blocks["mlp"], JointGateUpMLPBridge)
        assert adapter.supports_fold_ln is False

    def test_text_attention_stays_native(self, adapter):
        """mRoPE lives in HF's forward."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True

    def test_vision_components(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["vision_encoder"].name == "model.visual"
        assert mapping["vision_projector"].name == "model.visual.merger"
        assert adapter.cfg.is_multimodal is True
        assert adapter.cfg.default_prepend_bos is False


class TestGlm4vRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Glm4vForConditionalGeneration"] is Glm4vArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="glm4v", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Glm4vForConditionalGeneration"
