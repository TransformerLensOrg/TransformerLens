"""Unit tests for the Gemma4TextArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.gemma4_text import (
    Gemma4TextArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "Gemma4ForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> Gemma4TextArchitectureAdapter:
    return Gemma4TextArchitectureAdapter(_make_cfg())


class TestGemma4TextComponentMapping:
    def test_text_stack_at_model_root(self, adapter):
        """No language_model nesting and no vision components."""
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"
        assert "vision_encoder" not in mapping
        assert "vision_projector" not in mapping
        assert adapter.cfg.is_multimodal is False

    def test_inherits_gemma4_phases(self, adapter):
        """PLE / layer_scalar / MoE topology is not fold-safe."""
        assert adapter.applicable_phases == [1, 2, 4]


class TestGemma4TextRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Gemma4ForCausalLM"] is Gemma4TextArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="gemma4_text", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Gemma4ForCausalLM"
