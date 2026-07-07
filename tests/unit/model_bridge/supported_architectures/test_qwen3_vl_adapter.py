"""Unit tests for the Qwen3VLArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.qwen3_vl import (
    Qwen3VLArchitectureAdapter,
    _DeepStackMergerBridge,
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
        architecture="Qwen3VLForConditionalGeneration",
    )


@pytest.fixture(scope="class")
def adapter() -> Qwen3VLArchitectureAdapter:
    return Qwen3VLArchitectureAdapter(_make_cfg())


class TestQwen3VLComponentMapping:
    def test_deepstack_mergers_wrapped_per_level(self, adapter):
        """DeepStack's novelty: per-level mergers must be individually
        hookable (the injection add itself is not a module call)."""
        tower = adapter.component_mapping["vision_encoder"]
        mergers = tower.submodules["deepstack_mergers"]
        assert isinstance(mergers, _DeepStackMergerBridge)
        assert mergers.is_list_item is True
        assert mergers.name == "deepstack_merger_list"

    def test_text_attention_stays_native(self, adapter):
        """Interleaved mRoPE + per-head QK-norm live in HF's forward."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True
        assert attn.submodules["q_norm"].name == "q_norm"

    def test_qwen_conventions(self, adapter):
        assert adapter.cfg.default_prepend_bos is False
        assert adapter.cfg.is_multimodal is True
        assert adapter.component_mapping["embed"].name == "model.language_model.embed_tokens"
        assert adapter.component_mapping["unembed"].name == "lm_head"


class TestQwen3VLRegistration:
    def test_factory_lookup(self):
        assert (
            SUPPORTED_ARCHITECTURES["Qwen3VLForConditionalGeneration"] is Qwen3VLArchitectureAdapter
        )

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="qwen3_vl", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Qwen3VLForConditionalGeneration"

    def test_multimodal_loader_class(self):
        from transformer_lens.utilities.architectures import MULTIMODAL_ARCHITECTURES

        assert "Qwen3VLForConditionalGeneration" in MULTIMODAL_ARCHITECTURES
