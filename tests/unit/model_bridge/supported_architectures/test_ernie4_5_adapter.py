"""Unit tests for the Ernie4_5ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.ernie4_5 import (
    Ernie4_5ArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
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
        architecture="Ernie4_5ForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> Ernie4_5ArchitectureAdapter:
    return Ernie4_5ArchitectureAdapter(_make_cfg())


class TestErnie4_5Adapter:
    def test_inherits_llama_layout(self, adapter):
        assert isinstance(adapter, LlamaArchitectureAdapter)
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"

    def test_no_bos_prepending(self, adapter):
        """Verified against baidu/ERNIE-4.5-0.3B-PT's tokenizer."""
        assert adapter.cfg.default_prepend_bos is False

    def test_interleaved_rope_flag(self, adapter):
        """ERNIE rotates adjacent pairs (GLM convention); the standard llama
        rotation diverged by ~13 logits before this flag existed."""
        assert adapter.cfg.rotary_adjacent_pairs is True


class TestErnie4_5Registration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Ernie4_5ForCausalLM"] is Ernie4_5ArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="ernie4_5", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Ernie4_5ForCausalLM"
