"""Unit tests for the SeedOssArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.seed_oss import (
    SeedOssArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "SeedOssForCausalLM",
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
def adapter() -> SeedOssArchitectureAdapter:
    return SeedOssArchitectureAdapter(_make_cfg())


class TestSeedOssAdapter:
    def test_inherits_llama_layout(self, adapter):
        """Seed-OSS is a Llama-layout decoder; only tokenizer policy differs."""
        assert isinstance(adapter, LlamaArchitectureAdapter)
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        attn = mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"

    def test_no_bos_prepending(self, adapter):
        """Verified against ByteDance-Seed/Seed-OSS-36B-Instruct's tokenizer."""
        assert adapter.cfg.default_prepend_bos is False


class TestSeedOssRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["SeedOssForCausalLM"] is SeedOssArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="seed_oss", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "SeedOssForCausalLM"
