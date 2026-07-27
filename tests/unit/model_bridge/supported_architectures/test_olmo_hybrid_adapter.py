"""Unit tests for the OlmoHybridArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.olmo_hybrid import (
    OlmoHybridArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "OlmoHybridForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=4,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> OlmoHybridArchitectureAdapter:
    return OlmoHybridArchitectureAdapter(_make_cfg())


class TestOlmoHybridComponentMapping:
    def test_per_layer_type_optionality(self, adapter):
        """Full-attention layers are OLMo2 post-norm (no input_layernorm,
        with post_feedforward_layernorm); linear layers are pre-norm."""
        blocks = adapter.component_mapping["blocks"].submodules
        assert blocks["ln1"].optional is True
        assert blocks["ln1"].name == "input_layernorm"
        assert blocks["ln2_post"].optional is True
        assert blocks["attn"].optional is True
        assert blocks["attn"].maintain_native_attention is True
        assert blocks["linear_attn"].optional is True

    def test_stateful(self, adapter):
        assert adapter.cfg.is_stateful is True
        assert adapter.supports_fold_ln is False


class TestOlmoHybridRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["OlmoHybridForCausalLM"] is OlmoHybridArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="olmo_hybrid", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "OlmoHybridForCausalLM"


class TestOlmoHybridResidMidDropped:
    """No single hook_resid_mid target is correct for both layer types, so the
    block drops the alias type-visibly (ParallelBlockBridge precedent)."""

    def test_hook_resid_mid_alias_absent(self, adapter) -> None:
        block = adapter.component_mapping["blocks"]
        assert "hook_resid_mid" not in block.hook_aliases
