"""Unit tests for the Ernie4_5_MoeArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tests.unit.model_bridge.supported_architectures.helpers import (
    DENSE_KEYS,
    make_bridge_cfg,
)
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    MoEBridge,
)
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.model_bridge.supported_architectures.ernie4_5_moe import (
    Ernie4_5_MoeArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "Ernie4_5_MoeForCausalLM",
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
def adapter() -> Ernie4_5_MoeArchitectureAdapter:
    return Ernie4_5_MoeArchitectureAdapter(_make_cfg())


class TestErnie4_5MoeAdapterConfig:
    def test_shares_dense_ernie_conventions(self, adapter):
        assert adapter.cfg.rotary_adjacent_pairs is True
        assert adapter.cfg.default_prepend_bos is False
        assert adapter.cfg.normalization_type == "RMS"


class TestErnie4_5MoeComponentMapping:
    def test_moe_with_optional_shared_experts(self, adapter):
        """Dense-prefix layers hold a plain gated MLP — shared experts and the
        router are absent there, so shared_experts is optional and the custom
        sigmoid router is fully delegated."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        # dense_* are covered by the roster in test_moe_dense_dispatch.py.
        assert set(mlp.submodules) - DENSE_KEYS == {"gate", "shared_experts"}
        assert mlp.submodules["gate"].optional is True
        shared = mlp.submodules["shared_experts"]
        assert isinstance(shared, GatedMLPBridge)
        assert shared.optional is True
        assert shared.submodules["gate"].name == "gate_proj"


class TestErnie4_5MoeRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Ernie4_5_MoeForCausalLM"] is Ernie4_5_MoeArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="ernie4_5_moe", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Ernie4_5_MoeForCausalLM"


class TestErnie4_5MoeCompatibilityMode:
    def test_compatibility_mode_matches_hf(self):
        """Layer 0 is dense and layer 1 sparse (moe_layer_start_index=1); folding
        must recognise the sparse layer as MoE instead of asserting on a dense W_in."""
        hf_config = AutoConfig.for_model(
            "ernie4_5_moe",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            moe_intermediate_size=24,
            moe_num_experts=4,
            moe_k=2,
        )
        hf_config.architectures = ["Ernie4_5_MoeForCausalLM"]
        torch.manual_seed(0)
        hf_model = AutoModelForCausalLM.from_config(hf_config).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            expected = hf_model(tokens).logits.log_softmax(-1)

        bridge = build_bridge_from_module(hf_model, "Ernie4_5_MoeForCausalLM", hf_config=hf_config)
        bridge.enable_compatibility_mode(disable_warnings=True)
        with torch.no_grad():
            actual = bridge(tokens).log_softmax(-1)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
