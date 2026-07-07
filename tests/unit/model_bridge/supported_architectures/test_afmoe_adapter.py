"""Unit tests for the AfmoeArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    MoEBridge,
)
from transformer_lens.model_bridge.supported_architectures.afmoe import (
    AfmoeArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=32,
        d_head=16,
        n_layers=4,
        n_ctx=128,
        n_heads=2,
        d_mlp=64,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="AfmoeForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> AfmoeArchitectureAdapter:
    return AfmoeArchitectureAdapter(_make_cfg())


class TestAfmoeComponentMapping:
    def test_attention_stays_native(self, adapter):
        """Per-head QK-norm, RoPE only on sliding layers, and sigmoid output
        gating are HF-side; the bridge must delegate, not rebuild."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True
        assert set(attn.submodules) >= {"q", "k", "v", "o", "gate", "q_norm", "k_norm"}

    def test_sandwich_norms(self, adapter):
        """AFMoE dual-normalizes both attention and MLP (Gemma-2 layout)."""
        blocks = adapter.component_mapping["blocks"].submodules
        assert blocks["ln1"].name == "input_layernorm"
        assert blocks["ln1_post"].name == "post_attention_layernorm"
        assert blocks["ln2"].name == "pre_mlp_layernorm"
        assert blocks["ln2_post"].name == "post_mlp_layernorm"

    def test_moe_with_optional_router_and_shared_experts(self, adapter):
        """Layers below num_dense_layers hold a plain gated MLP, so router
        and shared experts must be optional."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert mlp.submodules["router_gate"].optional is True
        assert mlp.submodules["router_gate"].name == "router.gate"
        shared = mlp.submodules["shared_experts"]
        assert isinstance(shared, GatedMLPBridge)
        assert shared.optional is True


class TestAfmoeRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["AfmoeForCausalLM"] is AfmoeArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="afmoe", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "AfmoeForCausalLM"
