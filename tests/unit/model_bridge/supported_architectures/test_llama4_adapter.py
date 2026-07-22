"""Unit tests for the Llama4ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.supported_architectures.llama4 import (
    Llama4ArchitectureAdapter,
    _Llama4MoEBridge,
    _Llama4SharedExpertBridge,
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
        architecture="Llama4ForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> Llama4ArchitectureAdapter:
    return Llama4ArchitectureAdapter(_make_cfg())


class TestLlama4ComponentMapping:
    def test_attention_stays_native(self, adapter):
        """Complex-tensor RoPE, NoPE temperature tuning, L2 QK-norm, and
        chunked masks are HF-side; the bridge must delegate, not rebuild."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.maintain_native_attention is True
        assert attn.name == "self_attn"

    def test_moe_with_optional_shared_expert(self, adapter):
        """The router returns a tuple so it stays unwrapped; non-MoE layers
        hold a dense gated MLP under the same feed_forward name."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, _Llama4MoEBridge)
        assert mlp.name == "feed_forward"
        assert set(mlp.submodules) == {"shared_expert", "dense_gate", "dense_in", "dense_out"}
        for key in ("dense_gate", "dense_in", "dense_out"):
            assert mlp.submodules[key].optional is True
        shared = mlp.submodules["shared_expert"]
        assert isinstance(shared, _Llama4SharedExpertBridge)
        assert shared.optional is True


class TestLlama4Registration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Llama4ForCausalLM"] is Llama4ArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="llama4_text", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Llama4ForCausalLM"
