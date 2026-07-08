"""Unit tests for the Starcoder2ArchitectureAdapter.

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
    MLPBridge,
    NormalizationBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.starcoder2 import (
    Starcoder2ArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Starcoder2ForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> Starcoder2ArchitectureAdapter:
    return Starcoder2ArchitectureAdapter(_make_cfg())


class TestStarcoder2AdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is False
        assert adapter.cfg.gated_mlp is False
        assert adapter.cfg.n_key_value_heads == 2

    def test_qkv_bias_conversions_use_kv_head_count(self, adapter):
        """StarCoder2 biases every q/k/v; compat-mode weight access needs the
        K/V bias reshapes split by the kv-head count (GQA), not n_heads."""
        conv = adapter.weight_processing_conversions
        for key in ("blocks.{i}.attn.k.bias", "blocks.{i}.attn.v.bias"):
            assert key in conv, f"missing {key}"
            assert conv[key].tensor_conversion.axes_lengths["h"] == adapter.cfg.n_key_value_heads


class TestStarcoder2ComponentMapping:
    def test_norms_are_plain_layernorm(self, adapter):
        """StarCoder2 uses nn.LayerNorm despite its llama-like shape."""
        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["ln1"], NormalizationBridge)
        assert not isinstance(submodules["ln1"], RMSNormalizationBridge)
        assert submodules["ln1"].name == "input_layernorm"
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_mlp_is_plain_c_fc_c_proj(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert mlp.submodules["in"].name == "c_fc"
        assert mlp.submodules["out"].name == "c_proj"


class TestStarcoder2Registration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Starcoder2ForCausalLM"] is Starcoder2ArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="starcoder2", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Starcoder2ForCausalLM"
