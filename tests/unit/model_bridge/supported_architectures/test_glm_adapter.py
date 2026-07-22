"""Unit tests for the GlmArchitectureAdapter (dense GLM-4 / glm-edge).

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    JointGateUpMLPBridge,
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.glm import (
    GlmArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "GlmForCausalLM",
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
def adapter() -> GlmArchitectureAdapter:
    return GlmArchitectureAdapter(_make_cfg())


class TestGlmAdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.default_prepend_bos is False
        assert adapter.cfg.n_key_value_heads == 2

    def test_interleaved_rope_flag(self, adapter):
        """GLM rotates adjacent element pairs, like ERNIE."""
        assert adapter.cfg.rotary_adjacent_pairs is True

    def test_joint_mlp_disables_folding(self, adapter):
        assert adapter.supports_fold_ln is False

    def test_qkv_bias_conversions_use_kv_head_count(self, adapter):
        """GLM has attention biases; without kv-head-aware bias reshapes,
        compat mode's value-bias fold crashes on GQA (n_key_value_heads !=
        n_heads). K/V bias reshapes must split by the kv-head count."""
        conv = adapter.weight_processing_conversions
        for key in ("blocks.{i}.attn.k.bias", "blocks.{i}.attn.v.bias"):
            assert key in conv, f"missing {key}"
            assert conv[key].tensor_conversion.axes_lengths["h"] == adapter.cfg.n_key_value_heads
        assert (
            conv["blocks.{i}.attn.q.bias"].tensor_conversion.axes_lengths["h"]
            == adapter.cfg.n_heads
        )


class TestGlmComponentMapping:
    def test_joint_gate_up_mlp(self, adapter):
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, JointGateUpMLPBridge)
        assert mlp.submodules["out"].name == "down_proj"

    def test_attention_paths(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["o"].name == "o_proj"


class TestGlmRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["GlmForCausalLM"] is GlmArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="glm", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "GlmForCausalLM"
