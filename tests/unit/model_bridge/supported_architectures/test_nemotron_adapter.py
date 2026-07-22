"""Unit tests for the NemotronArchitectureAdapter (dense Nemotron/Minitron).

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
    GatedMLPBridge,
    MLPBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.nemotron import (
    NemotronArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    return make_bridge_cfg(
        "NemotronForCausalLM",
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        n_key_value_heads=2,
        default_prepend_bos=True,
    )


@pytest.fixture(scope="class")
def adapter() -> NemotronArchitectureAdapter:
    return NemotronArchitectureAdapter(_make_cfg())


class TestNemotronAdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is False
        assert adapter.cfg.gated_mlp is False
        assert adapter.cfg.n_key_value_heads == 2

    def test_layernorm1p_disables_folding(self, adapter):
        """LayerNorm1P applies gamma as (weight + 1); folding the stored weight
        without the offset would silently corrupt compat-mode numerics."""
        assert adapter.supports_fold_ln is False
        assert adapter.supports_center_writing_weights is False


class TestNemotronComponentMapping:
    def test_mlp_is_plain_not_gated(self, adapter):
        """Squared-ReLU MLP: up_proj/down_proj only, no gate projection."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert set(mlp.submodules) == {"in", "out"}
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_norms_are_layernorm_bridges(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["ln1"], NormalizationBridge)
        assert submodules["ln1"].name == "input_layernorm"
        assert submodules["ln2"].name == "post_attention_layernorm"
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_attention_paths(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["o"].name == "o_proj"


class TestNemotronRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["NemotronForCausalLM"] is NemotronArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="nemotron", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "NemotronForCausalLM"
