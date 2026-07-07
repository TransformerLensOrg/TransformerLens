"""Unit tests for the Exaone4ArchitectureAdapter.

Download-free: synthetic configs and structural assertions only.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.exaone import (
    ExaoneArchitectureAdapter,
)
from transformer_lens.model_bridge.supported_architectures.exaone4 import (
    Exaone4ArchitectureAdapter,
)


def _make_cfg() -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="Exaone4ForCausalLM",
    )
    cfg.layer_types = ["sliding_attention", "full_attention"]
    return cfg


@pytest.fixture(scope="class")
def adapter() -> Exaone4ArchitectureAdapter:
    return Exaone4ArchitectureAdapter(_make_cfg())


class TestExaone4AdapterConfig:
    def test_distinct_from_exaone3_family(self, adapter):
        """EXAONE 4.0 is native transformers, not the remote-code 3.x family."""
        assert not isinstance(adapter, ExaoneArchitectureAdapter)

    def test_post_norm_disables_folding(self, adapter):
        """Norms are applied after each sublayer inside the residual branch."""
        assert adapter.supports_fold_ln is False
        assert adapter.supports_center_writing_weights is False

    def test_layer_types_surfaced(self, adapter):
        assert adapter.cfg.layer_types == ["sliding_attention", "full_attention"]

    def test_no_bos_prepending(self, adapter):
        assert adapter.cfg.default_prepend_bos is False


class TestExaone4ComponentMapping:
    def test_post_norm_positions(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln1"].name == "post_attention_layernorm"
        assert submodules["ln2"].name == "post_feedforward_layernorm"

    def test_attention_has_per_head_qk_norms(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert isinstance(attn.submodules["q_norm"], RMSNormalizationBridge)
        assert attn.submodules["q_norm"].name == "q_norm"
        assert attn.submodules["k_norm"].name == "k_norm"


class TestExaone4Registration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Exaone4ForCausalLM"] is Exaone4ArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="exaone4", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Exaone4ForCausalLM"
