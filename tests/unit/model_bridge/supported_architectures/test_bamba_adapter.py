"""Unit tests for the BambaArchitectureAdapter.

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
    GatedRMSNormBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    SSM2MixerBridge,
)
from transformer_lens.model_bridge.supported_architectures.bamba import (
    BambaArchitectureAdapter,
)


def _make_cfg(**overrides) -> TransformerBridgeConfig:
    defaults = dict(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=128,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="BambaForCausalLM",
    )
    defaults.update(overrides)
    cfg = TransformerBridgeConfig(**defaults)
    cfg.layers_block_type = ["mamba", "attention"]
    return cfg


@pytest.fixture(scope="class")
def adapter() -> BambaArchitectureAdapter:
    return BambaArchitectureAdapter(_make_cfg())


class TestBambaAdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.is_stateful is True

    def test_layers_block_type_normalized(self, adapter):
        # Legacy input normalized to the canonical vocabulary.
        assert adapter.cfg.layers_block_type == ["linear_attention", "full_attention"]

    def test_hybrid_keeps_raw_weights(self, adapter):
        assert adapter.supports_fold_ln is False
        assert adapter.weight_processing_conversions == {}


class TestBambaComponentMapping:
    def test_top_level_paths(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        # Bamba names its final norm final_layernorm, not norm.
        assert mapping["ln_final"].name == "model.final_layernorm"
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)

    def test_both_mixers_optional(self, adapter):
        """Each layer holds exactly one of self_attn / mamba per layers_block_type."""
        submodules = adapter.component_mapping["blocks"].submodules
        attn = submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "self_attn"
        assert attn.optional is True
        mixer = submodules["mixer"]
        assert isinstance(mixer, SSM2MixerBridge)
        assert mixer.name == "mamba"
        assert mixer.optional is True
        assert isinstance(mixer.submodules["inner_norm"], GatedRMSNormBridge)

    def test_block_norm_and_mlp_paths(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert submodules["ln1"].name == "input_layernorm"
        assert submodules["ln2"].name == "pre_ff_layernorm"
        mlp = submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "feed_forward"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"


class TestBambaRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["BambaForCausalLM"] is BambaArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="bamba", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "BambaForCausalLM"
