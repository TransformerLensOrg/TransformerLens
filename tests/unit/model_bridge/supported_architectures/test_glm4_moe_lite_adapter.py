"""Unit tests for the Glm4MoeLiteArchitectureAdapter (GLM-4.7-Flash).

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
    LinearBridge,
    MLAAttentionBridge,
    MLABlockBridge,
    MoEBridge,
    RMSNormalizationBridge,
)
from transformer_lens.model_bridge.supported_architectures.glm4_moe_lite import (
    Glm4MoeLiteArchitectureAdapter,
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
        architecture="Glm4MoeLiteForCausalLM",
    )


@pytest.fixture(scope="class")
def adapter() -> Glm4MoeLiteArchitectureAdapter:
    return Glm4MoeLiteArchitectureAdapter(_make_cfg())


class TestGlm4MoeLiteAdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.uses_rms_norm is True

    def test_no_bos_prepending(self, adapter):
        """GLM-4.7-Flash's tokenizer has no BOS token."""
        assert adapter.cfg.default_prepend_bos is False

    def test_mla_weights_keep_hf_layout(self, adapter):
        """MLA projections are not per-head rearrangeable — no conversions."""
        assert adapter.weight_processing_conversions == {}


class TestGlm4MoeLiteComponentMapping:
    def test_blocks_use_mla_block_bridge(self, adapter):
        assert isinstance(adapter.component_mapping["blocks"], MLABlockBridge)

    def test_attention_is_mla_with_lora_projections(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, MLAAttentionBridge)
        assert set(attn.submodules.keys()) == {
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "q_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
            "o",
        }
        # Q-compression path is optional (q_lora_rank may be None on variants).
        assert attn.submodules["q_a_proj"].optional is True
        assert attn.submodules["q_proj"].optional is True
        assert attn.submodules["kv_a_proj_with_mqa"].optional is False

    def test_moe_with_optional_router_and_shared_expert(self, adapter):
        """Dense layers in mlp_layer_types have neither router nor shared expert."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert set(mlp.submodules.keys()) == {"gate", "shared_experts"}
        assert mlp.submodules["gate"].optional is True
        shared = mlp.submodules["shared_experts"]
        assert isinstance(shared, GatedMLPBridge)
        assert shared.optional is True
        assert shared.submodules["gate"].name == "gate_proj"
        assert shared.submodules["in"].name == "up_proj"
        assert shared.submodules["out"].name == "down_proj"

    def test_kv_a_layernorm_is_rms(self, adapter):
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["kv_a_layernorm"], RMSNormalizationBridge)

    def test_top_level_paths(self, adapter):
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"
        o = mapping["blocks"].submodules["attn"].submodules["o"]
        assert isinstance(o, LinearBridge)
        assert o.name == "o_proj"


class TestGlm4MoeLiteRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["Glm4MoeLiteForCausalLM"] is Glm4MoeLiteArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="glm4_moe_lite", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "Glm4MoeLiteForCausalLM"
