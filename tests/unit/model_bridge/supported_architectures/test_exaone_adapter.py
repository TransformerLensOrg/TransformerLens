"""Unit tests for the ExaoneArchitectureAdapter (EXAONE-3.x, remote code).

Download-free: synthetic configs and structural assertions only. The remote
modeling code cannot be instantiated in-memory without trust_remote_code, so
end-to-end parity lives in the CI-gated integration test.
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.exaone import (
    ExaoneArchitectureAdapter,
)


def _make_cfg(**overrides) -> TransformerBridgeConfig:
    defaults = dict(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_mlp=256,
        d_vocab=512,
        n_key_value_heads=2,
        architecture="ExaoneForCausalLM",
    )
    defaults.update(overrides)
    return TransformerBridgeConfig(**defaults)


@pytest.fixture(scope="class")
def adapter() -> ExaoneArchitectureAdapter:
    return ExaoneArchitectureAdapter(_make_cfg())


class TestExaoneAdapterConfig:
    def test_config_flags(self, adapter):
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.final_rms is True
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.uses_rms_norm is True

    def test_no_bos_prepending(self, adapter):
        """Verified against LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct's tokenizer."""
        assert adapter.cfg.default_prepend_bos is False

    def test_qkvo_conversions_registered(self, adapter):
        keys = set(adapter.weight_processing_conversions)
        assert {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        } <= keys


class TestExaoneComponentMapping:
    """EXAONE hides llama-style internals under GPT-2-flavored names."""

    def test_gpt2_flavored_top_level_paths(self, adapter):
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "transformer.wte"
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert mapping["rotary_emb"].name == "transformer.rotary"
        assert isinstance(mapping["blocks"], BlockBridge)
        assert mapping["blocks"].name == "transformer.h"
        assert mapping["ln_final"].name == "transformer.ln_f"
        assert mapping["unembed"].name == "lm_head"

    def test_attention_is_double_nested(self, adapter):
        """ExaoneAttentionBlock wraps ExaoneAttention — path is attn.attention."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.name == "attn.attention"
        for name, path in (
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "out_proj"),
        ):
            sub = attn.submodules[name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == path

    def test_norms_use_gpt2_names(self, adapter):
        submodules = adapter.component_mapping["blocks"].submodules
        assert isinstance(submodules["ln1"], RMSNormalizationBridge)
        assert submodules["ln1"].name == "ln_1"
        assert submodules["ln2"].name == "ln_2"

    def test_gated_mlp_projection_order(self, adapter):
        """c_fc_0 is the activated gate; c_fc_1 is the up projection."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.submodules["gate"].name == "c_fc_0"
        assert mlp.submodules["in"].name == "c_fc_1"
        assert mlp.submodules["out"].name == "c_proj"


class TestExaoneRegistration:
    def test_factory_lookup(self):
        assert SUPPORTED_ARCHITECTURES["ExaoneForCausalLM"] is ExaoneArchitectureAdapter

    def test_model_type_detection(self):
        from transformer_lens.model_bridge.sources.transformers import (
            determine_architecture_from_hf_config,
        )

        cfg = SimpleNamespace(model_type="exaone", architectures=[])
        assert determine_architecture_from_hf_config(cfg) == "ExaoneForCausalLM"

    def test_verify_models_grants_remote_code_to_lgai(self):
        from transformer_lens.tools.model_registry.verify_models import (
            _BRIDGE_REMOTE_CODE_PREFIXES,
        )

        assert "LGAI-EXAONE/" in _BRIDGE_REMOTE_CODE_PREFIXES
