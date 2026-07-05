"""Unit tests for the PhiMoEArchitectureAdapter - no model downloads."""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    NormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.phimoe import (
    PhiMoEArchitectureAdapter,
)


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    bridge_cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=2,
        d_vocab=256,
        d_mlp=32,
        architecture="PhiMoEForCausalLM",
    )
    bridge_cfg.num_experts = 8
    bridge_cfg.experts_per_token = 2
    bridge_cfg.attention_bias = True
    bridge_cfg.lm_head_bias = True
    bridge_cfg.router_jitter_noise = 0.01
    bridge_cfg.input_jitter_noise = 0.01
    bridge_cfg.rope_parameters = {"rope_theta": 10_000.0, "rope_type": "default"}
    bridge_cfg.eos_token_id = 32000
    return bridge_cfg


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> PhiMoEArchitectureAdapter:
    return PhiMoEArchitectureAdapter(cfg)


class TestPhiMoEAdapterConfig:
    def test_config_flags(self, adapter: PhiMoEArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.attn_implementation == "eager"
        assert adapter.cfg.default_prepend_bos is False
        assert adapter.cfg.rotary_base == 10_000.0
        assert adapter.cfg.eos_token_id == [32000, 32007]


class TestPhiMoEWeightConversions:
    def test_conversion_keys_include_attention_biases(
        self, adapter: PhiMoEArchitectureAdapter
    ) -> None:
        assert set(adapter.weight_processing_conversions) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
        }

    def test_kv_conversions_use_n_key_value_heads(self, adapter: PhiMoEArchitectureAdapter) -> None:
        for key in ("blocks.{i}.attn.k.weight", "blocks.{i}.attn.v.weight"):
            conv = adapter.weight_processing_conversions[key]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.axes_lengths["n"] == 2


class TestPhiMoEComponentMapping:
    def test_component_types(self, adapter: PhiMoEArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_hf_paths(self, adapter: PhiMoEArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

        subs = mapping["blocks"].submodules
        assert subs["ln1"].name == "input_layernorm"
        assert subs["ln2"].name == "post_attention_layernorm"
        assert subs["attn"].name == "self_attn"
        assert subs["mlp"].name == "mlp"

    def test_block_submodule_types(self, adapter: PhiMoEArchitectureAdapter) -> None:
        subs = adapter.component_mapping["blocks"].submodules
        assert isinstance(subs["ln1"], NormalizationBridge)
        assert isinstance(subs["ln2"], NormalizationBridge)
        assert isinstance(subs["attn"], AttentionBridge)
        assert isinstance(subs["mlp"], MoEBridge)
        assert isinstance(subs["mlp"].submodules["gate"], LinearBridge)
        assert subs["mlp"].submodules["gate"].name == "router"
