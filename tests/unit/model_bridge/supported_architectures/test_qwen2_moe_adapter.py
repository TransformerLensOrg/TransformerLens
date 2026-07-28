"""Unit tests for the Qwen2MoeArchitectureAdapter."""

import pytest
import torch
from transformers import Qwen2MoeConfig

from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
)
from transformer_lens.model_bridge.supported_architectures.qwen2_moe import (
    Qwen2MoeArchitectureAdapter,
)


def _tiny_config() -> Qwen2MoeConfig:
    return Qwen2MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


@pytest.fixture
def bridge_cfg():
    return build_bridge_config_from_hf(
        _tiny_config(),
        "Qwen2MoeForCausalLM",
        "qwen2-moe-test",
        torch.float32,
    )


@pytest.fixture
def adapter(bridge_cfg) -> Qwen2MoeArchitectureAdapter:
    return Qwen2MoeArchitectureAdapter(bridge_cfg)


class TestQwen2MoeDetection:
    def test_model_type_detects_qwen2_moe(self) -> None:
        assert determine_architecture_from_hf_config(_tiny_config()) == "Qwen2MoeForCausalLM"

    def test_factory_selects_adapter(self, bridge_cfg) -> None:
        selected = ArchitectureAdapterFactory.select_architecture_adapter(bridge_cfg)
        assert isinstance(selected, Qwen2MoeArchitectureAdapter)


class TestQwen2MoeConfigMapping:
    def test_core_fields_map_from_hf_config(self, bridge_cfg) -> None:
        hf_cfg = _tiny_config()
        assert bridge_cfg.d_vocab == hf_cfg.vocab_size
        assert bridge_cfg.d_model == hf_cfg.hidden_size
        assert bridge_cfg.n_layers == hf_cfg.num_hidden_layers
        assert bridge_cfg.n_heads == hf_cfg.num_attention_heads
        assert bridge_cfg.n_key_value_heads == hf_cfg.num_key_value_heads
        assert bridge_cfg.d_mlp == hf_cfg.intermediate_size

    def test_moe_fields_map_from_hf_config(self, bridge_cfg) -> None:
        hf_cfg = _tiny_config()
        assert bridge_cfg.num_experts == hf_cfg.num_experts
        assert bridge_cfg.experts_per_token == hf_cfg.num_experts_per_tok
        assert getattr(bridge_cfg, "moe_intermediate_size") == hf_cfg.moe_intermediate_size
        assert (
            getattr(bridge_cfg, "shared_expert_intermediate_size")
            == hf_cfg.shared_expert_intermediate_size
        )


class TestQwen2MoeComponentMapping:
    def test_reuses_qwen2_attention_mapping(self, adapter: Qwen2MoeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        attn = blocks.submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_is_moe_bridge(self, adapter: Qwen2MoeArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MoEBridge)
        assert mlp.name == "mlp"

    def test_moe_submodules(self, adapter: Qwen2MoeArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {
            "gate",
            "experts",
            "shared_expert",
            "shared_expert_gate",
        }
        assert isinstance(mlp.submodules["gate"], LinearBridge)
        assert isinstance(mlp.submodules["experts"], MoEBridge)
        assert isinstance(mlp.submodules["shared_expert"], GatedMLPBridge)
        assert isinstance(mlp.submodules["shared_expert_gate"], LinearBridge)

    def test_moe_hf_paths(self, adapter: Qwen2MoeArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate"
        assert mlp.submodules["experts"].name == "experts"
        assert mlp.submodules["shared_expert"].name == "shared_expert"
        assert mlp.submodules["shared_expert_gate"].name == "shared_expert_gate"

    def test_shared_expert_submodules(self, adapter: Qwen2MoeArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        shared_expert = mlp.submodules["shared_expert"]
        assert set(shared_expert.submodules.keys()) == {"gate", "in", "out"}
        assert shared_expert.submodules["gate"].name == "gate_proj"
        assert shared_expert.submodules["in"].name == "up_proj"
        assert shared_expert.submodules["out"].name == "down_proj"


class TestQwen2MoeWeightConversions:
    def test_kv_rearrange_uses_n_key_value_heads(
        self, adapter: Qwen2MoeArchitectureAdapter
    ) -> None:
        conversions = adapter.weight_processing_conversions
        assert conversions is not None
        for slot in ("k", "v"):
            conv = conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            assert conv.tensor_conversion.axes_lengths["n"] == 2
