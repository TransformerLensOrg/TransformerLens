"""Unit tests for the Lfm2MoeArchitectureAdapter — no model downloads."""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.lfm2_moe import (
    Lfm2MoeArchitectureAdapter,
    Lfm2MoeBlockBridge,
)


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    bridge_cfg = TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=4,
        n_ctx=128,
        n_heads=4,
        n_key_value_heads=2,
        d_vocab=256,
        d_mlp=224,
        architecture="Lfm2MoeForCausalLM",
    )
    bridge_cfg.layer_types = ["conv", "conv", "full_attention", "conv"]
    bridge_cfg.moe_intermediate_size = 56
    bridge_cfg.num_experts = 8
    bridge_cfg.experts_per_token = 2
    bridge_cfg.norm_eps = 1e-5
    bridge_cfg.rope_parameters = {"rope_theta": 5_000_000, "rope_type": "default"}
    return bridge_cfg


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> Lfm2MoeArchitectureAdapter:
    return Lfm2MoeArchitectureAdapter(cfg)


class TestLfm2MoeAdapterConfig:
    def test_hybrid_config_is_propagated(self, adapter: Lfm2MoeArchitectureAdapter) -> None:
        assert adapter.cfg.layer_types == ["conv", "conv", "full_attention", "conv"]
        assert adapter.cfg.moe_intermediate_size == 56
        assert adapter.cfg.num_experts == 8
        assert adapter.cfg.experts_per_token == 2

    def test_norm_and_rope_config(self, adapter: Lfm2MoeArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.eps == 1e-5
        assert adapter.cfg.rotary_base == 5_000_000

    def test_default_prepend_bos_is_false(self, adapter: Lfm2MoeArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False


class TestLfm2MoeComponentMapping:
    def test_has_residual_only_top_level_mapping(self, adapter: Lfm2MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping is not None
        assert set(mapping) == {"embed", "blocks", "ln_final", "unembed"}

    def test_component_types(self, adapter: Lfm2MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["blocks"], Lfm2MoeBlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_hf_module_paths(self, adapter: Lfm2MoeArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.embedding_norm"
        assert mapping["unembed"].name == "lm_head"

    def test_blocks_only_advertise_supported_residual_aliases(
        self, adapter: Lfm2MoeArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.hook_aliases == {
            "hook_resid_pre": "hook_in",
            "hook_resid_post": "hook_out",
        }
        assert blocks.submodules == {}
