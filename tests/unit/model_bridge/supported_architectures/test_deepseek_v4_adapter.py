"""Unit tests for the DeepSeek V4 architecture adapter."""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    MoEBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v4 import (
    DeepSeekV4ArchitectureAdapter,
    DeepseekV4BlockBridge,
    DeepseekV4CompressorBridge,
    DeepseekV4HyperConnectionBridge,
)


@pytest.fixture
def adapter() -> DeepSeekV4ArchitectureAdapter:
    cfg = TransformerBridgeConfig(
        d_model=32,
        d_head=8,
        n_heads=4,
        n_layers=3,
        n_ctx=32,
        d_vocab=64,
        d_mlp=16,
        n_key_value_heads=1,
        architecture="DeepseekV4ForCausalLM",
    )
    return DeepSeekV4ArchitectureAdapter(cfg)


def test_top_level_mapping(adapter: DeepSeekV4ArchitectureAdapter) -> None:
    assert set(adapter.component_mapping) == {
        "embed",
        "rotary_emb",
        "blocks",
        "hc_head",
        "ln_final",
        "unembed",
    }
    assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)
    assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)
    assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)
    assert adapter.component_mapping["hc_head"].name == "model.hc_head"


def test_block_mapping_preserves_mhc_topology(adapter: DeepSeekV4ArchitectureAdapter) -> None:
    blocks = adapter.component_mapping["blocks"]
    assert isinstance(blocks, DeepseekV4BlockBridge)
    assert blocks.name == "model.layers"
    assert set(blocks.submodules) == {
        "attn_hc",
        "ln1",
        "attn",
        "mlp_hc",
        "ln2",
        "mlp",
    }
    assert isinstance(blocks.submodules["attn_hc"], DeepseekV4HyperConnectionBridge)
    assert isinstance(blocks.submodules["mlp_hc"], DeepseekV4HyperConnectionBridge)
    assert blocks.submodules["mlp_hc"].name == "ffn_hc"
    assert blocks.hook_aliases == {}


def test_attention_mapping_exposes_compression_surfaces(
    adapter: DeepSeekV4ArchitectureAdapter,
) -> None:
    attention = adapter.component_mapping["blocks"].submodules["attn"]
    assert isinstance(attention, GeneralizedComponent)
    assert attention.name == "self_attn"
    assert set(attention.submodules) == {
        "q_a_proj",
        "q_a_norm",
        "q_b_proj",
        "q_b_norm",
        "kv_proj",
        "kv_norm",
        "compressor",
        "o_a_proj",
        "o_b_proj",
    }

    compressor = attention.submodules["compressor"]
    assert isinstance(compressor, DeepseekV4CompressorBridge)
    assert compressor.optional
    assert compressor.submodules["indexer"].optional
    indexer = compressor.submodules["indexer"]
    assert set(indexer.submodules) == {
        "kv_proj",
        "gate_proj",
        "kv_norm",
        "q_b_proj",
        "scorer",
        "rotary_emb",
    }
    assert set(indexer.submodules["scorer"].submodules) == {
        "weights_proj",
    }


def test_moe_mapping_exposes_both_router_types_and_experts(
    adapter: DeepSeekV4ArchitectureAdapter,
) -> None:
    mlp = adapter.component_mapping["blocks"].submodules["mlp"]
    assert isinstance(mlp, MoEBridge)
    assert set(mlp.submodules) == {"gate", "experts", "shared_experts"}
    assert mlp.submodules["gate"].name == "gate"
    assert mlp.submodules["experts"].name == "experts"


def test_config_and_processing_guards(adapter: DeepSeekV4ArchitectureAdapter) -> None:
    assert adapter.cfg.normalization_type == "RMS"
    assert adapter.cfg.uses_rms_norm
    assert adapter.cfg.final_rms
    assert not adapter.cfg.rmsnorm_uses_offset
    assert adapter.cfg.positional_embedding_type == "rotary"
    assert adapter.cfg.gated_mlp
    assert adapter.applicable_phases == [2, 4]
    assert not adapter.supports_fold_ln
    assert not adapter.supports_center_writing_weights
