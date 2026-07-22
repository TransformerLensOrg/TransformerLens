"""Unit tests for DeepSeekV2ArchitectureAdapter.

Tests cover:
- Config flags set by the adapter
- Component mapping structure (bridge types and HF module names)
- ``optional`` flags encoding the V2-full vs V2-Lite attention split
- Weight conversion key set

Behavioural coverage (forward pass, hook firing) lives in
``tests/integration/model_bridge/test_deepseek_v2_adapter.py``.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MLAAttentionBridge,
    MLABlockBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.deepseek_v2 import (
    DeepSeekV2ArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N_HEADS = 8
N_KV_HEADS = 2
D_MODEL = 64
D_MLP = 256
N_LAYERS = 4
N_CTX = 128
D_VOCAB = 1000


def _make_cfg() -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for DeepSeek V2 adapter tests."""
    return TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        n_heads=N_HEADS,
        d_vocab=D_VOCAB,
        d_mlp=D_MLP,
        n_key_value_heads=N_KV_HEADS,
        architecture="DeepseekV2ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> DeepSeekV2ArchitectureAdapter:
    return DeepSeekV2ArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestDeepSeekV2AdapterConfig:
    """Tests that the adapter sets the correct config flags."""

    def test_normalization_type(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        """DeepSeek V2 uses partial RoPE, reported as rotary."""
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_final_rms(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_uses_rms_norm(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestDeepSeekV2AdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    def test_top_level_keys(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_top_level_bridge_types(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], MLABlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_block_bridge_types(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], MLAAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MoEBridge)

    def test_block_hf_paths(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["attn"].name == "self_attn"
        assert blocks.submodules["mlp"].name == "mlp"


# ---------------------------------------------------------------------------
# MLA attention tests
# ---------------------------------------------------------------------------


class TestDeepSeekV2AdapterMLAAttention:
    """Tests the Multi-Head Latent Attention submodule mapping."""

    def test_attention_submodule_keys(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
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

    def test_attention_hf_paths(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q_a_proj"].name == "q_a_proj"
        assert attn.submodules["q_a_layernorm"].name == "q_a_layernorm"
        assert attn.submodules["q_b_proj"].name == "q_b_proj"
        assert attn.submodules["q_proj"].name == "q_proj"
        assert attn.submodules["kv_a_proj_with_mqa"].name == "kv_a_proj_with_mqa"
        assert attn.submodules["kv_a_layernorm"].name == "kv_a_layernorm"
        assert attn.submodules["kv_b_proj"].name == "kv_b_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attention_bridge_types(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["q_a_proj"], LinearBridge)
        assert isinstance(attn.submodules["q_b_proj"], LinearBridge)
        assert isinstance(attn.submodules["q_proj"], LinearBridge)
        assert isinstance(attn.submodules["kv_a_proj_with_mqa"], LinearBridge)
        assert isinstance(attn.submodules["kv_b_proj"], LinearBridge)
        assert isinstance(attn.submodules["o"], LinearBridge)
        assert isinstance(attn.submodules["kv_a_layernorm"], RMSNormalizationBridge)

    def test_q_a_layernorm_is_plain_component(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        """q_a_layernorm is called directly by MLAAttentionBridge, so it needs no bridge type."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        q_a_layernorm = attn.submodules["q_a_layernorm"]
        assert isinstance(q_a_layernorm, GeneralizedComponent)
        assert not isinstance(q_a_layernorm, RMSNormalizationBridge)

    def test_q_path_submodules_are_optional(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        """Q projections differ between V2-full and V2-Lite, so all are optional."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for key in ("q_a_proj", "q_a_layernorm", "q_b_proj", "q_proj"):
            assert attn.submodules[key].optional is True, f"{key} should be optional"

    def test_kv_path_submodules_are_required(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        """The KV compression path is present across all V2 variants."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for key in ("kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj", "o"):
            assert attn.submodules[key].optional is False, f"{key} should be required"


# ---------------------------------------------------------------------------
# MoE tests
# ---------------------------------------------------------------------------


class TestDeepSeekV2AdapterMoE:
    """Tests the MoE submodule mapping and its dense-layer fallback."""

    def test_moe_submodule_keys(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        """The routing gate is deliberately not bridged — DeepseekV2Moe.forward bypasses it."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"shared_experts"}

    def test_shared_experts_is_optional(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        """Dense layers (idx < first_k_dense_replace) have no shared_experts."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["shared_experts"].optional is True

    def test_shared_experts_bridge_type(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["shared_experts"], GatedMLPBridge)

    def test_shared_experts_submodule_keys(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        shared_experts = (
            adapter.component_mapping["blocks"].submodules["mlp"].submodules["shared_experts"]
        )
        assert set(shared_experts.submodules.keys()) == {"gate", "in", "out"}

    def test_shared_experts_hf_paths(self, adapter: DeepSeekV2ArchitectureAdapter) -> None:
        shared_experts = (
            adapter.component_mapping["blocks"].submodules["mlp"].submodules["shared_experts"]
        )
        assert shared_experts.submodules["gate"].name == "gate_proj"
        assert shared_experts.submodules["in"].name == "up_proj"
        assert shared_experts.submodules["out"].name == "down_proj"

    def test_shared_experts_linear_bridge_types(
        self, adapter: DeepSeekV2ArchitectureAdapter
    ) -> None:
        shared_experts = (
            adapter.component_mapping["blocks"].submodules["mlp"].submodules["shared_experts"]
        )
        for submodule in shared_experts.submodules.values():
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# Weight conversion key tests
# ---------------------------------------------------------------------------


class TestDeepSeekV2AdapterWeightConversions:
    """DeepSeek V2 keeps raw HF weights — MLA projections are not remapped."""

    def test_weight_processing_conversions_are_empty(
        self, adapter: DeepSeekV2ArchitectureAdapter
    ) -> None:
        assert adapter.weight_processing_conversions == {}
