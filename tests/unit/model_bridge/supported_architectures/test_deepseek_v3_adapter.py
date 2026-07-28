"""Unit tests for DeepSeekV3ArchitectureAdapter.

Tests cover:
- Config flags set by the adapter
- Component mapping structure (bridge types and HF module names)
- The MLA attention submodule mapping (V3 always compresses Q, so unlike V2
  every Q/KV projection is required and there is no direct ``q_proj`` fallback)
- The MoE mapping, including the bridged (optional) router ``gate``
- Weight conversion key set

Behavioural coverage (forward pass, hook firing) lives in
``tests/integration/model_bridge/test_deepseek_adapter.py``.
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
from transformer_lens.model_bridge.supported_architectures.deepseek_v3 import (
    DeepSeekV3ArchitectureAdapter,
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
    """Return a minimal TransformerBridgeConfig for DeepSeek V3 adapter tests."""
    return TransformerBridgeConfig(
        d_model=D_MODEL,
        d_head=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
        n_ctx=N_CTX,
        n_heads=N_HEADS,
        d_vocab=D_VOCAB,
        d_mlp=D_MLP,
        n_key_value_heads=N_KV_HEADS,
        architecture="DeepseekV3ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> DeepSeekV3ArchitectureAdapter:
    return DeepSeekV3ArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestDeepSeekV3AdapterConfig:
    """Tests that the adapter sets the correct config flags."""

    def test_normalization_type(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        """DeepSeek V3 uses partial RoPE, reported as rotary."""
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_final_rms(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_uses_rms_norm(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestDeepSeekV3AdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    def test_top_level_keys(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_top_level_bridge_types(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], MLABlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_block_bridge_types(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], MLAAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MoEBridge)

    def test_block_hf_paths(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["attn"].name == "self_attn"
        assert blocks.submodules["mlp"].name == "mlp"


# ---------------------------------------------------------------------------
# MLA attention tests
# ---------------------------------------------------------------------------


class TestDeepSeekV3AdapterMLAAttention:
    """Tests the Multi-Head Latent Attention submodule mapping.

    Unlike V2, V3 always compresses Q via the two-stage LoRA path, so there is
    no direct ``q_proj`` fallback and every projection is required.
    """

    def test_attention_submodule_keys(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {
            "q_a_proj",
            "q_a_layernorm",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "kv_b_proj",
            "o",
        }

    def test_attention_hf_paths(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q_a_proj"].name == "q_a_proj"
        assert attn.submodules["q_a_layernorm"].name == "q_a_layernorm"
        assert attn.submodules["q_b_proj"].name == "q_b_proj"
        assert attn.submodules["kv_a_proj_with_mqa"].name == "kv_a_proj_with_mqa"
        assert attn.submodules["kv_a_layernorm"].name == "kv_a_layernorm"
        assert attn.submodules["kv_b_proj"].name == "kv_b_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_attention_bridge_types(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules["q_a_proj"], LinearBridge)
        assert isinstance(attn.submodules["q_b_proj"], LinearBridge)
        assert isinstance(attn.submodules["kv_a_proj_with_mqa"], LinearBridge)
        assert isinstance(attn.submodules["kv_b_proj"], LinearBridge)
        assert isinstance(attn.submodules["o"], LinearBridge)
        assert isinstance(attn.submodules["q_a_layernorm"], RMSNormalizationBridge)
        assert isinstance(attn.submodules["kv_a_layernorm"], RMSNormalizationBridge)

    def test_no_direct_q_proj(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        """V3 has no V2-Lite-style direct Q projection."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert "q_proj" not in attn.submodules

    def test_all_projections_required(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        """Every attention submodule is present in all V3 layers, so none are optional."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        for key, submodule in attn.submodules.items():
            assert submodule.optional is False, f"{key} should be required"


# ---------------------------------------------------------------------------
# MoE tests
# ---------------------------------------------------------------------------


class TestDeepSeekV3AdapterMoE:
    """Tests the MoE submodule mapping and its dense-layer fallback."""

    def test_moe_submodule_keys(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        """V3 bridges the router gate (a custom Module), unlike V2."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "shared_experts"}

    def test_gate_is_optional_plain_component(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        """The router gate is a custom Module (not nn.Linear) and absent on dense layers."""
        gate = adapter.component_mapping["blocks"].submodules["mlp"].submodules["gate"]
        assert isinstance(gate, GeneralizedComponent)
        assert not isinstance(gate, LinearBridge)
        assert gate.optional is True
        assert gate.name == "gate"

    def test_shared_experts_is_optional(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        """Dense layers (idx < first_k_dense_replace) have no shared_experts."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["shared_experts"].optional is True

    def test_shared_experts_bridge_type(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["shared_experts"], GatedMLPBridge)

    def test_shared_experts_submodule_keys(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        shared_experts = (
            adapter.component_mapping["blocks"].submodules["mlp"].submodules["shared_experts"]
        )
        assert set(shared_experts.submodules.keys()) == {"gate", "in", "out"}

    def test_shared_experts_hf_paths(self, adapter: DeepSeekV3ArchitectureAdapter) -> None:
        shared_experts = (
            adapter.component_mapping["blocks"].submodules["mlp"].submodules["shared_experts"]
        )
        assert shared_experts.submodules["gate"].name == "gate_proj"
        assert shared_experts.submodules["in"].name == "up_proj"
        assert shared_experts.submodules["out"].name == "down_proj"

    def test_shared_experts_linear_bridge_types(
        self, adapter: DeepSeekV3ArchitectureAdapter
    ) -> None:
        shared_experts = (
            adapter.component_mapping["blocks"].submodules["mlp"].submodules["shared_experts"]
        )
        for submodule in shared_experts.submodules.values():
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# Weight conversion key tests
# ---------------------------------------------------------------------------


class TestDeepSeekV3AdapterWeightConversions:
    """DeepSeek V3 keeps raw HF weights — MLA projections are not remapped."""

    def test_weight_processing_conversions_are_empty(
        self, adapter: DeepSeekV3ArchitectureAdapter
    ) -> None:
        assert adapter.weight_processing_conversions == {}
