"""Unit tests for MistralArchitectureAdapter.

Tests cover:
- Config attribute validation
- Component mapping structure
- GQA support
- Weight conversion keys
- Attention behavior flags
- Factory registration
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cfg(
    n_heads: int = 8,
    d_model: int = 128,
    n_layers: int = 2,
    d_vocab: int = 1000,
    n_key_value_heads: int = 4,
) -> TransformerBridgeConfig:
    """Create minimal config for Mistral adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        n_layers=n_layers,
        n_ctx=512,
        d_vocab=d_vocab,
        default_prepend_bos=True,
        architecture="MistralForCausalLM",
    )
    cfg.n_key_value_heads = n_key_value_heads
    return cfg


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> MistralArchitectureAdapter:
    return MistralArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestMistralConfig:
    """Validate adapter config settings."""

    def test_rms_norm(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_rotary(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_false(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_true(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_false(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm_true(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------

class TestMistralComponents:

    def test_embed(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None

        assert isinstance(
            adapter.component_mapping["embed"],
            EmbeddingBridge,
        )
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None

        assert isinstance(
            adapter.component_mapping["rotary_emb"],
            RotaryEmbeddingBridge,
        )

    def test_blocks(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_ln1_ln2(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None

        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)

    def test_attn(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None

        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_flags(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True
        assert attn.requires_position_embeddings is True

    def test_mlp(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)

    def test_proj_names(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_unembed(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["unembed"].name == "lm_head"


# ---------------------------------------------------------------------------
# GQA tests
# ---------------------------------------------------------------------------

class TestMistralGQA:

    def test_n_key_value_heads(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 4
        assert adapter.default_config["n_key_value_heads"] == 4


# ---------------------------------------------------------------------------
# Weight conversion tests
# ---------------------------------------------------------------------------

class TestMistralWeights:

    def test_qkv_keys_exist(self, adapter: MistralArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None

        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------

class TestMistralFactory:

    def test_registry_key_exists(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "MistralForCausalLM" in SUPPORTED_ARCHITECTURES