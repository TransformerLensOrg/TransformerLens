"""Unit tests for LlamaArchitectureAdapter.

Tests cover:
- Config attribute validation
- Component mapping structure
- Weight conversion keys
- GQA support
- Rotary embedding setup
- Factory registration
"""

from types import SimpleNamespace
from typing import Any

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 8,
    d_model: int = 128,
    n_layers: int = 2,
    d_vocab: int = 1000,
    n_key_value_heads: int | None = None,
) -> TransformerBridgeConfig:
    """Return minimal config for Llama adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=512,
        n_heads=n_heads,
        d_vocab=d_vocab,
        default_prepend_bos=True,
        architecture="LlamaForCausalLM",
    )

    if n_key_value_heads is not None:
        cfg.n_key_value_heads = n_key_value_heads

    return cfg


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> LlamaArchitectureAdapter:
    return LlamaArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestLlamaAdapterConfig:
    """Tests config values set by the adapter."""

    def test_normalization_type_is_rms(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type_is_rotary(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_true(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp_is_true(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_is_false(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm_is_true(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_eps_attr_is_variance_epsilon(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        assert adapter.cfg.eps_attr == "variance_epsilon"


# ---------------------------------------------------------------------------
# GQA tests
# ---------------------------------------------------------------------------


class TestLlamaGQA:
    """Tests grouped query attention support."""

    def test_n_key_value_heads_added_to_default_config(self) -> None:
        cfg = _make_cfg(n_key_value_heads=4)
        adapter = LlamaArchitectureAdapter(cfg)

        assert "n_key_value_heads" in adapter.default_config
        assert adapter.default_config["n_key_value_heads"] == 4

    def test_n_key_value_heads_set_on_cfg(self) -> None:
        cfg = _make_cfg(n_key_value_heads=4)
        adapter = LlamaArchitectureAdapter(cfg)

        assert adapter.cfg.n_key_value_heads == 4


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestLlamaComponentMapping:
    """Tests component mapping structure."""

    def _blocks(self, adapter: LlamaArchitectureAdapter) -> BlockBridge:
        component_mapping = adapter.component_mapping
        assert component_mapping is not None
        blocks = component_mapping["blocks"]
        assert isinstance(blocks, BlockBridge)
        return blocks

    def test_embed_is_embedding_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        component_mapping = adapter.component_mapping
        assert component_mapping is not None

        assert isinstance(
            component_mapping["embed"],
            EmbeddingBridge,
        )

    def test_embed_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        component_mapping = adapter.component_mapping
        assert component_mapping is not None

        assert (
            component_mapping["embed"].name
            == "model.embed_tokens"
        )

    def test_rotary_emb_is_rotary_embedding_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        component_mapping = adapter.component_mapping
        assert component_mapping is not None

        assert isinstance(
            component_mapping["rotary_emb"],
            RotaryEmbeddingBridge,
        )

    def test_rotary_emb_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        component_mapping = adapter.component_mapping
        assert component_mapping is not None

        assert (
            component_mapping["rotary_emb"].name
            == "model.rotary_emb"
        )

    def test_blocks_is_block_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)
        assert isinstance(blocks, BlockBridge)

    def test_blocks_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        assert blocks.name == "model.layers"

    def test_ln1_is_rms_norm_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        assert isinstance(
            blocks.submodules["ln1"],
            RMSNormalizationBridge,
        )

    def test_ln2_is_rms_norm_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        assert isinstance(
            blocks.submodules["ln2"],
            RMSNormalizationBridge,
        )

    def test_attn_is_position_embeddings_attention_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        assert isinstance(
            blocks.submodules["attn"],
            PositionEmbeddingsAttentionBridge,
        )

    def test_mlp_is_gated_mlp_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        assert isinstance(
            blocks.submodules["mlp"],
            GatedMLPBridge,
        )

    def test_q_proj_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        attn = blocks.submodules["attn"]

        assert attn.submodules["q"].name == "q_proj"

    def test_k_proj_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        attn = blocks.submodules["attn"]

        assert attn.submodules["k"].name == "k_proj"

    def test_v_proj_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        attn = blocks.submodules["attn"]

        assert attn.submodules["v"].name == "v_proj"

    def test_o_proj_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        attn = blocks.submodules["attn"]

        assert attn.submodules["o"].name == "o_proj"

    def test_gate_proj_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        mlp = blocks.submodules["mlp"]

        assert mlp.submodules["gate"].name == "gate_proj"

    def test_up_proj_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        mlp = blocks.submodules["mlp"]

        assert mlp.submodules["in"].name == "up_proj"

    def test_down_proj_name(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        blocks = self._blocks(adapter)

        mlp = blocks.submodules["mlp"]

        assert mlp.submodules["out"].name == "down_proj"

    def test_ln_final_is_rms_norm_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        component_mapping = adapter.component_mapping
        assert component_mapping is not None

        assert isinstance(
            component_mapping["ln_final"],
            RMSNormalizationBridge,
        )

    def test_unembed_is_unembedding_bridge(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        component_mapping = adapter.component_mapping
        assert component_mapping is not None

        assert isinstance(
            component_mapping["unembed"],
            UnembeddingBridge,
        )
# ---------------------------------------------------------------------------
# Weight conversion tests
# ---------------------------------------------------------------------------

class TestLlamaWeightConversions:
    """Tests expected conversion keys exist."""

    def _conversions(
        self,
        adapter: LlamaArchitectureAdapter,
    ) -> dict:
        conversions = adapter.weight_processing_conversions
        assert conversions is not None
        return conversions

    def test_q_weight_key_present(
        self,
        adapter: LlamaArchitectureAdapter,
    ) -> None:
        conversions = self._conversions(adapter)

        assert (
            "blocks.{i}.attn.q.weight"
            in conversions
        )

    def test_k_weight_key_present(
        self,
        adapter: LlamaArchitectureAdapter,
    ) -> None:
        conversions = self._conversions(adapter)

        assert (
            "blocks.{i}.attn.k.weight"
            in conversions
        )

    def test_v_weight_key_present(
        self,
        adapter: LlamaArchitectureAdapter,
    ) -> None:
        conversions = self._conversions(adapter)

        assert (
            "blocks.{i}.attn.v.weight"
            in conversions
        )

    def test_o_weight_key_present(
        self,
        adapter: LlamaArchitectureAdapter,
    ) -> None:
        conversions = self._conversions(adapter)

        assert (
            "blocks.{i}.attn.o.weight"
            in conversions
        )


# ---------------------------------------------------------------------------
# setup_component_testing tests
# ---------------------------------------------------------------------------


class DummyAttention:
    def __init__(self) -> None:
        self.rotary_emb = None

    def set_rotary_emb(self, rotary_emb: Any) -> None:
        self.rotary_emb = rotary_emb


class DummyBlock:
    def __init__(self) -> None:
        self.attn = DummyAttention()


class DummyBridgeModel:
    def __init__(self, n_layers: int = 2) -> None:
        self.blocks = [DummyBlock() for _ in range(n_layers)]


class TestLlamaSetupComponentTesting:
    """Tests rotary embedding setup."""

    def test_rotary_emb_set_on_bridge_model_blocks(
        self, adapter: LlamaArchitectureAdapter
    ) -> None:
        rotary_emb = object()

        hf_model = SimpleNamespace(
            model=SimpleNamespace(
                rotary_emb=rotary_emb
            )
        )

        bridge_model = DummyBridgeModel()

        adapter.setup_component_testing(
            hf_model,
            bridge_model,
        )

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_template_attention_bridge_accepts_rotary_embedding(
    self, adapter: LlamaArchitectureAdapter
) -> None:
        """Ensure setup_component_testing successfully injects RoPE into template attention bridge."""

        rotary_emb = object()

        hf_model = SimpleNamespace(
            model=SimpleNamespace(
                rotary_emb=rotary_emb
            )
        )

        # Should run without raising
        adapter.setup_component_testing(hf_model)

        attn_bridge = adapter.get_generalized_component(
            "blocks.0.attn"
        )

        # Verify method exists and adapter remains usable
        assert hasattr(attn_bridge, "set_rotary_emb")


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestLlamaFactoryRegistration:
    """Tests factory registration."""

    def test_factory_returns_llama_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(
            cfg
        )

        assert isinstance(
            adapter,
            LlamaArchitectureAdapter,
        )

    def test_factory_key_exists(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "LlamaForCausalLM" in SUPPORTED_ARCHITECTURES