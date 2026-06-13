"""Unit tests for NeoxArchitectureAdapter.

Tests cover:
- Component mapping structure (bridge types and HF module names)
- Weight conversion key set and shared source keys
- setup_component_testing rotary embedding wiring
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.neox import (
    NeoxArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for NeoX adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=False,
        architecture="GPTNeoXForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> NeoxArchitectureAdapter:
    return NeoxArchitectureAdapter(cfg)


def _fake_hf_model(rotary_emb: object) -> SimpleNamespace:
    return SimpleNamespace(gpt_neox=SimpleNamespace(rotary_emb=rotary_emb))


class DummyAttention:
    def __init__(self) -> None:
        self.rotary_emb = None

    def set_rotary_emb(self, rotary_emb: object) -> None:
        self.rotary_emb = rotary_emb


class DummyBlock:
    def __init__(self, has_attention: bool = True) -> None:
        if has_attention:
            self.attn = DummyAttention()


class DummyBridgeModel:
    def __init__(self, blocks: list[DummyBlock]) -> None:
        self.blocks = blocks


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestNeoxAdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    def test_top_level_keys(self, adapter: NeoxArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_bridge_types(self, adapter: NeoxArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        blocks = mapping["blocks"]
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(blocks, ParallelBlockBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: NeoxArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "gpt_neox.embed_in"
        assert mapping["rotary_emb"].name == "gpt_neox.rotary_emb"
        assert mapping["blocks"].name == "gpt_neox.layers"
        assert mapping["ln_final"].name == "gpt_neox.final_layer_norm"
        assert mapping["unembed"].name == "embed_out"

    def test_no_pos_embed_key(self, adapter: NeoxArchitectureAdapter) -> None:
        """NeoX uses rotary embeddings — no learned positional embedding component."""
        assert "pos_embed" not in adapter.component_mapping

    def test_block_submodule_keys(self, adapter: NeoxArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_attention_submodule_keys(self, adapter: NeoxArchitectureAdapter) -> None:
        """NeoX uses a combined QKV projection alongside derived q/k/v split bridges."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"qkv", "q", "k", "v", "o"}

    def test_mlp_submodule_keys(self, adapter: NeoxArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"in", "out"}

    def test_block_bridge_types(self, adapter: NeoxArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)
        assert isinstance(blocks.submodules["attn"], JointQKVPositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_attention_hf_paths(self, adapter: NeoxArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.name == "attention"
        assert attn.submodules["qkv"].name == "query_key_value"
        assert attn.submodules["o"].name == "dense"

    def test_attn_requires_attention_mask(self, adapter: NeoxArchitectureAdapter) -> None:
        """GPTNeoX requires an explicit attention mask."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is True

    def test_block_hf_paths(self, adapter: NeoxArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["mlp"].name == "mlp"
        assert blocks.submodules["mlp"].submodules["in"].name == "dense_h_to_4h"
        assert blocks.submodules["mlp"].submodules["out"].name == "dense_4h_to_h"

    def test_linear_submodule_bridge_types(self, adapter: NeoxArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        for submodule in [*attn.submodules.values(), *mlp.submodules.values()]:
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestNeoxAdapterWeightConversions:
    """Tests that weight_processing_conversions has the expected key set and source keys.

    NeoX stores Q, K, V (weights and biases) in a single interleaved matrix
    gpt_neox.layers.{i}.attention.query_key_value.{weight,bias}.
    All three projections share the same source key — each conversion extracts
    its slice via SplitTensorConversion.
    """

    def test_exact_conversion_key_set(self, adapter: NeoxArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q",
            "blocks.{i}.attn.k",
            "blocks.{i}.attn.v",
            "blocks.{i}.attn.b_Q",
            "blocks.{i}.attn.b_K",
            "blocks.{i}.attn.b_V",
            "blocks.{i}.attn.o",
        }

    def test_qkv_weights_share_source_key(self, adapter: NeoxArchitectureAdapter) -> None:
        """Q, K, V weights all come from the same interleaved QKV matrix."""
        expected = "gpt_neox.layers.{i}.attention.query_key_value.weight"
        for key in ("blocks.{i}.attn.q", "blocks.{i}.attn.k", "blocks.{i}.attn.v"):
            assert adapter.weight_processing_conversions[key].source_key == expected

    def test_qkv_biases_share_source_key(self, adapter: NeoxArchitectureAdapter) -> None:
        """Q, K, V biases all come from the same interleaved QKV bias vector."""
        expected = "gpt_neox.layers.{i}.attention.query_key_value.bias"
        for key in ("blocks.{i}.attn.b_Q", "blocks.{i}.attn.b_K", "blocks.{i}.attn.b_V"):
            assert adapter.weight_processing_conversions[key].source_key == expected

    def test_o_projection_source_key(self, adapter: NeoxArchitectureAdapter) -> None:
        expected = "gpt_neox.layers.{i}.attention.dense.weight"
        assert adapter.weight_processing_conversions["blocks.{i}.attn.o"].source_key == expected


# ---------------------------------------------------------------------------
# setup_component_testing — rotary embedding wiring
# ---------------------------------------------------------------------------


class TestNeoxSetupComponentTesting:
    """setup_component_testing must wire NeoX's rotary embedding into attention bridges."""

    def test_sets_rotary_emb_on_bridge_model_blocks(self, adapter: NeoxArchitectureAdapter) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(self, adapter: NeoxArchitectureAdapter) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb

    def test_no_bridge_model_does_not_raise(self, adapter: NeoxArchitectureAdapter) -> None:
        """setup_component_testing without a bridge_model should not raise."""
        adapter.setup_component_testing(_fake_hf_model(object()))
