"""Unit tests for HunYuanDenseV1ArchitectureAdapter.

Tests cover:
- Config attribute validation
- Component mapping structure
- Weight conversion keys and rearrange patterns
- Architecture guards
- Setup component tests
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.hunyuan_v1_dense import (
    HunYuanDenseV1ArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 32,
    n_key_value_heads: int = 4,
    d_model: int = 128,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for HunYuanDenseV1 adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_key_value_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        architecture="HunYuanDenseV1ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> HunYuanDenseV1ArchitectureAdapter:
    return HunYuanDenseV1ArchitectureAdapter(cfg)


# For rotary embedding and attention implementation check in setup component testing


def _fake_attn(layer_idx: int) -> SimpleNamespace:
    """Per-layer self_attn with a mutable .config so the eager flip is observable."""
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        layer_idx=layer_idx,
    )


def _fake_hf_model(rotary_emb: object, n_layers: int = 2) -> SimpleNamespace:
    """Stub hf_model exposing everything setup_component_testing walks:
    - .model.rotary_emb                                        -> rotary wiring
    - .config._attn_implementation                             -> top-level eager flip
    - .model.layers[*].self_attn.config._attn_implementation   -> per-layer eager flip
    """
    return SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        model=SimpleNamespace(
            rotary_emb=rotary_emb,
            layers=[SimpleNamespace(self_attn=_fake_attn(i)) for i in range(n_layers)],
        ),
    )


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
# Config attribute tests
# ---------------------------------------------------------------------------


class TestHunYuanDenseV1AdapterConfig:
    """Adapter must set all required config flags to the values HunYuanDenseV1 expects."""

    def test_attn_implementation_is_eager(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        """Set to eager."""
        assert adapter.cfg.attn_implementation == "eager"


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestHunYuanDenseV1AdapterComponentMapping:
    """Component mapping must have the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_is_rotary_embedding_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_rotary_emb_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_is_block_bridge(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        """Sequential attn and MLP requires BlockBridge, not ParallelBlockBridge."""
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_is_rms_normalization_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_ln_final_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_is_unembedding_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_ln1_is_rms_normalization_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)

    def test_blocks_ln1_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_blocks_ln2_is_rms_normalization_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)

    def test_blocks_ln2_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"

    def test_attn_is_position_embeddings_attention_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)

    def test_attn_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_requires_attention_mask_is_true(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_attention_mask is True

    def test_attn_requires_position_embeddings_is_true(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_position_embeddings is True

    def test_mlp_is_gated_mlp_bridge(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)

    def test_mlp_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "mlp"

    # -- Attention submodules --

    @pytest.mark.parametrize("slot", ["q", "k", "v", "o"])
    def test_attn_submodule_is_linear_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter, slot: str
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules[slot], LinearBridge)

    @pytest.mark.parametrize("slot", ["q_norm", "k_norm"])
    def test_attn_submodule_is_rms_normalization_bridge(
        self, adapter: HunYuanDenseV1ArchitectureAdapter, slot: str
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules[slot], RMSNormalizationBridge)

    @pytest.mark.parametrize(
        "slot, hf_name",
        [
            ("q", "q_proj"),
            ("k", "k_proj"),
            ("v", "v_proj"),
            ("o", "o_proj"),
            ("q_norm", "query_layernorm"),
            ("k_norm", "key_layernorm"),
        ],
    )
    def test_attn_submodule_name(
        self, adapter: HunYuanDenseV1ArchitectureAdapter, slot: str, hf_name: str
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules[slot].name == hf_name

    # -- MLP submodules --

    def test_mlp_gate_is_linear_bridge(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["gate"], LinearBridge)

    def test_mlp_gate_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["gate"].name == "gate_proj"

    def test_mlp_in_is_linear_bridge(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["in"], LinearBridge)

    def test_mlp_in_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "up_proj"

    def test_mlp_out_is_linear_bridge(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["out"], LinearBridge)

    def test_mlp_out_name(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestHunYuanDenseV1AdapterWeightConversions:
    """Adapter must define exactly the four QKVO weight conversions."""

    def test_conversion_keys_present(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        """HunYuanDenseV1 has 4 weight matrices (QKVO) per layer"""
        assert adapter.weight_processing_conversions.keys() == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    @pytest.mark.parametrize("slot", ["q", "k", "v"])
    def test_qkv_weight_uses_split_heads_pattern(
        self, adapter: HunYuanDenseV1ArchitectureAdapter, slot: str
    ) -> None:
        conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
        expected = adapter.cfg.n_key_value_heads if slot in ["k", "v"] else adapter.cfg.n_heads
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == expected

    def test_o_uses_merge_heads_pattern(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


class TestHunYuanDenseV1ArchitectureGuards:
    """Guard against accidental introduction of features HunYuanDenseV1 does not have."""

    def test_no_pos_embed_component(self, adapter: HunYuanDenseV1ArchitectureAdapter) -> None:
        """HunYuanDenseV1 uses rotary embeddings, so there is no learned positional embedding."""
        assert "pos_embed" not in adapter.component_mapping


# ---------------------------------------------------------------------------
# Setup component testing tests
# ---------------------------------------------------------------------------


class TestHunYuanDenseV1SetupComponentTesting:
    """setup_component_testing must wire HunYuanDenseV1's shared rotary embedding into attention bridges."""

    def test_setup_flips_top_level_attn_implementation_to_eager(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        """HF reference defaults to sdpa; setup must flip the top-level config to eager."""
        hf = _fake_hf_model(object())
        assert hf.config._attn_implementation == "sdpa"

        adapter.setup_component_testing(hf)

        assert hf.config._attn_implementation == "eager"

    def test_setup_flips_per_layer_attn_implementation_to_eager(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        """Each already-built attn layer caches its own config; setup must flip all of them."""
        hf = _fake_hf_model(object(), n_layers=3)
        assert all(l.self_attn.config._attn_implementation == "sdpa" for l in hf.model.layers)

        adapter.setup_component_testing(hf)

        for layer in hf.model.layers:
            assert layer.self_attn.config._attn_implementation == "eager"

    def test_sets_rotary_emb_on_template_attention(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(
        self, adapter: HunYuanDenseV1ArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb
