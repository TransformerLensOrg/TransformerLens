"""Unit tests for PhiArchitectureAdapter.

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
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.phi import (
    PhiArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Phi adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        architecture="PhiForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> PhiArchitectureAdapter:
    return PhiArchitectureAdapter(cfg)


def _fake_hf_model(rotary_emb: object) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(rotary_emb=rotary_emb))


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


class TestPhiAdapterConfig:
    """Adapter must set all required config flags to the values Phi expects."""

    def test_positional_embedding_type_is_rotary(self, adapter: PhiArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_parallel_attn_mlp_is_true(self, adapter: PhiArchitectureAdapter) -> None:
        """Phi runs attention and MLP in parallel within each block."""
        assert adapter.cfg.parallel_attn_mlp is True

    def test_use_fast_is_false(self, adapter: PhiArchitectureAdapter) -> None:
        """Do not use the rust based HF tokenizer. Uses python based version instead"""
        assert adapter.cfg.use_fast is False

    def test_default_prepend_bos_is_false(self, adapter: PhiArchitectureAdapter) -> None:
        """Phi was not trained with BOS token so TransformerLens should not append it"""
        assert adapter.cfg.default_prepend_bos is False


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestPhiAdapterComponentMapping:
    """Component mapping must have the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: PhiArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_is_rotary_embedding_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_rotary_emb_name(self, adapter: PhiArchitectureAdapter) -> None:
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_is_parallel_block_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        """Parallel attn+MLP requires ParallelBlockBridge, not sequential BlockBridge."""
        assert isinstance(adapter.component_mapping["blocks"], ParallelBlockBridge)

    def test_blocks_name(self, adapter: PhiArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: PhiArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.final_layernorm"

    def test_ln_final_use_native_layernorm_autograd_is_true(
        self, adapter: PhiArchitectureAdapter
    ) -> None:
        assert adapter.component_mapping["ln_final"].use_native_layernorm_autograd is True

    def test_unembed_is_unembedding_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: PhiArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_ln1_is_normalization_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_blocks_ln1_name(self, adapter: PhiArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_blocks_ln1_use_native_layernorm_autograd_is_true(
        self, adapter: PhiArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].use_native_layernorm_autograd is True

    def test_attn_is_position_embeddings_attention_bridge(
        self, adapter: PhiArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)

    def test_attn_name(self, adapter: PhiArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_requires_attention_mask_is_true(self, adapter: PhiArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_attention_mask is True

    def test_attn_requires_position_embeddings_is_true(
        self, adapter: PhiArchitectureAdapter
    ) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_position_embeddings is True

    def test_mlp_is_mlp_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: PhiArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "mlp"

    # -- Attention submodules --

    @pytest.mark.parametrize("slot", ["q", "k", "v", "o"])
    def test_attn_submodule_is_linear_bridge(
        self, adapter: PhiArchitectureAdapter, slot: str
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules[slot], LinearBridge)

    @pytest.mark.parametrize(
        "slot, hf_name",
        [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "dense")],
    )
    def test_attn_submodule_name(
        self, adapter: PhiArchitectureAdapter, slot: str, hf_name: str
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules[slot].name == hf_name

    # -- MLP submodules --

    def test_mlp_in_is_linear_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["in"], LinearBridge)

    def test_mlp_in_name(self, adapter: PhiArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "fc1"

    def test_mlp_out_is_linear_bridge(self, adapter: PhiArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["out"], LinearBridge)

    def test_mlp_out_name(self, adapter: PhiArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "fc2"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestPhiAdapterWeightConversions:
    """Adapter must define exactly the four QKVO weight and three QKV bias conversions."""

    def test_conversion_keys_present(self, adapter: PhiArchitectureAdapter) -> None:
        """Phi has 4 weight matrices (QKVO) and 3 bias vectors (QKV) per layer"""
        assert adapter.weight_processing_conversions.keys() == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.q.bias",
            "blocks.{i}.attn.k.bias",
            "blocks.{i}.attn.v.bias",
            "blocks.{i}.attn.o.weight",
        }

    @pytest.mark.parametrize("slot", ["q", "k", "v"])
    def test_qkv_weight_uses_split_heads_pattern(
        self, adapter: PhiArchitectureAdapter, slot: str
    ) -> None:
        conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    @pytest.mark.parametrize("slot", ["q", "k", "v"])
    def test_qkv_bias_uses_split_heads_pattern(
        self, adapter: PhiArchitectureAdapter, slot: str
    ) -> None:
        conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.bias"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) -> n h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_o_uses_merge_heads_pattern(self, adapter: PhiArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


class TestPhiArchitectureGuards:
    """Guard against accidental introduction of features Phi does not have."""

    def test_no_pos_embed_component(self, adapter: PhiArchitectureAdapter) -> None:
        """Phi uses rotary embeddings, so there is no learned positional embedding."""
        assert "pos_embed" not in adapter.component_mapping

    def test_no_ln2_in_blocks(self, adapter: PhiArchitectureAdapter) -> None:
        """Parallel attn+MLP shares a single ln_1; no ln2 exists."""
        blocks = adapter.component_mapping["blocks"]
        assert "ln2" not in blocks.submodules

    def test_no_gate_in_mlp(self, adapter: PhiArchitectureAdapter) -> None:
        """Phi uses a standard non-gated MLP; no gate submodule."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert "gate" not in mlp.submodules


# ---------------------------------------------------------------------------
# Setup component testing tests
# ---------------------------------------------------------------------------


class TestPhiSetupComponentTesting:
    """setup_component_testing must wire Phi's shared rotary embedding into attention bridges."""

    def test_sets_rotary_emb_on_template_attention(self, adapter: PhiArchitectureAdapter) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: PhiArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(self, adapter: PhiArchitectureAdapter) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb
