"""Unit tests for GptjArchitectureAdapter.

Tests cover:
- Config attribute validation
- Component mapping structure
- Weight conversion keys and rearrange patterns
- Architecture guards
- Factory registration
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gptj import (
    GptjArchitectureAdapter,
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
    """Return a minimal TransformerBridgeConfig for GPT-J adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="GPTJForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> GptjArchitectureAdapter:
    return GptjArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestGptjAdapterConfig:
    """Adapter must set all required config flags to the values GPT-J expects."""

    def test_normalization_type_is_ln(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_rotary(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms_is_false(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_parallel_attn_mlp_is_true(self, adapter: GptjArchitectureAdapter) -> None:
        """GPT-J runs attention and MLP in parallel within each block."""
        assert adapter.cfg.parallel_attn_mlp is True


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestGptjAdapterComponentMapping:
    """Component mapping must have the correct bridge types and HF module names."""

    # -- Top-level keys --

    def test_embed_is_embedding_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_blocks_is_block_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_is_parallel_block_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        """Parallel attn+MLP requires ParallelBlockBridge, not sequential BlockBridge."""
        assert isinstance(adapter.component_mapping["blocks"], ParallelBlockBridge)

    def test_blocks_name(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.h"

    def test_ln_final_is_normalization_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: GptjArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_ln1_is_normalization_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_blocks_ln1_name(self, adapter: GptjArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "ln_1"

    def test_attn_is_attention_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: GptjArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attn"

    def test_mlp_is_mlp_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: GptjArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "mlp"

    # -- Attention submodules --

    @pytest.mark.parametrize("slot", ["q", "k", "v", "o"])
    def test_attn_submodule_is_linear_bridge(
        self, adapter: GptjArchitectureAdapter, slot: str
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn.submodules[slot], LinearBridge)

    @pytest.mark.parametrize(
        "slot, hf_name",
        [("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "out_proj")],
    )
    def test_attn_submodule_name(
        self, adapter: GptjArchitectureAdapter, slot: str, hf_name: str
    ) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules[slot].name == hf_name

    # -- MLP submodules --

    def test_mlp_in_is_linear_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["in"], LinearBridge)

    def test_mlp_in_name(self, adapter: GptjArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "fc_in"

    def test_mlp_out_is_linear_bridge(self, adapter: GptjArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp.submodules["out"], LinearBridge)

    def test_mlp_out_name(self, adapter: GptjArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "fc_out"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestGptjAdapterWeightConversions:
    """Adapter must define exactly the four QKVO weight conversions."""

    @pytest.mark.parametrize(
        "key",
        [
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        ],
    )
    def test_conversion_key_present(self, adapter: GptjArchitectureAdapter, key: str) -> None:
        assert key in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: GptjArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4

    @pytest.mark.parametrize("slot", ["q", "k", "v"])
    def test_qkv_uses_split_heads_pattern(
        self, adapter: GptjArchitectureAdapter, slot: str
    ) -> None:
        conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_o_uses_merge_heads_pattern(self, adapter: GptjArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


# ---------------------------------------------------------------------------
# Architecture guards
# ---------------------------------------------------------------------------


class TestGptjArchitectureGuards:
    """Guard against accidental introduction of features GPT-J does not have."""

    def test_no_pos_embed_component(self, adapter: GptjArchitectureAdapter) -> None:
        """GPT-J uses rotary embeddings, so there is no learned positional embedding."""
        assert "pos_embed" not in adapter.component_mapping

    def test_no_top_level_rotary_emb(self, adapter: GptjArchitectureAdapter) -> None:
        """Rotary is applied inside attention; no standalone HF module to bind."""
        assert "rotary_emb" not in adapter.component_mapping

    def test_no_ln2_in_blocks(self, adapter: GptjArchitectureAdapter) -> None:
        """Parallel attn+MLP shares a single ln_1; no ln2 exists."""
        blocks = adapter.component_mapping["blocks"]
        assert "ln2" not in blocks.submodules

    def test_no_gate_in_mlp(self, adapter: GptjArchitectureAdapter) -> None:
        """GPT-J uses a standard non-gated MLP; no gate submodule."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert "gate" not in mlp.submodules

    def test_only_qkvo_conversion_keys(self, adapter: GptjArchitectureAdapter) -> None:
        """Only QKVO weights need reshape; no norm offsets, no embedding conversions."""
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestGptjFactoryRegistration:
    """ArchitectureAdapterFactory must resolve GPTJForCausalLM -> GptjArchitectureAdapter."""

    def test_factory_returns_gptj_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(_make_cfg())
        assert isinstance(adapter, GptjArchitectureAdapter)

    def test_gptj_in_supported_architectures_dict(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "GPTJForCausalLM" in SUPPORTED_ARCHITECTURES
        assert SUPPORTED_ARCHITECTURES["GPTJForCausalLM"] is GptjArchitectureAdapter
