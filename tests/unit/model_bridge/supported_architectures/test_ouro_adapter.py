"""Unit tests for OuroArchitectureAdapter.

Tests cover:
- Config attributes set by the adapter
- Component mapping structure and HF module paths, including Ouro's
  sandwich normalization (four RMSNorms per decoder layer)
- Standard Q/K/V/O weight conversion rules
- setup_component_testing rotary embedding wiring
- Factory registration

Ouro's Universal-Transformer loop and early-exit gate live in the remote-code
HF forward and are deliberately NOT mapped by the adapter; the top-level-keys
test pins that scope (no "gate" / per-step components).
"""

from types import SimpleNamespace

import pytest

from transformer_lens.config import TransformerBridgeConfig
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
from transformer_lens.model_bridge.supported_architectures.ouro import (
    OuroArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 100,
    n_ctx: int = 64,
) -> TransformerBridgeConfig:
    # Keep dimensions tiny so adapter tests do not need HF downloads or real checkpoints.
    # Ouro uses full MHA (num_key_value_heads == num_attention_heads).
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        n_key_value_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        architecture="OuroForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> OuroArchitectureAdapter:
    return OuroArchitectureAdapter(cfg)


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


class TestOuroAdapterConfig:
    """Adapter-owned config defaults that downstream bridge code relies on."""

    def test_normalization_flags(self, adapter: OuroArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"
        assert adapter.cfg.uses_rms_norm is True
        assert adapter.cfg.final_rms is True

    def test_rotary_and_mlp_flags(self, adapter: OuroArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"
        assert adapter.cfg.gated_mlp is True
        assert adapter.cfg.attn_only is False

    def test_no_rmsnorm_offset(self, adapter: OuroArchitectureAdapter) -> None:
        """Ouro's RMSNorm applies the weight directly (no Gemma-style +1.0 offset)."""
        assert not getattr(adapter.cfg, "rmsnorm_uses_offset", False)

    def test_supports_fold_ln_is_false(self, adapter: OuroArchitectureAdapter) -> None:
        """ln_final runs after every UT pass, feeding the next pass and the exit
        gate; folding it into W_U would corrupt passes 1..N-1 in the live module."""
        assert adapter.supports_fold_ln is False


class TestOuroComponentMapping:
    """The adapter contract: TL canonical names mapped to Ouro HF module paths."""

    def test_top_level_keys(self, adapter: OuroArchitectureAdapter) -> None:
        # No "gate" key: model.early_exit_gate is intentionally unmapped.
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_embed_path(self, adapter: OuroArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_path(self, adapter: OuroArchitectureAdapter) -> None:
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_blocks_path(self, adapter: OuroArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_path(self, adapter: OuroArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_path(self, adapter: OuroArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: OuroArchitectureAdapter) -> None:
        """Sandwich norm: four RMSNorms per block, not the usual two."""
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {
            "ln1",
            "ln1_post",
            "ln2",
            "ln2_post",
            "attn",
            "mlp",
        }

    def test_sandwich_norm_hf_paths(self, adapter: OuroArchitectureAdapter) -> None:
        """The extra norms map to Ouro's *_2 module names.

        Forward order in OuroDecoderLayer: input_layernorm (pre-attn) -> attn
        -> input_layernorm_2 (post-attn, pre-residual); post_attention_layernorm
        (pre-MLP) -> mlp -> post_attention_layernorm_2 (post-MLP, pre-residual).
        """
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln1_post"].name == "input_layernorm_2"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["ln2_post"].name == "post_attention_layernorm_2"

    def test_attention_submodule_keys(self, adapter: OuroArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"q", "k", "v", "o"}

    def test_mlp_submodule_keys(self, adapter: OuroArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_bridge_types(self, adapter: OuroArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        blocks = mapping["blocks"]
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(blocks, BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        for ln_key in ("ln1", "ln1_post", "ln2", "ln2_post"):
            assert isinstance(blocks.submodules[ln_key], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)

    def test_attention_hf_paths(self, adapter: OuroArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.name == "self_attn"
        assert attn.submodules["q"].name == "q_proj"
        assert attn.submodules["k"].name == "k_proj"
        assert attn.submodules["v"].name == "v_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_hf_paths(self, adapter: OuroArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.name == "mlp"
        assert mlp.submodules["gate"].name == "gate_proj"
        assert mlp.submodules["in"].name == "up_proj"
        assert mlp.submodules["out"].name == "down_proj"

    def test_linear_submodule_bridge_types(self, adapter: OuroArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        attn = blocks.submodules["attn"]
        mlp = blocks.submodules["mlp"]
        for submodule in [*attn.submodules.values(), *mlp.submodules.values()]:
            assert isinstance(submodule, LinearBridge)


class TestOuroWeightConversions:
    """Standard split-QKV conversion rules from _qkvo_weight_conversions()."""

    def test_qkvo_conversion_keys_present(self, adapter: OuroArchitectureAdapter) -> None:
        for key in (
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        ):
            assert key in adapter.weight_processing_conversions


class TestOuroSetupComponentTesting:
    """setup_component_testing must wire Ouro's shared rotary embedding into attention bridges."""

    def test_sets_rotary_emb_on_template_attention(self, adapter: OuroArchitectureAdapter) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, PositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: OuroArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(self, adapter: OuroArchitectureAdapter) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb


class TestOuroFactoryRegistration:
    """The factory resolves OuroForCausalLM to this adapter."""

    def test_supported_architectures_entry(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert SUPPORTED_ARCHITECTURES["OuroForCausalLM"] is OuroArchitectureAdapter

    def test_factory_selects_ouro_adapter(self, cfg: TransformerBridgeConfig) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, OuroArchitectureAdapter)
