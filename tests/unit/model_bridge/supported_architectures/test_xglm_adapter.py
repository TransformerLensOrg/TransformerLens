"""Unit tests for XGLMArchitectureAdapter.

Tests cover:
- Config attribute validation (all required attributes set correctly) [Phase A]
- Weight conversion keys and structure [Phase A]
- Component mapping structure (correct bridge types and HF module paths) [Phase B]
- Embedding scale hook compatibility [Phase C]
- Factory registration (XGLMForCausalLM maps to the right adapter) [Phase D]
"""

import math
from types import SimpleNamespace

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    NormalizationBridge,
    SymbolicBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.xglm import (
    XGLMArchitectureAdapter,
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
    """Return a minimal TransformerBridgeConfig for XGLM adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="XGLMForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> XGLMArchitectureAdapter:
    return XGLMArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Phase A: Config attribute tests
# ---------------------------------------------------------------------------


class TestXGLMAdapterConfig:
    """Adapter must set all required config attributes to the correct values."""

    def test_normalization_type_is_ln(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_standard(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "standard"

    def test_final_rms_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm_is_false(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is False


# ---------------------------------------------------------------------------
# Phase A: Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestXGLMAdapterWeightConversions:
    """Adapter must define exactly the four standard QKVO weight conversions."""

    def test_q_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: XGLMArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: XGLMArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4


# ---------------------------------------------------------------------------
# Phase B: Component mapping structure tests
# ---------------------------------------------------------------------------


class TestXGLMAdapterComponentMapping:
    """Component mapping must have the correct bridge types and HF module paths."""

    def test_embed_is_embedding_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_no_pos_embed_in_mapping(self, adapter: XGLMArchitectureAdapter) -> None:
        # Sinusoidal embeddings have no weights — no bridge entry expected
        assert "pos_embed" not in adapter.component_mapping

    def test_blocks_is_block_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "model.layer_norm"

    def test_unembed_is_unembedding_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: XGLMArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"

    def test_ln1_is_normalization_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_ln1_name(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "self_attn_layer_norm"

    def test_attn_is_attention_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], AttentionBridge)

    def test_attn_name(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_requires_attention_mask(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].requires_attention_mask is True

    def test_attn_attention_mask_4d(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].attention_mask_4d is True

    def test_attn_q_name(self, adapter: XGLMArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["q"].name == "q_proj"

    def test_attn_k_name(self, adapter: XGLMArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["k"].name == "k_proj"

    def test_attn_v_name(self, adapter: XGLMArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["v"].name == "v_proj"

    def test_attn_o_name_is_out_proj(self, adapter: XGLMArchitectureAdapter) -> None:
        # Critical: XGLM uses out_proj, not o_proj (scaffold error pattern)
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["o"].name == "out_proj"

    def test_ln2_is_normalization_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)

    def test_ln2_name(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln2"].name == "final_layer_norm"

    def test_mlp_is_symbolic_bridge(self, adapter: XGLMArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], SymbolicBridge)

    def test_mlp_in_name(self, adapter: XGLMArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["in"].name == "fc1"

    def test_mlp_out_name(self, adapter: XGLMArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "fc2"


# ---------------------------------------------------------------------------
# Phase C: Embedding scale hook compatibility tests
# ---------------------------------------------------------------------------


def _make_mock_bridge() -> SimpleNamespace:
    """Return a minimal mock bridge with embed.hook_out for hook-compat tests."""
    hook_out = SimpleNamespace(hook_conversion=None)
    embed = SimpleNamespace(hook_out=hook_out)
    return SimpleNamespace(embed=embed)


class TestXGLMAdapterHookCompatibility:
    """setup_hook_compatibility must attach a scale conversion to hook_embed."""

    def test_sets_hook_conversion_on_embed_hook_out(self, adapter: XGLMArchitectureAdapter) -> None:
        bridge = _make_mock_bridge()
        adapter.setup_hook_compatibility(bridge)
        assert bridge.embed.hook_out.hook_conversion is not None

    def test_scales_by_sqrt_d_model(self, adapter: XGLMArchitectureAdapter) -> None:
        # d_model=64, sqrt(64)=8 exactly
        bridge = _make_mock_bridge()
        adapter.setup_hook_compatibility(bridge)
        conv = bridge.embed.hook_out.hook_conversion
        x = torch.ones(2, 4, 64)
        result = conv.handle_conversion(x)
        expected_scale = math.sqrt(64)  # 8.0
        assert torch.allclose(result, x * expected_scale, atol=1e-6)

    def test_revert_inverts_scale(self, adapter: XGLMArchitectureAdapter) -> None:
        # round-trip: revert(handle_conversion(x)) == x; exact for sqrt(64)=8
        bridge = _make_mock_bridge()
        adapter.setup_hook_compatibility(bridge)
        conv = bridge.embed.hook_out.hook_conversion
        x = torch.randn(2, 4, 64)
        assert torch.allclose(conv.revert(conv.handle_conversion(x)), x, atol=1e-6)

    def test_no_error_when_embed_missing(self, adapter: XGLMArchitectureAdapter) -> None:
        # Guard: if bridge lacks embed, setup_hook_compatibility should not raise
        bridge = SimpleNamespace()  # no embed attribute
        adapter.setup_hook_compatibility(bridge)  # must not raise

    def test_no_error_when_hook_out_missing(self, adapter: XGLMArchitectureAdapter) -> None:
        # Guard: if embed lacks hook_out, no error expected
        bridge = SimpleNamespace(embed=SimpleNamespace())  # embed but no hook_out
        adapter.setup_hook_compatibility(bridge)  # must not raise


# ---------------------------------------------------------------------------
# Phase D: Factory registration tests
# ---------------------------------------------------------------------------


class TestXGLMFactoryRegistration:
    """XGLMForCausalLM must be registered in SUPPORTED_ARCHITECTURES and resolve correctly."""

    def test_factory_returns_xglm_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, XGLMArchitectureAdapter)

    def test_factory_key_is_xglm_for_causal_lm(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "XGLMForCausalLM" in SUPPORTED_ARCHITECTURES
