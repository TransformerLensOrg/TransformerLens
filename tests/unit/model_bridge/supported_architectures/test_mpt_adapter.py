"""Unit tests for MPTArchitectureAdapter — Phase A (config + weight conversions),
Phase B-1 (component mapping + QKV split), and Phase D (factory registration).

Tests cover:
- Config attribute validation (all required attributes set correctly)
- Weight conversion keys (four standard QKVO keys with .weight suffix)
- LayerNorm with bias=None wraps without error (MptBlock sets norm.bias = None)
- Component mapping keys (embed/blocks/ln_final/unembed; no pos_embed/rotary_emb)
- Block/attn/mlp submodule keys
- _split_mpt_qkv: output shapes and round-trip correctness
- Factory resolves MPTForCausalLM -> MPTArchitectureAdapter (no download)
"""

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.supported_architectures.mpt import (
    MPTArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 2,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 256,
    n_ctx: int = 128,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for MPT adapter tests.

    Uses tiny dimensions — no HF Hub download required.
    """
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=False,
        architecture="MPTForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> MPTArchitectureAdapter:
    return MPTArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestMPTAdapterConfig:
    """Verify all required config attributes are set correctly."""

    def test_normalization_type_is_ln(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_alibi(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "alibi"

    def test_gated_mlp_is_false(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_final_rms_is_false(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_attn_only_is_false(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_default_prepend_bos_is_false(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestMPTAdapterWeightConversions:
    """Verify weight_processing_conversions has exactly the four QKVO keys."""

    def test_q_weight_key_present(self, adapter: MPTArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: MPTArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: MPTArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: MPTArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: MPTArchitectureAdapter) -> None:
        # No MLP conversions — up_proj/down_proj use standard [out, in] layout.
        assert len(adapter.weight_processing_conversions) == 4

    def test_no_mlp_conversion_keys(self, adapter: MPTArchitectureAdapter) -> None:
        keys = adapter.weight_processing_conversions
        assert not any("mlp" in k for k in keys), "MLP weights need no special conversion"


# ---------------------------------------------------------------------------
# LayerNorm with bias=None test
# ---------------------------------------------------------------------------


class TestMPTLayerNormBiasNone:
    """Verify NormalizationBridge handles MPT's bias=None LayerNorm correctly."""

    def test_layernorm_bias_none_wraps_without_error(self, cfg: TransformerBridgeConfig) -> None:
        """NormalizationBridge must accept and forward through a bias=None LayerNorm.

        MptBlock.__init__ explicitly sets norm_1.bias = None for backward compatibility
        with Hub weights. This test front-loads any surprise from that pattern.
        """
        from transformer_lens.model_bridge.generalized_components import (
            NormalizationBridge,
        )

        # Replicate what MptBlock does: LayerNorm then strip bias
        ln = nn.LayerNorm(cfg.d_model, eps=1e-5)
        ln.bias = None  # exactly as MptBlock.__init__ does

        bridge = NormalizationBridge(name="norm_1", config=cfg)
        bridge.set_original_component(ln)

        x = torch.randn(2, 4, cfg.d_model)
        with torch.no_grad():
            out = bridge(x)

        assert out.shape == x.shape, "Output shape must match input shape"
        assert not torch.isnan(out).any(), "Output must not contain NaN"
        assert not torch.isinf(out).any(), "Output must not contain Inf"


# ---------------------------------------------------------------------------
# Component mapping structure tests (Phase B-1)
# ---------------------------------------------------------------------------


class TestMPTComponentMappingKeys:
    """Verify top-level and nested component mapping keys are correct."""

    def test_top_level_keys_present(self, adapter: MPTArchitectureAdapter) -> None:
        keys = set(adapter.component_mapping.keys())
        assert {"embed", "blocks", "ln_final", "unembed"} <= keys

    def test_no_pos_embed_key(self, adapter: MPTArchitectureAdapter) -> None:
        # ALiBi has no learnable positional embedding module.
        assert "pos_embed" not in adapter.component_mapping

    def test_no_rotary_emb_key(self, adapter: MPTArchitectureAdapter) -> None:
        assert "rotary_emb" not in adapter.component_mapping

    def test_block_submodule_keys(self, adapter: MPTArchitectureAdapter) -> None:
        block = adapter.component_mapping["blocks"]
        subkeys = set(block.submodules.keys())
        assert {"ln1", "attn", "ln2", "mlp"} <= subkeys

    def test_attn_submodule_keys(self, adapter: MPTArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        subkeys = set(attn.submodules.keys())
        # qkv and o are the projection submodules; q/k/v are created during split
        assert {"qkv", "o"} <= subkeys

    def test_mlp_submodule_keys(self, adapter: MPTArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        subkeys = set(mlp.submodules.keys())
        assert {"in", "out"} <= subkeys


# ---------------------------------------------------------------------------
# _split_mpt_qkv tests (Phase B-1)
# ---------------------------------------------------------------------------


class TestMPTSplitQKV:
    """Verify _split_mpt_qkv correctly decomposes Wqkv [3*d_model, d_model]."""

    def _make_fake_attn_component(self, d_model: int) -> object:
        """Return a stub object with a Wqkv Linear attribute (no bias, row-concat layout)."""

        class _FakeAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Wqkv: [3*d_model, d_model] — MPT row-wise concat layout
                self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)

        return _FakeAttn()

    def test_split_returns_three_linears(self, adapter: MPTArchitectureAdapter) -> None:
        d_model = adapter.cfg.d_model
        fake_attn = self._make_fake_attn_component(d_model)
        result = adapter._split_mpt_qkv(fake_attn)
        assert len(result) == 3
        assert all(isinstance(lin, nn.Linear) for lin in result)

    def test_split_output_shapes(self, adapter: MPTArchitectureAdapter) -> None:
        """Each output linear must have weight shape [d_model, d_model]."""
        d_model = adapter.cfg.d_model
        fake_attn = self._make_fake_attn_component(d_model)
        q_lin, k_lin, v_lin = adapter._split_mpt_qkv(fake_attn)
        for lin in (q_lin, k_lin, v_lin):
            assert lin.weight.shape == (
                d_model,
                d_model,
            ), f"Expected ({d_model}, {d_model}), got {lin.weight.shape}"

    def test_split_roundtrip(self, adapter: MPTArchitectureAdapter) -> None:
        """cat([q.weight, k.weight, v.weight], dim=0) must recover original Wqkv.weight.

        Uses batch_size=2 worth of distinct rows to surface any row/col transposition.
        """
        d_model = adapter.cfg.d_model
        fake_attn = self._make_fake_attn_component(d_model)
        original_w = fake_attn.Wqkv.weight.detach().clone()  # [3*d_model, d_model]

        q_lin, k_lin, v_lin = adapter._split_mpt_qkv(fake_attn)
        recovered = torch.cat([q_lin.weight, k_lin.weight, v_lin.weight], dim=0)

        assert torch.allclose(
            recovered, original_w
        ), "Round-trip failed: cat(Q,K,V) != original Wqkv"


# ---------------------------------------------------------------------------
# Factory registration test (Phase D)
# ---------------------------------------------------------------------------


class TestMPTFactoryRegistration:
    """ArchitectureAdapterFactory must resolve MPTForCausalLM -> MPTArchitectureAdapter."""

    def test_factory_resolves_mpt_architecture(self) -> None:
        """Factory returns an MPTArchitectureAdapter instance for MPTForCausalLM.

        Uses a fully programmatic config — no HF Hub download.
        """
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        cfg.architecture = "MPTForCausalLM"
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, MPTArchitectureAdapter)

    def test_factory_unknown_architecture_raises(self) -> None:
        """Factory raises ValueError for an unregistered architecture key."""
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        cfg.architecture = "NonExistentForCausalLM"
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ArchitectureAdapterFactory.select_architecture_adapter(cfg)

    def test_mpt_in_supported_architectures_dict(self) -> None:
        """MPTForCausalLM must appear in the SUPPORTED_ARCHITECTURES mapping."""
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "MPTForCausalLM" in SUPPORTED_ARCHITECTURES
        assert SUPPORTED_ARCHITECTURES["MPTForCausalLM"] is MPTArchitectureAdapter
