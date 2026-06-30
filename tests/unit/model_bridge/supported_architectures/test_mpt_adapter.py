"""Unit tests for MPTArchitectureAdapter."""

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
    """Minimal TransformerBridgeConfig for MPT adapter tests (no HF Hub download)."""
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


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> MPTArchitectureAdapter:
    return MPTArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestMPTAdapterWeightConversions:
    def test_exactly_four_conversion_keys(self, adapter: MPTArchitectureAdapter) -> None:
        # No MLP conversions: up_proj/down_proj use standard [out, in] layout.
        assert len(adapter.weight_processing_conversions) == 4

    def test_no_mlp_conversion_keys(self, adapter: MPTArchitectureAdapter) -> None:
        keys = adapter.weight_processing_conversions
        assert not any("mlp" in k for k in keys)


# ---------------------------------------------------------------------------
# Component mapping structure tests (Phase B-1)
# ---------------------------------------------------------------------------


class TestMPTComponentMappingKeys:
    def test_top_level_keys_present(self, adapter: MPTArchitectureAdapter) -> None:
        keys = set(adapter.component_mapping.keys())
        assert {"embed", "blocks", "ln_final", "unembed"} <= keys

    def test_no_pos_embed_key(self, adapter: MPTArchitectureAdapter) -> None:
        # ALiBi: no learnable positional embedding module.
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
        # qkv/o are projection submodules; q/k/v are created during split.
        assert {"qkv", "o"} <= subkeys

    def test_mlp_submodule_keys(self, adapter: MPTArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        subkeys = set(mlp.submodules.keys())
        assert {"in", "out"} <= subkeys


# ---------------------------------------------------------------------------
# _split_mpt_qkv tests (Phase B-1)
# ---------------------------------------------------------------------------


class TestMPTSplitQKV:
    """_split_mpt_qkv decomposes Wqkv [3*d_model, d_model]."""

    def _make_fake_attn_component(self, d_model: int) -> object:
        """Stub with a Wqkv Linear (no bias, row-concat layout)."""

        class _FakeAttn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # MPT layout: Wqkv [3*d_model, d_model] row-wise concat.
                self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)

        return _FakeAttn()

    def test_split_returns_three_linears(self, adapter: MPTArchitectureAdapter) -> None:
        d_model = adapter.cfg.d_model
        fake_attn = self._make_fake_attn_component(d_model)
        result = adapter._split_mpt_qkv(fake_attn)
        assert len(result) == 3
        assert all(isinstance(lin, nn.Linear) for lin in result)

    def test_split_output_shapes(self, adapter: MPTArchitectureAdapter) -> None:
        d_model = adapter.cfg.d_model
        fake_attn = self._make_fake_attn_component(d_model)
        q_lin, k_lin, v_lin = adapter._split_mpt_qkv(fake_attn)
        for lin in (q_lin, k_lin, v_lin):
            assert lin.weight.shape == (d_model, d_model)

    def test_split_roundtrip(self, adapter: MPTArchitectureAdapter) -> None:
        """cat([q, k, v], dim=0) must recover original Wqkv (catches row/col transposition)."""
        d_model = adapter.cfg.d_model
        fake_attn = self._make_fake_attn_component(d_model)
        original_w = fake_attn.Wqkv.weight.detach().clone()

        q_lin, k_lin, v_lin = adapter._split_mpt_qkv(fake_attn)
        recovered = torch.cat([q_lin.weight, k_lin.weight, v_lin.weight], dim=0)

        assert torch.allclose(recovered, original_w)


# ---------------------------------------------------------------------------
# Component-mapping HF paths
# ---------------------------------------------------------------------------


class TestMPTComponentMappingPaths:
    """HF module paths per component slot (refactor-drift guard)."""

    def test_embed_path(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_blocks_path(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.blocks"

    def test_ln_final_path(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.norm_f"

    def test_unembed_path(self, adapter: MPTArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"


# ---------------------------------------------------------------------------
# Component-mapping bridge types
# ---------------------------------------------------------------------------


class TestMPTComponentTypes:
    """Component bridge classes (guards against silent type substitution)."""

    def test_embed_type(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components import EmbeddingBridge

        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_blocks_type(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components import BlockBridge

        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_ln_final_type(self, adapter: MPTArchitectureAdapter) -> None:
        # MPT uses LayerNorm (bias=None), not RMSNorm.
        from transformer_lens.model_bridge.generalized_components import (
            NormalizationBridge,
            RMSNormalizationBridge,
        )

        ln_final = adapter.component_mapping["ln_final"]
        assert isinstance(ln_final, NormalizationBridge)
        assert not isinstance(ln_final, RMSNormalizationBridge)

    def test_unembed_type(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components import (
            UnembeddingBridge,
        )

        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)


# ---------------------------------------------------------------------------
# Block-submodule structure (types + HF paths)
# ---------------------------------------------------------------------------


class TestMPTBlockSubmoduleStructure:
    """Each block submodule has the correct bridge type and HF path."""

    def test_ln1_is_layernorm_at_norm_1(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components import (
            NormalizationBridge,
        )

        block = adapter.component_mapping["blocks"]
        ln1 = block.submodules["ln1"]
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.name == "norm_1"

    def test_ln2_is_layernorm_at_norm_2(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components import (
            NormalizationBridge,
        )

        block = adapter.component_mapping["blocks"]
        ln2 = block.submodules["ln2"]
        assert isinstance(ln2, NormalizationBridge)
        assert ln2.name == "norm_2"

    def test_attn_is_mpt_alibi_attention_at_attn(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components.mpt_alibi_attention import (
            MPTALiBiAttentionBridge,
        )

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, MPTALiBiAttentionBridge)
        assert attn.name == "attn"

    def test_attn_does_not_require_position_embeddings(
        self, adapter: MPTArchitectureAdapter
    ) -> None:
        # ALiBi bakes position into the score bias: no rotary, no learned pos.
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_position_embeddings is False

    def test_attn_does_not_require_attention_mask(self, adapter: MPTArchitectureAdapter) -> None:
        # ALiBi bias slope IS the position-aware signal.
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.requires_attention_mask is False

    def test_attn_qkv_submodule_is_joint(self, adapter: MPTArchitectureAdapter) -> None:
        # MPT joint-QKV ("Wqkv") wires the joint Linear at the explicit "qkv" slot.
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert "qkv" in attn.submodules
        qkv_sub = attn.submodules["qkv"]
        assert isinstance(qkv_sub, LinearBridge)
        assert qkv_sub.name == "Wqkv"

    def test_attn_split_qkv_callback_wired(self, adapter: MPTArchitectureAdapter) -> None:
        # Bound methods are unwrapped on each access; compare via MethodType attrs.
        from types import MethodType

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        callback = attn.split_qkv_matrix
        assert isinstance(callback, MethodType)
        assert callback.__func__ is MPTArchitectureAdapter._split_mpt_qkv
        assert callback.__self__ is adapter

    def test_attn_o_submodule(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        attn = adapter.component_mapping["blocks"].submodules["attn"]
        o_sub = attn.submodules["o"]
        assert isinstance(o_sub, LinearBridge)
        assert o_sub.name == "out_proj"

    def test_mlp_is_plain_mlp_at_ffn(self, adapter: MPTArchitectureAdapter) -> None:
        # MPT MLP is non-gated.
        from transformer_lens.model_bridge.generalized_components import (
            GatedMLPBridge,
            MLPBridge,
        )

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, MLPBridge)
        assert not isinstance(mlp, GatedMLPBridge)
        assert mlp.name == "ffn"

    def test_mlp_submodule_paths(self, adapter: MPTArchitectureAdapter) -> None:
        from transformer_lens.model_bridge.generalized_components import LinearBridge

        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        for sub_name, expected_path in (("in", "up_proj"), ("out", "down_proj")):
            sub = mlp.submodules[sub_name]
            assert isinstance(sub, LinearBridge)
            assert sub.name == expected_path


# ---------------------------------------------------------------------------
# Weight conversion semantics (patterns + classes)
# ---------------------------------------------------------------------------


class TestMPTWeightConversionSemantics:
    """Each weight conversion entry uses the expected class and pattern."""

    def test_no_norm_offset_conversions(self, adapter: MPTArchitectureAdapter) -> None:
        # Plain LayerNorm: no +1 trick like Gemma.
        for key in adapter.weight_processing_conversions:
            assert not key.startswith("blocks.{i}.ln")
            assert key != "ln_final.weight"


# ---------------------------------------------------------------------------
# MQA / GQA propagation
# ---------------------------------------------------------------------------


class TestMPTMQASupport:
    """n_key_value_heads must reach K/V conversions (MPT supports MQA)."""

    def test_no_mqa_when_not_set(self) -> None:
        # Without n_key_value_heads, K/V default to n_heads.
        adapter = MPTArchitectureAdapter(_make_cfg(n_heads=2))
        kv_conv = adapter.weight_processing_conversions["blocks.{i}.attn.k.weight"]
        assert kv_conv.tensor_conversion.axes_lengths["n"] == 2


# ---------------------------------------------------------------------------
# Architecture-specific guards
# ---------------------------------------------------------------------------


class TestMPTArchitectureGuards:
    """No rotary, no pos_embed (MPT uses ALiBi)."""

    def test_no_rotary_emb_in_attn_submodules(self, adapter: MPTArchitectureAdapter) -> None:
        # ALiBi bias is computed inside the attention bridge: no rotary submodule.
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert "rotary_emb" not in attn.submodules
