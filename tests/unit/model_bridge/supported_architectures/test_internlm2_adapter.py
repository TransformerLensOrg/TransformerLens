"""Unit tests for InternLM2ArchitectureAdapter.

Tests cover (one class per phase):
- Phase A: Config attributes, weight conversion keys/types, split_wqkv numerics,
           preprocess_weights behaviour
- Phase D: Factory registration
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.internlm2 import (
    InternLM2ArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 8,
    n_key_value_heads: int = 2,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 100,
    n_ctx: int = 128,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for InternLM2 adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        n_key_value_heads=n_key_value_heads,
        default_prepend_bos=True,
        architecture="InternLM2ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> InternLM2ArchitectureAdapter:
    return InternLM2ArchitectureAdapter(cfg)


def _make_attn_component(
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    d_model: int,
    has_bias: bool = False,
) -> Any:
    """Synthetic attention namespace with a wqkv linear (no model download needed)."""
    total_out = (n_heads + 2 * n_kv_heads) * head_dim
    ns = SimpleNamespace()
    ns.wqkv = nn.Linear(d_model, total_out, bias=has_bias)
    return ns


def _fill_interleaved(
    wqkv_linear: nn.Linear,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    d_model: int,
    kv_group_vals: list[tuple[float, float, float]],
) -> None:
    """Fill wqkv weight with per-kv-group constants for layout verification.

    kv_group_vals: list of (q_val, k_val, v_val) per kv-head group.
    """
    n_kv_groups = n_heads // n_kv_heads
    gs = n_kv_groups + 2
    w = torch.zeros(n_kv_heads, gs, head_dim, d_model)
    for h, (q_val, k_val, v_val) in enumerate(kv_group_vals):
        w[h, :n_kv_groups, :, :] = q_val
        w[h, n_kv_groups, :, :] = k_val
        w[h, n_kv_groups + 1, :, :] = v_val
    wqkv_linear.weight = nn.Parameter(w.reshape((n_heads + 2 * n_kv_heads) * head_dim, d_model))


# ---------------------------------------------------------------------------
# Phase A — Config attribute tests
# ---------------------------------------------------------------------------


class TestInternLM2AdapterConfig:
    """Adapter must set all required config attributes."""

    def test_normalization_type(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_eps_attr(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.eps_attr == "variance_epsilon"

    def test_n_key_value_heads_propagated(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 2

    def test_supports_fold_ln_false(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # Must be False: fold_ln silently skips attn when wqkv is fused in bridge state dict.
        assert adapter.supports_fold_ln is False


# ---------------------------------------------------------------------------
# Phase A — Component mapping structure tests
# ---------------------------------------------------------------------------


class TestInternLM2AdapterComponentMapping:
    """component_mapping must have correct bridge types and InternLM2-specific names."""

    def test_embed_is_embedding_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses tok_embeddings, not embed_tokens
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["embed"].name == "model.tok_embeddings"

    def test_no_top_level_rotary_emb(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # Per-layer rotary injected via setup_component_testing, not top-level mapping
        assert adapter.component_mapping is not None
        assert "rotary_emb" not in adapter.component_mapping

    def test_blocks_is_block_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_is_rms_normalization_bridge(
        self, adapter: InternLM2ArchitectureAdapter
    ) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["ln_final"], RMSNormalizationBridge)

    def test_ln_final_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_is_unembedding_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses 'output', not 'lm_head'
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["unembed"].name == "output"

    def test_ln1_is_rms_normalization_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)

    def test_ln1_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses attention_norm, not input_layernorm
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "attention_norm"

    def test_ln2_is_rms_normalization_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)

    def test_ln2_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses ffn_norm, not post_attention_layernorm
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln2"].name == "ffn_norm"

    def test_attn_is_joint_qkv_position_embeddings_attention_bridge(
        self, adapter: InternLM2ArchitectureAdapter
    ) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], JointQKVPositionEmbeddingsAttentionBridge)

    def test_attn_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses 'attention', not 'self_attn'
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attention"

    def test_attn_qkv_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].submodules["qkv"].name == "wqkv"

    def test_attn_o_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].submodules["o"].name == "wo"

    def test_mlp_is_gated_mlp_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)

    def test_mlp_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses 'feed_forward', not 'mlp'
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "feed_forward"

    def test_mlp_gate_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # w1 = gate projection
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["gate"].name == "w1"

    def test_mlp_in_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # w3 = up/in projection
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["in"].name == "w3"

    def test_mlp_out_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # w2 = down/out projection
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["out"].name == "w2"


# ---------------------------------------------------------------------------
# Phase A — Weight conversion key and type tests
# ---------------------------------------------------------------------------


class TestInternLM2AdapterWeightConversions:
    """weight_processing_conversions must have correct keys, types, and rearrange patterns."""

    def test_q_weight_key_present(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # No bias entries for the bias=False shipped config
        assert adapter.weight_processing_conversions is not None
        assert len(adapter.weight_processing_conversions) == 4

    def test_q_conversion_is_param_processing_conversion(
        self, adapter: InternLM2ArchitectureAdapter
    ) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)

    def test_q_tensor_conversion_is_rearrange(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)

    def test_q_rearrange_pattern(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_q_rearrange_n_equals_n_heads(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_k_rearrange_n_equals_n_kv_heads(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.k.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_v_rearrange_n_equals_n_kv_heads(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.v.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_key_value_heads

    def test_o_rearrange_pattern(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_o_rearrange_n_equals_n_heads(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_no_source_key_on_q(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # preprocess_weights writes split keys; no cross-key lookup needed at rearrange time
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert conv.source_key is None


# ---------------------------------------------------------------------------
# Phase A — _split_internlm2_wqkv numerical tests
# ---------------------------------------------------------------------------


class TestInternLM2SplitWqkv:
    """Numerical correctness of the interleaved GQA split function."""

    def _adapter(
        self,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        d_model: int = 32,
    ) -> InternLM2ArchitectureAdapter:
        head_dim = d_model // n_heads
        return InternLM2ArchitectureAdapter(
            _make_cfg(n_heads=n_heads, n_key_value_heads=n_kv_heads, d_model=d_model)
        )

    def test_returns_three_linears(self) -> None:
        adapter = self._adapter()
        attn = _make_attn_component(8, 2, 4, 32)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)

    def test_gqa_shapes(self) -> None:
        # n_heads=8, n_kv_heads=2, head_dim=4, d_model=32
        adapter = self._adapter(n_heads=8, n_kv_heads=2, d_model=32)
        attn = _make_attn_component(8, 2, 4, 32)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert q.weight.shape == (8 * 4, 32)
        assert k.weight.shape == (2 * 4, 32)
        assert v.weight.shape == (2 * 4, 32)

    def test_mha_shapes(self) -> None:
        # MHA: n_heads == n_kv_heads → gs=3 (standard [Q|K|V])
        adapter = self._adapter(n_heads=4, n_kv_heads=4, d_model=32)
        attn = _make_attn_component(4, 4, 8, 32)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert q.weight.shape == (4 * 8, 32)
        assert k.weight.shape == (4 * 8, 32)
        assert v.weight.shape == (4 * 8, 32)

    def test_interleaved_layout_correctness(self) -> None:
        # n_heads=4, n_kv_heads=2, head_dim=4, d_model=16 → gs=4 (2 q-groups + k + v)
        n_heads, n_kv_heads, head_dim, d_model = 4, 2, 4, 16
        adapter = self._adapter(n_heads=n_heads, n_kv_heads=n_kv_heads, d_model=d_model)
        attn = _make_attn_component(n_heads, n_kv_heads, head_dim, d_model)
        # kv-group 0: Q=1.0, K=2.0, V=3.0; kv-group 1: Q=4.0, K=5.0, V=6.0
        _fill_interleaved(
            attn.wqkv,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        q, k, v = adapter._split_internlm2_wqkv(attn)

        n_kv_groups = n_heads // n_kv_heads  # 2
        # Q: rows 0..n_kv_groups*head_dim-1 come from kv-group 0 Q slots (1.0),
        #    rows n_kv_groups*head_dim..n_heads*head_dim-1 from kv-group 1 Q slots (4.0)
        assert torch.all(q.weight[: n_kv_groups * head_dim] == 1.0), "Q group-0 rows should be 1.0"
        assert torch.all(q.weight[n_kv_groups * head_dim :] == 4.0), "Q group-1 rows should be 4.0"
        # K: row 0..head_dim-1 = kv-group 0 K (2.0), head_dim..2*head_dim-1 = kv-group 1 K (5.0)
        assert torch.all(k.weight[:head_dim] == 2.0), "K group-0 rows should be 2.0"
        assert torch.all(k.weight[head_dim:] == 5.0), "K group-1 rows should be 5.0"
        # V analogous
        assert torch.all(v.weight[:head_dim] == 3.0), "V group-0 rows should be 3.0"
        assert torch.all(v.weight[head_dim:] == 6.0), "V group-1 rows should be 6.0"

    def test_no_bias(self) -> None:
        adapter = self._adapter()
        attn = _make_attn_component(8, 2, 4, 32, has_bias=False)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert q.bias is None
        assert k.bias is None
        assert v.bias is None

    def test_with_bias_shapes(self) -> None:
        n_heads, n_kv_heads, head_dim, d_model = 8, 2, 4, 32
        adapter = self._adapter(n_heads=n_heads, n_kv_heads=n_kv_heads, d_model=d_model)
        attn = _make_attn_component(n_heads, n_kv_heads, head_dim, d_model, has_bias=True)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert q.bias is not None
        assert k.bias is not None
        assert v.bias is not None
        assert q.bias.shape == (n_heads * head_dim,)
        assert k.bias.shape == (n_kv_heads * head_dim,)
        assert v.bias.shape == (n_kv_heads * head_dim,)

    def test_with_bias_interleaved_values(self) -> None:
        # Verify bias values follow the same interleaved layout as weights
        n_heads, n_kv_heads, head_dim, d_model = 4, 2, 4, 16
        adapter = self._adapter(n_heads=n_heads, n_kv_heads=n_kv_heads, d_model=d_model)
        attn = _make_attn_component(n_heads, n_kv_heads, head_dim, d_model, has_bias=True)
        n_kv_groups = n_heads // n_kv_heads
        gs = n_kv_groups + 2
        # Bias: interleaved [q0_vals, q1_vals, k_val, v_val] per kv-head group
        b = torch.zeros((n_heads + 2 * n_kv_heads) * head_dim)
        b_grouped = b.reshape(n_kv_heads, gs, head_dim)
        b_grouped[0, :n_kv_groups, :] = 1.0  # kv-group 0 Q bias
        b_grouped[0, n_kv_groups, :] = 2.0  # kv-group 0 K bias
        b_grouped[0, n_kv_groups + 1, :] = 3.0  # kv-group 0 V bias
        b_grouped[1, :n_kv_groups, :] = 4.0  # kv-group 1 Q bias
        b_grouped[1, n_kv_groups, :] = 5.0
        b_grouped[1, n_kv_groups + 1, :] = 6.0
        attn.wqkv.bias = nn.Parameter(b_grouped.reshape(-1))

        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert torch.all(q.bias[: n_kv_groups * head_dim] == 1.0)
        assert torch.all(q.bias[n_kv_groups * head_dim :] == 4.0)
        assert torch.all(k.bias[:head_dim] == 2.0)
        assert torch.all(k.bias[head_dim:] == 5.0)
        assert torch.all(v.bias[:head_dim] == 3.0)
        assert torch.all(v.bias[head_dim:] == 6.0)

    def test_forward_output_shapes(self) -> None:
        n_heads, n_kv_heads, head_dim, d_model = 8, 2, 4, 32
        adapter = self._adapter(n_heads=n_heads, n_kv_heads=n_kv_heads, d_model=d_model)
        attn = _make_attn_component(n_heads, n_kv_heads, head_dim, d_model)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        x = torch.randn(2, 5, d_model)
        assert q(x).shape == (2, 5, n_heads * head_dim)
        assert k(x).shape == (2, 5, n_kv_heads * head_dim)
        assert v(x).shape == (2, 5, n_kv_heads * head_dim)


# ---------------------------------------------------------------------------
# Phase A — preprocess_weights tests
# ---------------------------------------------------------------------------


class TestInternLM2PreprocessWeights:
    """preprocess_weights must split fused wqkv and fold layer norms."""

    def _make_state_dict_with_fused_qkv(
        self,
        adapter: InternLM2ArchitectureAdapter,
        n_kv_heads: int,
        head_dim: int,
        d_model: int,
        n_layers: int,
        ln1_scale: float = 1.0,
        qkv_val: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Build a bridge-format state dict with fused qkv.weight for each layer."""
        n_heads = adapter.cfg.n_heads
        n_kv_groups = n_heads // n_kv_heads
        gs = n_kv_groups + 2
        state: dict[str, torch.Tensor] = {}
        for i in range(n_layers):
            total_rows = (n_heads + 2 * n_kv_heads) * head_dim
            state[f"blocks.{i}.attn.qkv.weight"] = torch.full((total_rows, d_model), qkv_val)
            state[f"blocks.{i}.ln1.weight"] = torch.full((d_model,), ln1_scale)
            state[f"blocks.{i}.ln2.weight"] = torch.ones(d_model)
            state[f"blocks.{i}.mlp.gate.weight"] = torch.ones(16, d_model)
            state[f"blocks.{i}.mlp.in.weight"] = torch.ones(16, d_model)
        state["ln_final.weight"] = torch.ones(d_model)
        state["unembed.weight"] = torch.ones(100, d_model)
        return state

    def test_fused_key_removed_and_split_keys_written(self) -> None:
        adapter = InternLM2ArchitectureAdapter(_make_cfg())
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(adapter, n_kv_heads, head_dim, d_model, 2)

        result = adapter.preprocess_weights(sd)

        assert "blocks.0.attn.qkv.weight" not in result, "fused qkv key must be deleted"
        assert "blocks.0.attn.q.weight" in result
        assert "blocks.0.attn.k.weight" in result
        assert "blocks.0.attn.v.weight" in result

    def test_split_q_shape(self) -> None:
        adapter = InternLM2ArchitectureAdapter(
            _make_cfg(n_heads=8, n_key_value_heads=2, d_model=64)
        )
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(adapter, n_kv_heads, head_dim, d_model, 2)
        result = adapter.preprocess_weights(sd)
        assert result["blocks.0.attn.q.weight"].shape == (8 * 8, 64)
        assert result["blocks.0.attn.k.weight"].shape == (2 * 8, 64)
        assert result["blocks.0.attn.v.weight"].shape == (2 * 8, 64)

    def test_ln1_fold_applied_to_q(self) -> None:
        """After folding ln1 scale=2.0 into qkv (all 1.0), q/k/v weights should be 2.0."""
        adapter = InternLM2ArchitectureAdapter(
            _make_cfg(n_heads=8, n_key_value_heads=2, d_model=64)
        )
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(
            adapter, n_kv_heads, head_dim, d_model, 2, ln1_scale=2.0, qkv_val=1.0
        )
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["blocks.0.attn.q.weight"] == 2.0)
        assert torch.all(result["blocks.0.attn.k.weight"] == 2.0)
        assert torch.all(result["blocks.0.attn.v.weight"] == 2.0)

    def test_ln1_reset_to_ones(self) -> None:
        adapter = InternLM2ArchitectureAdapter(_make_cfg())
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(
            adapter, n_kv_heads, head_dim, d_model, 2, ln1_scale=3.0
        )
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["blocks.0.ln1.weight"] == 1.0)

    def test_ln2_fold_applied_to_mlp_gate(self) -> None:
        adapter = InternLM2ArchitectureAdapter(_make_cfg())
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(adapter, n_kv_heads, head_dim, d_model, 2)
        # Override ln2 with scale=3.0
        sd["blocks.0.ln2.weight"] = torch.full((d_model,), 3.0)
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["blocks.0.mlp.gate.weight"] == 3.0)
        assert torch.all(result["blocks.0.mlp.in.weight"] == 3.0)

    def test_ln2_reset_to_ones(self) -> None:
        adapter = InternLM2ArchitectureAdapter(_make_cfg())
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(adapter, n_kv_heads, head_dim, d_model, 2)
        sd["blocks.0.ln2.weight"] = torch.full((d_model,), 5.0)
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["blocks.0.ln2.weight"] == 1.0)

    def test_ln_final_fold_applied_to_unembed(self) -> None:
        adapter = InternLM2ArchitectureAdapter(_make_cfg())
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(adapter, n_kv_heads, head_dim, d_model, 2)
        sd["ln_final.weight"] = torch.full((d_model,), 2.0)
        sd["unembed.weight"] = torch.ones(100, d_model)
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["unembed.weight"] == 2.0)
        assert torch.all(result["ln_final.weight"] == 1.0)

    def test_no_fold_when_not_requested(self) -> None:
        adapter = InternLM2ArchitectureAdapter(_make_cfg())
        adapter._fold_ln_requested = False
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(
            adapter, n_kv_heads, head_dim, d_model, 2, ln1_scale=5.0
        )
        result = adapter.preprocess_weights(sd)
        # Fused key must still be present; no splitting or scaling
        assert "blocks.0.attn.qkv.weight" in result
        assert "blocks.0.attn.q.weight" not in result

    def test_dtype_preserved(self) -> None:
        adapter = InternLM2ArchitectureAdapter(_make_cfg())
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(adapter, n_kv_heads, head_dim, d_model, 1)
        # Cast to bfloat16
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        result = adapter.preprocess_weights(sd)
        assert result["blocks.0.attn.q.weight"].dtype == torch.bfloat16

    def test_bias_split_when_present(self) -> None:
        """config.bias=True: fused bias must be split into q/k/v bias keys."""
        # Use consistent d_model/n_heads so head_dim = d_model // n_heads = 64 // 4 = 16
        n_heads, n_kv_heads, d_model = 4, 2, 64
        head_dim = d_model // n_heads  # 16
        adapter = InternLM2ArchitectureAdapter(
            _make_cfg(n_heads=n_heads, n_key_value_heads=n_kv_heads, d_model=d_model)
        )
        adapter._fold_ln_requested = True
        total_rows = (n_heads + 2 * n_kv_heads) * head_dim
        sd: dict[str, torch.Tensor] = {
            "blocks.0.attn.qkv.weight": torch.ones(total_rows, d_model),
            "blocks.0.attn.qkv.bias": torch.zeros(total_rows),
            "blocks.0.ln1.weight": torch.ones(d_model),
            "blocks.0.ln2.weight": torch.ones(d_model),
            "blocks.0.mlp.gate.weight": torch.ones(16, d_model),
            "blocks.0.mlp.in.weight": torch.ones(16, d_model),
            "ln_final.weight": torch.ones(d_model),
            "unembed.weight": torch.ones(100, d_model),
        }
        result = adapter.preprocess_weights(sd)
        assert "blocks.0.attn.qkv.bias" not in result
        assert "blocks.0.attn.q.bias" in result
        assert "blocks.0.attn.k.bias" in result
        assert "blocks.0.attn.v.bias" in result
        assert result["blocks.0.attn.q.bias"].shape == (n_heads * head_dim,)
        assert result["blocks.0.attn.k.bias"].shape == (n_kv_heads * head_dim,)
        assert result["blocks.0.attn.v.bias"].shape == (n_kv_heads * head_dim,)

    def test_all_layers_processed(self) -> None:
        """Verify that all n_layers are processed, not just layer 0."""
        adapter = InternLM2ArchitectureAdapter(_make_cfg(n_layers=3))
        adapter._fold_ln_requested = True
        n_kv_heads, head_dim, d_model = 2, 8, 64
        sd = self._make_state_dict_with_fused_qkv(adapter, n_kv_heads, head_dim, d_model, 3)
        result = adapter.preprocess_weights(sd)
        for i in range(3):
            assert f"blocks.{i}.attn.qkv.weight" not in result
            assert f"blocks.{i}.attn.q.weight" in result


# ---------------------------------------------------------------------------
# Phase D — Factory registration (will pass after Phase D implemented)
# ---------------------------------------------------------------------------


class TestInternLM2FactoryRegistration:
    """Factory must map InternLM2ForCausalLM to InternLM2ArchitectureAdapter."""

    def test_factory_returns_internlm2_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(
            adapter, InternLM2ArchitectureAdapter
        ), f"Expected InternLM2ArchitectureAdapter, got {type(adapter).__name__}"

    def test_factory_key_in_supported_architectures(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "InternLM2ForCausalLM" in SUPPORTED_ARCHITECTURES
