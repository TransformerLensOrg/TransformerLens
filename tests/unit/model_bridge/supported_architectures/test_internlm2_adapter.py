"""Unit tests for InternLM2ArchitectureAdapter: weight conversions, split_wqkv, component mapping."""

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
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.internlm2 import (
    InternLM2ArchitectureAdapter,
)


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


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
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
    """Fill wqkv weight with per-kv-group (q,k,v) constants for layout verification."""
    n_kv_groups = n_heads // n_kv_heads
    gs = n_kv_groups + 2
    w = torch.zeros(n_kv_heads, gs, head_dim, d_model)
    for h, (q_val, k_val, v_val) in enumerate(kv_group_vals):
        w[h, :n_kv_groups, :, :] = q_val
        w[h, n_kv_groups, :, :] = k_val
        w[h, n_kv_groups + 1, :, :] = v_val
    wqkv_linear.weight = nn.Parameter(w.reshape((n_heads + 2 * n_kv_heads) * head_dim, d_model))


class TestInternLM2AdapterComponentMapping:
    """component_mapping has correct bridge types and InternLM2-specific names."""

    def test_embed_is_embedding_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses tok_embeddings, not embed_tokens.
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["embed"].name == "model.tok_embeddings"

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
        # InternLM2 uses 'output', not 'lm_head'.
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["unembed"].name == "output"

    def test_ln1_is_rms_normalization_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)

    def test_ln1_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses attention_norm, not input_layernorm.
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "attention_norm"

    def test_ln2_is_rms_normalization_bridge(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)

    def test_ln2_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 uses ffn_norm, not post_attention_layernorm.
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
        # InternLM2 uses 'attention', not 'self_attn'.
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
        # InternLM2 uses 'feed_forward', not 'mlp'.
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "feed_forward"

    def test_mlp_gate_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["gate"].name == "w1"

    def test_mlp_in_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["in"].name == "w3"

    def test_mlp_out_submodule_name(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["out"].name == "w2"


class TestInternLM2AdapterWeightConversions:
    """weight_processing_conversions has correct keys, types, and rearrange patterns."""

    def test_exactly_four_conversion_keys(self, adapter: InternLM2ArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert len(adapter.weight_processing_conversions) == 4

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
        # The attention bridge writes split keys; no cross-key lookup at rearrange time.
        assert adapter.weight_processing_conversions is not None
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert conv.source_key is None


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

    def test_gqa_shapes(self) -> None:
        adapter = self._adapter(n_heads=8, n_kv_heads=2, d_model=32)
        attn = _make_attn_component(8, 2, 4, 32)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert q.weight.shape == (8 * 4, 32)
        assert k.weight.shape == (2 * 4, 32)
        assert v.weight.shape == (2 * 4, 32)

    def test_mha_shapes(self) -> None:
        # MHA: n_heads == n_kv_heads → gs=3 (standard [Q|K|V]).
        adapter = self._adapter(n_heads=4, n_kv_heads=4, d_model=32)
        attn = _make_attn_component(4, 4, 8, 32)
        q, k, v = adapter._split_internlm2_wqkv(attn)
        assert q.weight.shape == (4 * 8, 32)
        assert k.weight.shape == (4 * 8, 32)
        assert v.weight.shape == (4 * 8, 32)

    def test_interleaved_layout_correctness(self) -> None:
        n_heads, n_kv_heads, head_dim, d_model = 4, 2, 4, 16
        adapter = self._adapter(n_heads=n_heads, n_kv_heads=n_kv_heads, d_model=d_model)
        attn = _make_attn_component(n_heads, n_kv_heads, head_dim, d_model)
        _fill_interleaved(
            attn.wqkv,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        q, k, v = adapter._split_internlm2_wqkv(attn)

        n_kv_groups = n_heads // n_kv_heads
        assert torch.all(q.weight[: n_kv_groups * head_dim] == 1.0)
        assert torch.all(q.weight[n_kv_groups * head_dim :] == 4.0)
        assert torch.all(k.weight[:head_dim] == 2.0)
        assert torch.all(k.weight[head_dim:] == 5.0)
        assert torch.all(v.weight[:head_dim] == 3.0)
        assert torch.all(v.weight[head_dim:] == 6.0)

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
        n_heads, n_kv_heads, head_dim, d_model = 4, 2, 4, 16
        adapter = self._adapter(n_heads=n_heads, n_kv_heads=n_kv_heads, d_model=d_model)
        attn = _make_attn_component(n_heads, n_kv_heads, head_dim, d_model, has_bias=True)
        n_kv_groups = n_heads // n_kv_heads
        gs = n_kv_groups + 2
        b = torch.zeros((n_heads + 2 * n_kv_heads) * head_dim)
        b_grouped = b.reshape(n_kv_heads, gs, head_dim)
        b_grouped[0, :n_kv_groups, :] = 1.0
        b_grouped[0, n_kv_groups, :] = 2.0
        b_grouped[0, n_kv_groups + 1, :] = 3.0
        b_grouped[1, :n_kv_groups, :] = 4.0
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


class TestInternLM2ComponentMappingPresence:
    """Component slots exist (deletion guard)."""

    def test_required_top_level_keys(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # No top-level rotary_emb (per-layer instead).
        expected = {"embed", "blocks", "ln_final", "unembed"}
        assert set(adapter.component_mapping.keys()) == expected


class TestInternLM2BlockLinearBridges:
    """All attn/mlp submodule projections are LinearBridge instances."""

    @pytest.fixture(scope="class")
    def blocks(self, adapter: InternLM2ArchitectureAdapter) -> BlockBridge:
        return adapter.component_mapping["blocks"]

    def test_attn_qkv_is_linear_bridge(self, blocks: BlockBridge) -> None:
        attn = blocks.submodules["attn"]
        assert isinstance(attn.submodules["qkv"], LinearBridge)

    def test_attn_o_is_linear_bridge(self, blocks: BlockBridge) -> None:
        attn = blocks.submodules["attn"]
        assert isinstance(attn.submodules["o"], LinearBridge)

    def test_mlp_gate_is_linear_bridge(self, blocks: BlockBridge) -> None:
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp.submodules["gate"], LinearBridge)

    def test_mlp_in_is_linear_bridge(self, blocks: BlockBridge) -> None:
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp.submodules["in"], LinearBridge)

    def test_mlp_out_is_linear_bridge(self, blocks: BlockBridge) -> None:
        mlp = blocks.submodules["mlp"]
        assert isinstance(mlp.submodules["out"], LinearBridge)


class TestInternLM2GQASupport:
    """GQA propagation through weight_processing_conversions."""

    def test_no_gqa_falls_back_to_n_heads(self) -> None:
        cfg = _make_cfg()
        cfg.n_key_value_heads = None
        adapter = InternLM2ArchitectureAdapter(cfg)
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_gqa_propagates_to_kv_conversions(self) -> None:
        cfg = _make_cfg(n_heads=8, n_key_value_heads=2)
        adapter = InternLM2ArchitectureAdapter(cfg)
        for slot in ("k", "v"):
            conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
            assert conv.tensor_conversion.axes_lengths["n"] == 2

    def test_gqa_does_not_change_q_or_o_conversions(self) -> None:
        cfg = _make_cfg(n_heads=8, n_key_value_heads=2)
        adapter = InternLM2ArchitectureAdapter(cfg)
        q_conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        o_conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert q_conv.tensor_conversion.axes_lengths["n"] == 8
        assert o_conv.tensor_conversion.axes_lengths["n"] == 8


class TestInternLM2ArchitectureGuards:
    """Guards against drift toward neighbouring adapter patterns."""

    def test_no_norm_offset_conversions(self, adapter: InternLM2ArchitectureAdapter) -> None:
        # InternLM2 is not Gemma — no +1 norm offset entries.
        for key in adapter.weight_processing_conversions:
            assert "ln1" not in key
            assert "ln2" not in key
            assert "ln_final" not in key

    def test_no_mlp_weight_conversions(self, adapter: InternLM2ArchitectureAdapter) -> None:
        for key in adapter.weight_processing_conversions:
            assert "mlp" not in key

    def test_block_uses_block_bridge_not_parallel(
        self, adapter: InternLM2ArchitectureAdapter
    ) -> None:
        # Sequential, not parallel-attn-mlp — guard against borrowing Cohere's pattern.
        from transformer_lens.model_bridge.generalized_components import (
            ParallelBlockBridge,
        )

        blocks = adapter.component_mapping["blocks"]
        assert not isinstance(blocks, ParallelBlockBridge)
        assert isinstance(blocks, BlockBridge)


def test_wqkv_split_reproduces_hf_interleaved_layout() -> None:
    """TL's split must match HF's own rearrange: wqkv packs [h, groups+2, d]
    interleaved per KV head (q rows first, then k, then v — NOT [Q|K|V]).
    Row order was never source-verified before; pinned numerically here."""
    from types import SimpleNamespace

    import torch
    from einops import rearrange

    torch.manual_seed(0)
    d_model, n_kv, groups, d_head = 32, 2, 4, 4  # n_heads = 8
    wqkv = torch.nn.Linear(d_model, n_kv * (groups + 2) * d_head, bias=False)
    x = torch.randn(2, 5, d_model)

    qkv = rearrange(wqkv(x), "b q (h gs d) -> b q h gs d", gs=groups + 2, d=d_head)
    q_hf = rearrange(qkv[..., :groups, :], "b q h gs d -> b q (h gs) d")
    k_hf = qkv[..., -2, :]
    v_hf = qkv[..., -1, :]

    cfg = _make_cfg()
    cfg.d_model, cfg.n_heads, cfg.d_head = d_model, 8, d_head
    cfg.n_key_value_heads = n_kv
    adapter = InternLM2ArchitectureAdapter(cfg)
    q_mod, k_mod, v_mod = adapter._split_internlm2_wqkv(SimpleNamespace(wqkv=wqkv))
    with torch.no_grad():
        torch.testing.assert_close(q_mod(x).view(2, 5, 8, d_head), q_hf)
        torch.testing.assert_close(k_mod(x).view(2, 5, n_kv, d_head), k_hf)
        torch.testing.assert_close(v_mod(x).view(2, 5, n_kv, d_head), v_hf)
