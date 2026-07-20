"""Unit tests for GPTBigCodeArchitectureAdapter."""

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
    JointQKVAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.gpt_bigcode import (
    GPTBigCodeArchitectureAdapter,
    MQAQKVConversionRule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 100,
    n_ctx: int = 64,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for GPTBigCode adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        n_key_value_heads=1,
        default_prepend_bos=True,
        architecture="GPTBigCodeForCausalLM",
    )


class FakeMQAAttention(nn.Module):
    """Minimal GPTBigCodeAttention-like module (no downloaded weights)."""

    def __init__(self, d_model: int, d_head: int, multi_query: bool = True) -> None:
        super().__init__()
        self.multi_query = multi_query
        # MQA c_attn out = embed_dim + 2*head_dim; MHA = 3*embed_dim.
        out_features = d_model + 2 * d_head if multi_query else 3 * d_model
        self.c_attn = nn.Linear(d_model, out_features)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.c_proj(x)


@pytest.fixture(scope="class")
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture(scope="class")
def adapter(cfg: TransformerBridgeConfig) -> GPTBigCodeArchitectureAdapter:
    return GPTBigCodeArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeAdapterComponentMapping:
    def test_embed_is_embedding_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.component_mapping["embed"].name == "transformer.wte"

    def test_pos_embed_is_pos_embed_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["pos_embed"], PosEmbedBridge)

    def test_pos_embed_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.component_mapping["pos_embed"].name == "transformer.wpe"

    def test_blocks_is_block_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.component_mapping["blocks"].name == "transformer.h"

    def test_ln1_is_normalization_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_ln1_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "ln_1"

    def test_attn_is_gpt_bigcode_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], JointQKVAttentionBridge)

    def test_attn_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "attn"

    def test_attn_qkv_is_linear_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"].submodules["qkv"], LinearBridge)

    def test_attn_qkv_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].submodules["qkv"].name == "c_attn"

    def test_attn_o_is_linear_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"].submodules["o"], LinearBridge)

    def test_attn_o_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].submodules["o"].name == "c_proj"

    def test_ln2_is_normalization_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln2"], NormalizationBridge)

    def test_ln2_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln2"].name == "ln_2"

    def test_mlp_is_mlp_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_mlp_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "mlp"

    def test_mlp_in_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["in"].name == "c_fc"

    def test_mlp_out_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].submodules["out"].name == "c_proj"

    def test_ln_final_is_normalization_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.component_mapping["ln_final"].name == "transformer.ln_f"

    def test_unembed_is_unembedding_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.component_mapping["unembed"].name == "lm_head"


# ---------------------------------------------------------------------------
# Weight processing conversion tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeWeightConversionSemantics:
    """MQA pins n=1 on K/V; Q/O stay at n_heads."""

    @pytest.fixture(scope="class")
    def adapter(self) -> GPTBigCodeArchitectureAdapter:
        return GPTBigCodeArchitectureAdapter(_make_cfg())

    @pytest.mark.parametrize("slot", ["k", "v"])
    def test_kv_uses_mqa_n_equals_one(
        self, adapter: GPTBigCodeArchitectureAdapter, slot: str
    ) -> None:
        # MQA: K/V have exactly 1 KV head.
        conv = adapter.weight_processing_conversions[f"blocks.{{i}}.attn.{slot}.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"
        assert conv.tensor_conversion.axes_lengths["n"] == 1

    def test_q_conversion_type_and_pattern(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_q_n_equals_n_heads(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_o_conversion_type_and_pattern(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads


class TestGPTBigCodeCombinedQKVFlags:
    """Combined-QKV flags the loader depends on."""

    @pytest.fixture(scope="class")
    def adapter(self) -> GPTBigCodeArchitectureAdapter:
        return GPTBigCodeArchitectureAdapter(_make_cfg())


class TestGPTBigCodeArchitectureGuards:
    """Learned-pos LayerNorm arch: no rotary, no Gemma offsets."""

    @pytest.fixture(scope="class")
    def adapter(self) -> GPTBigCodeArchitectureAdapter:
        return GPTBigCodeArchitectureAdapter(_make_cfg())

    def test_no_top_level_rotary_emb(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert "rotary_emb" not in adapter.component_mapping

    def test_only_qkvo_conversion_keys(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }

    def test_ln_bridges_resolve_to_layernorm(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        """NormalizationBridges resolve to LayerNorm (mean-subtracting), not RMSNorm.

        The bridge type is shared by RMS and LN families; only uses_rms_norm
        distinguishes them. With no original_component wired, the bridge's
        uses_rms_norm property -- the value its forward reads to decide whether
        to subtract the mean -- falls through to config.uses_rms_norm.
        """
        blocks = adapter.component_mapping["blocks"]
        ln1 = blocks.submodules["ln1"]
        ln2 = blocks.submodules["ln2"]
        ln_final = adapter.component_mapping["ln_final"]
        assert isinstance(ln1, NormalizationBridge)
        assert isinstance(ln2, NormalizationBridge)
        assert isinstance(ln_final, NormalizationBridge)
        assert ln1.uses_rms_norm is False
        assert ln2.uses_rms_norm is False
        assert ln_final.uses_rms_norm is False


# ---------------------------------------------------------------------------
# MQAQKVConversionRule tests
# ---------------------------------------------------------------------------


class TestMQAQKVConversionRule:
    """Branching QKV activation rearrangement for MQA."""

    N_HEADS = 4
    D_HEAD = 16
    D_MODEL = N_HEADS * D_HEAD  # 64
    BATCH, SEQ = 2, 8

    @pytest.fixture(scope="class")
    def rule(self) -> MQAQKVConversionRule:
        return MQAQKVConversionRule(n_heads=self.N_HEADS, d_head=self.D_HEAD)

    def test_q_shaped_input_gives_n_heads_dimension(self, rule: MQAQKVConversionRule) -> None:
        """Q input [batch, seq, embed_dim] -> [batch, seq, n_heads, d_head]."""
        x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        out = rule.handle_conversion(x)
        assert out.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_kv_shaped_input_gives_one_head_dimension(self, rule: MQAQKVConversionRule) -> None:
        """K/V input [batch, seq, head_dim] -> [batch, seq, 1, d_head]."""
        x = torch.randn(self.BATCH, self.SEQ, self.D_HEAD)
        out = rule.handle_conversion(x)
        assert out.shape == (self.BATCH, self.SEQ, 1, self.D_HEAD)

    def test_4d_input_passes_through_unchanged(self, rule: MQAQKVConversionRule) -> None:
        """4D input is already in heads format; returned as-is."""
        x = torch.randn(self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)
        out = rule.handle_conversion(x)
        assert out.shape == x.shape
        assert torch.equal(out, x)

    def test_revert_q_shaped(self, rule: MQAQKVConversionRule) -> None:
        """revert undoes handle_conversion for Q-shaped input."""
        x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        converted = rule.handle_conversion(x)
        reverted = rule.revert(converted)
        assert reverted.shape == x.shape
        assert torch.allclose(reverted, x)

    def test_revert_kv_shaped(self, rule: MQAQKVConversionRule) -> None:
        """revert undoes handle_conversion for K/V-shaped input."""
        x = torch.randn(self.BATCH, self.SEQ, self.D_HEAD)
        converted = rule.handle_conversion(x)
        reverted = rule.revert(converted)
        assert reverted.shape == x.shape
        assert torch.allclose(reverted, x)

    def test_revert_3d_passes_through(self, rule: MQAQKVConversionRule) -> None:
        """revert on a 3D tensor (already flat) is a no-op."""
        x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        out = rule.revert(x)
        assert torch.equal(out, x)

    def test_invalid_ndim_raises(self, rule: MQAQKVConversionRule) -> None:
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            rule.handle_conversion(torch.randn(self.D_MODEL))


# ---------------------------------------------------------------------------
# _split_qkv_matrix tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeMQASplitQKVMatrix:
    """Numerical tests for the MQA asymmetric QKV split."""

    N_HEADS = 4
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS  # 16
    BATCH, SEQ = 2, 8

    @pytest.fixture(scope="class")
    def adapter(self) -> GPTBigCodeArchitectureAdapter:
        cfg = _make_cfg(n_heads=self.N_HEADS, d_model=self.D_MODEL)
        return GPTBigCodeArchitectureAdapter(cfg)

    @pytest.fixture(scope="class")
    def fake_attn(self) -> FakeMQAAttention:
        return FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=True)

    @pytest.fixture(scope="class")
    def fake_attn_nobias(self) -> FakeMQAAttention:
        attn = FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=True)
        attn.c_attn = nn.Linear(self.D_MODEL, self.D_MODEL + 2 * self.D_HEAD, bias=False)
        return attn

    def test_returns_three_linear_modules(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        q, k, v = adapter._split_qkv_matrix(fake_attn)
        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)

    def test_q_bias_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        q, _, _ = adapter._split_qkv_matrix(fake_attn)
        assert q.bias is not None
        assert q.bias.shape == (self.D_MODEL,)

    def test_k_bias_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        _, k, _ = adapter._split_qkv_matrix(fake_attn)
        assert k.bias is not None
        assert k.bias.shape == (self.D_HEAD,)

    def test_v_bias_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        _, _, v = adapter._split_qkv_matrix(fake_attn)
        assert v.bias is not None
        assert v.bias.shape == (self.D_HEAD,)

    def test_no_bias_case_all_none(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn_nobias: FakeMQAAttention
    ) -> None:
        q, k, v = adapter._split_qkv_matrix(fake_attn_nobias)
        assert q.bias is None
        assert k.bias is None
        assert v.bias is None

    def test_weight_values_match_c_attn_rows(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        """Q/K/V rows match the corresponding c_attn.weight rows exactly."""
        nn.init.normal_(fake_attn.c_attn.weight)
        original_weight = fake_attn.c_attn.weight.detach()
        q, k, v = adapter._split_qkv_matrix(fake_attn)
        assert torch.equal(q.weight, original_weight[: self.D_MODEL])
        assert torch.equal(k.weight, original_weight[self.D_MODEL : self.D_MODEL + self.D_HEAD])
        assert torch.equal(v.weight, original_weight[self.D_MODEL + self.D_HEAD :])

    def test_multi_query_false_raises_assertion(
        self, adapter: GPTBigCodeArchitectureAdapter
    ) -> None:
        """multi_query=False checkpoints must raise (adapter is MQA-only)."""
        mha_attn = FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=False)
        with pytest.raises(AssertionError, match="multi_query=True"):
            adapter._split_qkv_matrix(mha_attn)


# ---------------------------------------------------------------------------
# End-to-end hook shape tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeHookShapes:
    """End-to-end hook_q/hook_k/hook_v shapes via a fake MQA attention.

    Explicit hooks on hook_out trigger hook_conversion (MQAQKVConversionRule)
    so captured tensors reflect the converted shapes.
    """

    N_HEADS = 4
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS  # 16
    BATCH, SEQ = 2, 8

    @pytest.fixture(scope="class")
    def adapter(self) -> GPTBigCodeArchitectureAdapter:
        cfg = _make_cfg(n_heads=self.N_HEADS, d_model=self.D_MODEL)
        return GPTBigCodeArchitectureAdapter(cfg)

    @pytest.fixture(scope="class")
    def wired_attn_bridge(self, adapter: GPTBigCodeArchitectureAdapter) -> JointQKVAttentionBridge:
        """Attn bridge wired to a fake MQA attention module."""
        fake_attn = FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=True)
        blocks = adapter.component_mapping["blocks"]
        attn_bridge: JointQKVAttentionBridge = blocks.submodules["attn"]  # type: ignore[assignment]
        attn_bridge.set_original_component(fake_attn)
        return attn_bridge

    def _run_and_capture(
        self, attn_bridge: JointQKVAttentionBridge
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hook q/k/v hook_out, run forward, return captured tensors."""
        captured: dict[str, torch.Tensor] = {}

        def _capture(name: str) -> Any:
            def _hook(x: torch.Tensor, hook: Any) -> torch.Tensor:
                captured[name] = x
                return x

            return _hook

        attn_bridge.q.hook_out.add_hook(_capture("q"))
        attn_bridge.k.hook_out.add_hook(_capture("k"))
        attn_bridge.v.hook_out.add_hook(_capture("v"))

        hidden = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        attn_bridge(hidden)

        return captured["q"], captured["k"], captured["v"]

    def test_hook_q_shape(self, wired_attn_bridge: JointQKVAttentionBridge) -> None:
        """hook_q must be [batch, seq, n_heads, d_head]."""
        q, _, _ = self._run_and_capture(wired_attn_bridge)
        assert q.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_hook_k_shape(self, wired_attn_bridge: JointQKVAttentionBridge) -> None:
        """hook_k must be [batch, seq, 1, d_head] (1 KV head)."""
        _, k, _ = self._run_and_capture(wired_attn_bridge)
        assert k.shape == (self.BATCH, self.SEQ, 1, self.D_HEAD)

    def test_hook_v_shape(self, wired_attn_bridge: JointQKVAttentionBridge) -> None:
        """hook_v must be [batch, seq, 1, d_head] (1 KV head)."""
        _, _, v = self._run_and_capture(wired_attn_bridge)
        assert v.shape == (self.BATCH, self.SEQ, 1, self.D_HEAD)

    def test_attn_output_shape(self, wired_attn_bridge: JointQKVAttentionBridge) -> None:
        """Full attention output must be [batch, seq, d_model]."""
        hidden = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        out = wired_attn_bridge(hidden)
        out_tensor = out[0] if isinstance(out, tuple) else out
        assert out_tensor.shape == (self.BATCH, self.SEQ, self.D_MODEL)
