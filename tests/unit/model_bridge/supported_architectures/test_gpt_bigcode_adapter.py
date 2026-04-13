"""Unit tests for GPTBigCodeArchitectureAdapter.

Tests cover:
- Config attribute validation
- Component mapping structure (correct bridge types and HF module paths)
- Weight conversion keys
- MQAQKVConversionRule (Q and K/V branches, revert, passthrough)
- _split_qkv_matrix correctness (shapes, bias, no-bias, value correctness)
- multi_query assertion in _split_qkv_matrix
- End-to-end hook shapes with a fake MQA attention module (no downloads)
- Factory registration
"""

from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components import (
    JointQKVAttentionBridge,
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
    """Minimal GPTBigCodeAttention-like module for testing (no downloaded weights)."""

    def __init__(self, d_model: int, d_head: int, multi_query: bool = True) -> None:
        super().__init__()
        self.multi_query = multi_query
        # MQA: c_attn output = embed_dim + 2*head_dim
        out_features = d_model + 2 * d_head if multi_query else 3 * d_model
        self.c_attn = nn.Linear(d_model, out_features)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.c_proj(x)


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> GPTBigCodeArchitectureAdapter:
    return GPTBigCodeArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeAdapterConfig:
    """Verifies all required config attributes are set correctly."""

    def test_normalization_type_is_ln(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_positional_embedding_type_is_standard(
        self, adapter: GPTBigCodeArchitectureAdapter
    ) -> None:
        assert adapter.cfg.positional_embedding_type == "standard"

    def test_final_rms_is_false(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

    def test_gated_mlp_is_false(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is False

    def test_attn_only_is_false(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_n_key_value_heads_is_one(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert adapter.cfg.n_key_value_heads == 1


# ---------------------------------------------------------------------------
# Component mapping structure tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeAdapterComponentMapping:
    """Verifies component_mapping has the correct bridge types and HF paths."""

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


class TestGPTBigCodeAdapterWeightConversions:
    """Verifies weight_processing_conversions has expected keys."""

    def test_q_weight_key_present(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exactly_four_conversion_keys(self, adapter: GPTBigCodeArchitectureAdapter) -> None:
        assert len(adapter.weight_processing_conversions) == 4


# ---------------------------------------------------------------------------
# MQAQKVConversionRule tests
# ---------------------------------------------------------------------------


class TestMQAQKVConversionRule:
    """Verifies the branching QKV activation rearrangement for MQA."""

    N_HEADS = 4
    D_HEAD = 16
    D_MODEL = N_HEADS * D_HEAD  # 64
    BATCH, SEQ = 2, 8

    @pytest.fixture
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
        """4D input is already in heads format — return as-is."""
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
    """Numerical correctness tests for the MQA asymmetric QKV split."""

    N_HEADS = 4
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS  # 16
    BATCH, SEQ = 2, 8

    @pytest.fixture
    def adapter(self) -> GPTBigCodeArchitectureAdapter:
        cfg = _make_cfg(n_heads=self.N_HEADS, d_model=self.D_MODEL)
        return GPTBigCodeArchitectureAdapter(cfg)

    @pytest.fixture
    def fake_attn(self) -> FakeMQAAttention:
        return FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=True)

    @pytest.fixture
    def fake_attn_nobias(self) -> FakeMQAAttention:
        attn = FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=True)
        # Remove bias from c_attn
        attn.c_attn = nn.Linear(self.D_MODEL, self.D_MODEL + 2 * self.D_HEAD, bias=False)
        return attn

    def test_returns_three_linear_modules(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        q, k, v = adapter._split_qkv_matrix(fake_attn)
        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)

    def test_q_weight_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        q, _, _ = adapter._split_qkv_matrix(fake_attn)
        assert q.weight.shape == (self.D_MODEL, self.D_MODEL)

    def test_k_weight_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        _, k, _ = adapter._split_qkv_matrix(fake_attn)
        assert k.weight.shape == (self.D_HEAD, self.D_MODEL)

    def test_v_weight_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        _, _, v = adapter._split_qkv_matrix(fake_attn)
        assert v.weight.shape == (self.D_HEAD, self.D_MODEL)

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

    def test_q_k_v_weights_are_distinct(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        """With non-trivial c_attn weight, Q/K/V must differ."""
        nn.init.normal_(fake_attn.c_attn.weight)
        q, k, v = adapter._split_qkv_matrix(fake_attn)
        # K and V have the same shape [d_head, d_model] so compare directly
        assert not torch.allclose(k.weight, v.weight), "K and V weights must differ"

    def test_q_forward_output_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        q, _, _ = adapter._split_qkv_matrix(fake_attn)
        x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        assert q(x).shape == (self.BATCH, self.SEQ, self.D_MODEL)

    def test_k_forward_output_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        _, k, _ = adapter._split_qkv_matrix(fake_attn)
        x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        assert k(x).shape == (self.BATCH, self.SEQ, self.D_HEAD)

    def test_v_forward_output_shape(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        _, _, v = adapter._split_qkv_matrix(fake_attn)
        x = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        assert v(x).shape == (self.BATCH, self.SEQ, self.D_HEAD)

    def test_weight_values_match_c_attn_rows(
        self, adapter: GPTBigCodeArchitectureAdapter, fake_attn: FakeMQAAttention
    ) -> None:
        """Q/K/V weight rows must exactly match the corresponding rows of c_attn.weight."""
        nn.init.normal_(fake_attn.c_attn.weight)
        original_weight = fake_attn.c_attn.weight.detach()
        q, k, v = adapter._split_qkv_matrix(fake_attn)
        assert torch.equal(q.weight, original_weight[: self.D_MODEL])
        assert torch.equal(k.weight, original_weight[self.D_MODEL : self.D_MODEL + self.D_HEAD])
        assert torch.equal(v.weight, original_weight[self.D_MODEL + self.D_HEAD :])

    def test_multi_query_false_raises_assertion(
        self, adapter: GPTBigCodeArchitectureAdapter
    ) -> None:
        """Adapter must raise AssertionError for multi_query=False checkpoints."""
        mha_attn = FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=False)
        with pytest.raises(AssertionError, match="multi_query=True"):
            adapter._split_qkv_matrix(mha_attn)


# ---------------------------------------------------------------------------
# End-to-end hook shape tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeHookShapes:
    """End-to-end forward pass verifying hook_q/hook_k/hook_v shapes.

    Uses a fake MQA attention nn.Module (no model downloads). Registers explicit
    hooks on hook_out so that hook_conversion (MQAQKVConversionRule) fires and
    the captured tensors reflect the converted shapes.
    """

    N_HEADS = 4
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS  # 16
    BATCH, SEQ = 2, 8

    @pytest.fixture
    def adapter(self) -> GPTBigCodeArchitectureAdapter:
        cfg = _make_cfg(n_heads=self.N_HEADS, d_model=self.D_MODEL)
        return GPTBigCodeArchitectureAdapter(cfg)

    @pytest.fixture
    def wired_attn_bridge(
        self, adapter: GPTBigCodeArchitectureAdapter
    ) -> JointQKVAttentionBridge:
        """Return attn bridge wired to a fake MQA attention module."""
        fake_attn = FakeMQAAttention(self.D_MODEL, self.D_HEAD, multi_query=True)
        blocks = adapter.component_mapping["blocks"]
        attn_bridge: JointQKVAttentionBridge = blocks.submodules["attn"]  # type: ignore[assignment]
        attn_bridge.set_original_component(fake_attn)
        return attn_bridge

    def _run_and_capture(
        self, attn_bridge: JointQKVAttentionBridge
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Register hooks on q/k/v hook_out, run forward, return captured tensors."""
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


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestGPTBigCodeFactoryRegistration:
    """Verifies the factory maps GPTBigCodeForCausalLM to the correct adapter."""

    def test_factory_key_present(self) -> None:
        assert "GPTBigCodeForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_maps_to_correct_adapter_class(self) -> None:
        assert SUPPORTED_ARCHITECTURES["GPTBigCodeForCausalLM"] is GPTBigCodeArchitectureAdapter

    def test_factory_returns_correct_instance(self) -> None:
        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, GPTBigCodeArchitectureAdapter)
