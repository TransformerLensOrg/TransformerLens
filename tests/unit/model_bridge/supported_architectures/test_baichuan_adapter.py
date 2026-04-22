"""Unit tests for BaichuanArchitectureAdapter.

Tests cover:
- Config attributes
- Component mapping structure and HF module names
- Weight conversion keys/types
- split_qkv_matrix (W_pack) numerical correctness
- preprocess_weights (QKV split, fold_ln, NormHead normalization)
- Factory registration (both v1 and v2 class names)
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
from transformer_lens.model_bridge.supported_architectures.baichuan import (
    BaichuanArchitectureAdapter,
    _BaichuanAttentionBridge,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 32,
    d_model: int = 64,
    n_layers: int = 2,
    d_vocab: int = 100,
    n_ctx: int = 128,
) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig for Baichuan adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        default_prepend_bos=True,
        architecture="BaichuanForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg(n_heads=8, d_model=64)


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> BaichuanArchitectureAdapter:
    return BaichuanArchitectureAdapter(cfg)


def _make_w_pack_component(d_model: int) -> Any:
    """Synthetic attention namespace with W_pack linear."""
    ns = SimpleNamespace()
    ns.W_pack = nn.Linear(d_model, 3 * d_model, bias=False)
    return ns


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestBaichuanAdapterConfig:
    def test_normalization_type(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_final_rms(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is True

    def test_gated_mlp(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_uses_rms_norm(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.cfg.uses_rms_norm is True

    def test_eps_attr(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.cfg.eps_attr == "variance_epsilon"

    def test_supports_fold_ln_false(self, adapter: BaichuanArchitectureAdapter) -> None:
        assert adapter.supports_fold_ln is False


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestBaichuanAdapterComponentMapping:
    @staticmethod
    def _mapping(adapter: BaichuanArchitectureAdapter) -> dict[str, Any]:
        mapping = adapter.component_mapping
        assert mapping is not None
        return mapping

    def test_embed_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert mapping["embed"].name == "model.embed_tokens"

    def test_no_top_level_rotary_emb(self, adapter: BaichuanArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert "rotary_emb" not in mapping

    def test_blocks_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert mapping["blocks"].name == "model.layers"

    def test_ln_final_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert mapping["ln_final"].name == "model.norm"

    def test_unembed_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        mapping = self._mapping(adapter)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert mapping["unembed"].name == "lm_head"

    def test_ln1_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_ln2_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"

    def test_attn_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["attn"], JointQKVPositionEmbeddingsAttentionBridge)
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_qkv_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["attn"].submodules["qkv"].name == "W_pack"

    def test_attn_o_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["attn"].submodules["o"].name == "o_proj"

    def test_mlp_type_and_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)
        assert blocks.submodules["mlp"].name == "mlp"

    def test_mlp_gate_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["mlp"].submodules["gate"].name == "gate_proj"

    def test_mlp_in_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["mlp"].submodules["in"].name == "up_proj"

    def test_mlp_out_name(self, adapter: BaichuanArchitectureAdapter) -> None:
        blocks = self._mapping(adapter)["blocks"]
        assert blocks.submodules["mlp"].submodules["out"].name == "down_proj"


# ---------------------------------------------------------------------------
# Weight conversion tests
# ---------------------------------------------------------------------------


class TestBaichuanAdapterWeightConversions:
    def test_four_conversion_keys(self, adapter: BaichuanArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        assert len(convs) == 4

    def test_qkvo_keys_present(self, adapter: BaichuanArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        for key in [
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        ]:
            assert key in convs

    def test_q_conversion_type(self, adapter: BaichuanArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)

    def test_q_rearrange_pattern(self, adapter: BaichuanArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "(n h) m -> n m h"

    def test_q_rearrange_n_equals_n_heads(self, adapter: BaichuanArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_k_rearrange_n_equals_n_heads(self, adapter: BaichuanArchitectureAdapter) -> None:
        # Baichuan is MHA (no GQA), so K also uses n_heads
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.k.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.axes_lengths["n"] == adapter.cfg.n_heads

    def test_o_rearrange_pattern(self, adapter: BaichuanArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.o.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "m (n h) -> n h m"

    def test_no_source_key_on_q(self, adapter: BaichuanArchitectureAdapter) -> None:
        convs = adapter.weight_processing_conversions
        assert convs is not None
        conv = convs["blocks.{i}.attn.q.weight"]
        assert isinstance(conv, ParamProcessingConversion)
        assert conv.source_key is None


# ---------------------------------------------------------------------------
# split_qkv_matrix (W_pack) tests
# ---------------------------------------------------------------------------


class TestBaichuanSplitWPack:
    def _adapter(self, n_heads: int = 8, d_model: int = 64) -> BaichuanArchitectureAdapter:
        return BaichuanArchitectureAdapter(_make_cfg(n_heads=n_heads, d_model=d_model))

    def test_returns_three_linears(self) -> None:
        adapter = self._adapter()
        attn = _make_w_pack_component(64)
        q, k, v = adapter._split_baichuan_w_pack(attn)
        assert isinstance(q, nn.Linear)
        assert isinstance(k, nn.Linear)
        assert isinstance(v, nn.Linear)

    def test_output_shapes(self) -> None:
        d_model = 64
        adapter = self._adapter(d_model=d_model)
        attn = _make_w_pack_component(d_model)
        q, k, v = adapter._split_baichuan_w_pack(attn)
        assert q.weight.shape == (d_model, d_model)
        assert k.weight.shape == (d_model, d_model)
        assert v.weight.shape == (d_model, d_model)

    def test_no_bias(self) -> None:
        adapter = self._adapter()
        attn = _make_w_pack_component(64)
        q, k, v = adapter._split_baichuan_w_pack(attn)
        assert q.bias is None
        assert k.bias is None
        assert v.bias is None

    def test_concatenated_split_correctness(self) -> None:
        """W_pack = [Q|K|V] concatenated — verify split recovers each part."""
        d_model = 32
        adapter = self._adapter(n_heads=4, d_model=d_model)
        attn = _make_w_pack_component(d_model)
        # Fill W_pack: Q=1.0, K=2.0, V=3.0
        w = torch.zeros(3 * d_model, d_model)
        w[:d_model, :] = 1.0
        w[d_model : 2 * d_model, :] = 2.0
        w[2 * d_model :, :] = 3.0
        attn.W_pack.weight = nn.Parameter(w)

        q, k, v = adapter._split_baichuan_w_pack(attn)
        assert torch.all(q.weight == 1.0), "Q should be 1.0"
        assert torch.all(k.weight == 2.0), "K should be 2.0"
        assert torch.all(v.weight == 3.0), "V should be 3.0"

    def test_round_trip_recombine(self) -> None:
        """Split → recombine must equal original W_pack weights."""
        d_model = 64
        adapter = self._adapter(d_model=d_model)
        attn = _make_w_pack_component(d_model)
        original_w = attn.W_pack.weight.data.clone()

        q, k, v = adapter._split_baichuan_w_pack(attn)
        recombined = torch.cat([q.weight.data, k.weight.data, v.weight.data], dim=0)
        assert torch.equal(recombined, original_w)

    def test_forward_output_shapes(self) -> None:
        d_model = 64
        adapter = self._adapter(d_model=d_model)
        attn = _make_w_pack_component(d_model)
        q, k, v = adapter._split_baichuan_w_pack(attn)
        x = torch.randn(2, 5, d_model)
        assert q(x).shape == (2, 5, d_model)
        assert k(x).shape == (2, 5, d_model)
        assert v(x).shape == (2, 5, d_model)


# ---------------------------------------------------------------------------
# preprocess_weights tests
# ---------------------------------------------------------------------------


class TestBaichuanPreprocessWeights:
    def _make_state_dict(
        self,
        adapter: BaichuanArchitectureAdapter,
        d_model: int = 64,
        n_layers: int = 2,
        d_mlp: int = 16,
        d_vocab: int = 100,
        ln1_scale: float = 1.0,
        qkv_val: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Bridge-format state dict with fused W_pack for each layer."""
        state: dict[str, torch.Tensor] = {}
        for i in range(n_layers):
            state[f"blocks.{i}.attn.qkv.weight"] = torch.full((3 * d_model, d_model), qkv_val)
            state[f"blocks.{i}.ln1.weight"] = torch.full((d_model,), ln1_scale)
            state[f"blocks.{i}.ln2.weight"] = torch.ones(d_model)
            state[f"blocks.{i}.mlp.gate.weight"] = torch.ones(d_mlp, d_model)
            state[f"blocks.{i}.mlp.in.weight"] = torch.ones(d_mlp, d_model)
            state[f"blocks.{i}.attn.o.weight"] = torch.ones(d_model, d_model)
        state["ln_final.weight"] = torch.ones(d_model)
        state["unembed.weight"] = torch.ones(d_vocab, d_model)
        return state

    def test_fused_key_removed_and_split_keys_written(self) -> None:
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=64))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter)
        result = adapter.preprocess_weights(sd)
        assert "blocks.0.attn.qkv.weight" not in result
        assert "blocks.0.attn.q.weight" in result
        assert "blocks.0.attn.k.weight" in result
        assert "blocks.0.attn.v.weight" in result

    def test_split_shapes(self) -> None:
        d_model = 64
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=d_model))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter, d_model=d_model)
        result = adapter.preprocess_weights(sd)
        # Baichuan is MHA: Q, K, V each have shape [d_model, d_model]
        assert result["blocks.0.attn.q.weight"].shape == (d_model, d_model)
        assert result["blocks.0.attn.k.weight"].shape == (d_model, d_model)
        assert result["blocks.0.attn.v.weight"].shape == (d_model, d_model)

    def test_ln1_fold_applied(self) -> None:
        d_model = 64
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=d_model))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter, d_model=d_model, ln1_scale=2.0, qkv_val=1.0)
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["blocks.0.attn.q.weight"] == 2.0)
        assert torch.all(result["blocks.0.attn.k.weight"] == 2.0)
        assert torch.all(result["blocks.0.attn.v.weight"] == 2.0)

    def test_ln1_reset_to_ones(self) -> None:
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=64))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter, ln1_scale=3.0)
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["blocks.0.ln1.weight"] == 1.0)

    def test_ln2_fold_applied(self) -> None:
        d_model = 64
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=d_model))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter, d_model=d_model)
        sd["blocks.0.ln2.weight"] = torch.full((d_model,), 3.0)
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["blocks.0.mlp.gate.weight"] == 3.0)
        assert torch.all(result["blocks.0.mlp.in.weight"] == 3.0)

    def test_no_fold_still_splits_qkv(self) -> None:
        """Without fold_ln, W_pack must still be split for weight conversions."""
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=64))
        adapter._fold_ln_requested = False
        sd = self._make_state_dict(adapter)
        result = adapter.preprocess_weights(sd)
        assert "blocks.0.attn.qkv.weight" not in result
        assert "blocks.0.attn.q.weight" in result
        assert "blocks.0.attn.k.weight" in result
        assert "blocks.0.attn.v.weight" in result

    def test_ln_final_fold_values(self) -> None:
        """ln_final fold multiplies unembed weights by ln_final scale."""
        d_model = 64
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=d_model))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter, d_model=d_model)
        sd["ln_final.weight"] = torch.full((d_model,), 2.0)
        sd["unembed.weight"] = torch.ones(100, d_model)
        result = adapter.preprocess_weights(sd)
        assert torch.all(result["unembed.weight"] == 2.0)
        assert torch.all(result["ln_final.weight"] == 1.0)

    def test_dtype_preserved(self) -> None:
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=64))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter)
        sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}
        result = adapter.preprocess_weights(sd)
        assert result["blocks.0.attn.q.weight"].dtype == torch.bfloat16

    def test_all_layers_processed(self) -> None:
        adapter = BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=64, n_layers=3))
        adapter._fold_ln_requested = True
        sd = self._make_state_dict(adapter, n_layers=3)
        result = adapter.preprocess_weights(sd)
        for i in range(3):
            assert f"blocks.{i}.attn.qkv.weight" not in result
            assert f"blocks.{i}.attn.q.weight" in result


# ---------------------------------------------------------------------------
# prepare_model tests (NormHead normalization)
# ---------------------------------------------------------------------------


class TestBaichuanPrepareModel:
    def _adapter(self) -> BaichuanArchitectureAdapter:
        return BaichuanArchitectureAdapter(_make_cfg(n_heads=8, d_model=64))

    def test_normhead_weights_normalized(self) -> None:
        """NormHead (has first_flag) should have row-normalized weights after prepare_model."""
        adapter = self._adapter()
        lm_head = SimpleNamespace(
            weight=nn.Parameter(torch.full((100, 64), 2.0)),
            first_flag=True,
        )
        hf_model = SimpleNamespace(lm_head=lm_head)
        adapter.prepare_model(hf_model)
        row_norms = lm_head.weight.data.float().norm(dim=-1)
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)

    def test_regular_linear_unchanged(self) -> None:
        """nn.Linear lm_head (no first_flag) should not be modified."""
        adapter = self._adapter()
        lm_head = nn.Linear(64, 100, bias=False)
        original_w = lm_head.weight.data.clone()
        hf_model = SimpleNamespace(lm_head=lm_head)
        adapter.prepare_model(hf_model)
        assert torch.equal(lm_head.weight.data, original_w)

    def test_no_lm_head_is_noop(self) -> None:
        """Model without lm_head should not raise."""
        adapter = self._adapter()
        hf_model = SimpleNamespace()
        adapter.prepare_model(hf_model)  # should not raise

    def test_recomputes_rotary_from_scratch_when_inv_freq_is_meta(self) -> None:
        """Baichuan2's inv_freq/cos_cached are plain attrs that land on meta under
        HF v5 meta-init; prepare_model must recompute real values regardless."""
        adapter = self._adapter()
        head_dim = adapter.cfg.d_model // adapter.cfg.n_heads
        # Meta-device rotary matching v2's plain-attribute shape
        rotary = SimpleNamespace(
            inv_freq=torch.empty(head_dim // 2, device="meta"),
            cos_cached=torch.empty(1, 1, 16, head_dim, device="meta"),
            sin_cached=torch.empty(1, 1, 16, head_dim, device="meta"),
            max_seq_len_cached=16,
        )
        layer = SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=rotary))
        hf_model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

        adapter.prepare_model(hf_model)

        assert rotary.inv_freq.device.type == "cpu"
        assert rotary.cos_cached.device.type == "cpu"
        assert rotary.sin_cached.device.type == "cpu"
        assert rotary.cos_cached.shape == (1, 1, 16, head_dim)
        # Sanity: cos(0) == 1 and position 0 of each head_dim element equals 1.
        assert torch.allclose(
            rotary.cos_cached[0, 0, 0, :],
            torch.ones(head_dim),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestBaichuanFactoryRegistration:
    def test_factory_v2_key(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "BaichuanForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_v1_key(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert "BaiChuanForCausalLM" in SUPPORTED_ARCHITECTURES

    def test_factory_v2_returns_baichuan_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg(n_heads=8, d_model=64)
        cfg.architecture = "BaichuanForCausalLM"
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, BaichuanArchitectureAdapter)

    def test_factory_v1_returns_baichuan_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg(n_heads=8, d_model=64)
        cfg.architecture = "BaiChuanForCausalLM"
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, BaichuanArchitectureAdapter)

    def test_import_from_init(self) -> None:
        from transformer_lens.model_bridge.supported_architectures import (
            BaichuanArchitectureAdapter as FromInit,
        )

        assert FromInit is BaichuanArchitectureAdapter


# ---------------------------------------------------------------------------
# Attention bridge: position_ids → position_embeddings conversion
# ---------------------------------------------------------------------------


class _FakeRotary(nn.Module):
    """Minimal stand-in for Baichuan's RotaryEmbedding (returns 4D cached cos/sin)."""

    def __init__(self, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_seq_len_cached = max_seq_len
        # Fill with position-dependent values so tests can verify indexing.
        cos = (
            torch.arange(max_seq_len, dtype=torch.float32)[:, None]
            .expand(max_seq_len, head_dim)
            .clone()
        )
        sin = -cos
        self.register_buffer("cos_cached", cos[None, None, :, :])
        self.register_buffer("sin_cached", sin[None, None, :, :])
        self.calls: list[int] = []

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(seq_len)
        cos_cached = self.cos_cached
        sin_cached = self.sin_cached
        assert isinstance(cos_cached, torch.Tensor)
        assert isinstance(sin_cached, torch.Tensor)
        return (
            cos_cached[:, :, :seq_len, :].to(dtype=x.dtype),
            sin_cached[:, :, :seq_len, :].to(dtype=x.dtype),
        )


class _FakeAttention(nn.Module):
    """nn.Module container that exposes a `rotary_emb` + `o_proj` to the bridge."""

    def __init__(self, rotary: _FakeRotary, d_model: int) -> None:
        super().__init__()
        self.rotary_emb = rotary
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.o_proj.weight)


def _make_attention_bridge(cfg: TransformerBridgeConfig) -> _BaichuanAttentionBridge:
    from transformer_lens.model_bridge.generalized_components import LinearBridge

    return _BaichuanAttentionBridge(
        name="self_attn",
        config=cfg,
        split_qkv_matrix=lambda _c: (
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
        ),
        submodules={
            "qkv": LinearBridge(name="W_pack"),
            "o": LinearBridge(name="o_proj"),
        },
    )


def _wire_bridge(
    cfg: TransformerBridgeConfig,
) -> tuple[_BaichuanAttentionBridge, _FakeRotary, int]:
    """Build a bridge with a fake HF attention (rotary + o_proj) attached."""
    head_dim = cfg.d_model // cfg.n_heads
    bridge = _make_attention_bridge(cfg)
    rotary = _FakeRotary(head_dim=head_dim, max_seq_len=32)
    fake_attn = _FakeAttention(rotary, cfg.d_model)
    bridge.set_original_component(fake_attn)
    # `o` LinearBridge is normally wired by setup_components via component_mapping;
    # wire it directly for unit tests that construct the bridge standalone.
    bridge.o.set_original_component(fake_attn.o_proj)
    return bridge, rotary, head_dim


class TestBaichuanAttentionBridgeRotary:
    """Regression tests for the attention bridge's rotary + KV-cache contract."""

    def test_uses_position_ids_when_position_embeddings_absent(
        self, cfg: TransformerBridgeConfig
    ) -> None:
        bridge, rotary, head_dim = _wire_bridge(cfg)

        batch, seq = 1, 4
        q = torch.zeros(batch, seq, cfg.d_model)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        position_ids = torch.tensor([[0, 1, 2, 3]])

        attn_output, _, present = bridge._reconstruct_attention(
            q, k, v, position_ids=position_ids, use_cache=True
        )

        # rotary_emb called once, with kv_seq_len=seq (no past)
        assert rotary.calls == [seq]
        assert attn_output.shape == (batch, seq, cfg.d_model)
        assert present is not None
        present_k, present_v = present
        assert present_k.shape == (batch, cfg.n_heads, seq, head_dim)
        assert present_v.shape == (batch, cfg.n_heads, seq, head_dim)

    def test_preserves_explicit_position_embeddings(self, cfg: TransformerBridgeConfig) -> None:
        bridge, rotary, head_dim = _wire_bridge(cfg)

        batch, seq = 1, 4
        q = torch.zeros(batch, seq, cfg.d_model)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        explicit = (
            torch.ones(batch, seq, head_dim) * 7,
            torch.ones(batch, seq, head_dim) * 9,
        )

        bridge._reconstruct_attention(
            q,
            k,
            v,
            position_embeddings=explicit,
            position_ids=torch.tensor([[0, 1, 2, 3]]),
            use_cache=True,
        )
        # Caller-supplied embeddings must win; rotary_emb must not be called.
        assert rotary.calls == []

    def test_use_cache_false_returns_none_present(self, cfg: TransformerBridgeConfig) -> None:
        bridge, _, _ = _wire_bridge(cfg)
        q = torch.zeros(1, 4, cfg.d_model)
        _, _, present = bridge._reconstruct_attention(
            q, q.clone(), q.clone(), position_ids=torch.tensor([[0, 1, 2, 3]])
        )
        assert present is None

    def test_concats_past_key_value_along_seq_dim(self, cfg: TransformerBridgeConfig) -> None:
        """With past cache of length P and current seq S, the present cache's
        k/v have seq dim P+S and rotary is requested with kv_seq_len=P+S."""
        bridge, rotary, head_dim = _wire_bridge(cfg)

        batch, past_len, seq = 1, 3, 2
        past_k = torch.randn(batch, cfg.n_heads, past_len, head_dim)
        past_v = torch.randn(batch, cfg.n_heads, past_len, head_dim)

        q = torch.zeros(batch, seq, cfg.d_model)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        # HF's Model.forward generates position_ids offset by past_len.
        position_ids = torch.tensor([[past_len, past_len + 1]])

        _, _, present = bridge._reconstruct_attention(
            q,
            k,
            v,
            past_key_value=(past_k, past_v),
            position_ids=position_ids,
            use_cache=True,
        )
        assert rotary.calls == [past_len + seq]
        assert present is not None
        present_k, present_v = present
        assert present_k.shape == (batch, cfg.n_heads, past_len + seq, head_dim)
        assert present_v.shape == (batch, cfg.n_heads, past_len + seq, head_dim)
        # First past_len slots must be the provided past, unchanged.
        assert torch.equal(present_k[:, :, :past_len, :], past_k)
        assert torch.equal(present_v[:, :, :past_len, :], past_v)


# ---------------------------------------------------------------------------
# prepare_loading: bitsandbytes preflight
# ---------------------------------------------------------------------------


class TestBaichuanPrepareLoadingBitsandbytes:
    """The adapter must point users at `uv sync --group quantization` when bnb is missing."""

    def test_preflight_raises_clean_import_error(
        self, adapter: BaichuanArchitectureAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import transformer_lens.model_bridge.supported_architectures.baichuan as baichuan_mod

        # Force the preflight path: make find_spec report bitsandbytes missing,
        # and make get_class_from_dynamic_module surface the transformers-style
        # "requires the following packages... bitsandbytes" error.
        monkeypatch.setattr(baichuan_mod.importlib.util, "find_spec", lambda name: None)

        def _raise_bnb(*_a: Any, **_k: Any) -> None:
            raise ImportError(
                "This modeling file requires the following packages that were "
                "not found in your environment: bitsandbytes"
            )

        import transformers.dynamic_module_utils as dmu

        monkeypatch.setattr(dmu, "get_class_from_dynamic_module", _raise_bnb)

        with pytest.raises(ImportError, match="uv sync --group quantization"):
            adapter.prepare_loading("baichuan-inc/Baichuan2-7B-Chat", {})

    def test_preflight_no_false_positive_when_bnb_installed(
        self, adapter: BaichuanArchitectureAdapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If bnb IS installed, the transformers error won't mention bnb, so no raise."""
        import transformers.dynamic_module_utils as dmu

        def _raise_generic(*_a: Any, **_k: Any) -> None:
            raise ValueError("some unrelated loader failure")

        monkeypatch.setattr(dmu, "get_class_from_dynamic_module", _raise_generic)
        # Must not raise — the generic failure path is swallowed (remote load
        # may legitimately fail for offline tests, e.g. no network access).
        adapter.prepare_loading("baichuan-inc/Baichuan2-7B-Chat", {})
