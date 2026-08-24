"""Unit tests for Phi3ArchitectureAdapter.

Tests cover:
- Component mapping structure (bridge types and HF module names)
- Weight conversion key set
- _SizedSplitConversion numerical correctness
- Config flags set by the adapter
- preprocess_weights LN folding
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    JointGateUpMLPBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.phi3 import (
    Phi3ArchitectureAdapter,
    _SizedSplitConversion,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

N_HEADS = 4
N_KV_HEADS = 2
D_MODEL = 64
D_HEAD = D_MODEL // N_HEADS  # 16
D_MLP = 128
N_LAYERS = 2
N_CTX = 128
D_VOCAB = 500


def _make_cfg(
    n_heads: int = N_HEADS,
    n_kv_heads: int = N_KV_HEADS,
    d_model: int = D_MODEL,
    n_layers: int = N_LAYERS,
    d_mlp: int = D_MLP,
    d_vocab: int = D_VOCAB,
    n_ctx: int = N_CTX,
    d_head: int | None = None,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Phi-3 adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_head if d_head is not None else d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        n_key_value_heads=n_kv_heads,
        default_prepend_bos=True,
        architecture="Phi3ForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> Phi3ArchitectureAdapter:
    return Phi3ArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config flag tests
# ---------------------------------------------------------------------------


class TestPhi3AdapterConfig:
    """Tests that the adapter sets the correct config flags."""

    def test_supports_fold_ln_false(self, adapter: Phi3ArchitectureAdapter) -> None:
        """Standard fold_ln is disabled — handled in preprocess_weights instead."""
        assert adapter.supports_fold_ln is False


# ---------------------------------------------------------------------------
# Component mapping tests
# ---------------------------------------------------------------------------


class TestPhi3AdapterComponentMapping:
    """Tests that component_mapping has the correct bridge types and HF module names."""

    def test_top_level_keys(self, adapter: Phi3ArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_bridge_types(self, adapter: Phi3ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(mapping["blocks"], BlockBridge)
        assert isinstance(mapping["ln_final"], RMSNormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)

    def test_top_level_hf_paths(self, adapter: Phi3ArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        assert mapping["embed"].name == "model.embed_tokens"
        assert mapping["rotary_emb"].name == "model.rotary_emb"
        assert mapping["blocks"].name == "model.layers"
        assert mapping["ln_final"].name == "model.norm"
        assert mapping["unembed"].name == "lm_head"

    def test_block_submodule_keys(self, adapter: Phi3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert set(blocks.submodules.keys()) == {"ln1", "ln2", "attn", "mlp"}

    def test_block_bridge_types(self, adapter: Phi3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["ln2"], RMSNormalizationBridge)
        assert isinstance(blocks.submodules["attn"], JointQKVPositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], JointGateUpMLPBridge)

    def test_block_hf_paths(self, adapter: Phi3ArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert blocks.submodules["attn"].name == "self_attn"
        assert blocks.submodules["mlp"].name == "mlp"

    def test_attention_submodule_keys(self, adapter: Phi3ArchitectureAdapter) -> None:
        """Phi-3 uses a fused qkv_proj with a separate o_proj."""
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert set(attn.submodules.keys()) == {"qkv", "q", "k", "v", "o"}

    def test_attention_hf_paths(self, adapter: Phi3ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert attn.submodules["qkv"].name == "qkv_proj"
        assert attn.submodules["o"].name == "o_proj"

    def test_mlp_submodule_keys(self, adapter: Phi3ArchitectureAdapter) -> None:
        """Phi-3 MLP exposes only the output projection; gate/up come from fused gate_up_proj."""
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert set(mlp.submodules.keys()) == {"gate", "in", "out"}

    def test_mlp_hf_paths(self, adapter: Phi3ArchitectureAdapter) -> None:
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert mlp.submodules["out"].name == "down_proj"

    def test_linear_submodule_bridge_types(self, adapter: Phi3ArchitectureAdapter) -> None:
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        for submodule in [*attn.submodules.values(), *mlp.submodules.values()]:
            assert isinstance(submodule, LinearBridge)


# ---------------------------------------------------------------------------
# Weight conversion key tests
# ---------------------------------------------------------------------------


class TestPhi3AdapterWeightConversions:
    """Tests that weight_processing_conversions has exactly the expected keys."""

    def test_exact_conversion_key_set(self, adapter: Phi3ArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q",
            "blocks.{i}.attn.k",
            "blocks.{i}.attn.v",
            "blocks.{i}.attn.o",
            "blocks.{i}.mlp.in",
            "blocks.{i}.mlp.gate",
        }

    def test_qkv_source_key(self, adapter: Phi3ArchitectureAdapter) -> None:
        """Q, K, V all source from the same fused qkv_proj weight."""
        for key in ["blocks.{i}.attn.q", "blocks.{i}.attn.k", "blocks.{i}.attn.v"]:
            conv = adapter.weight_processing_conversions[key]
            assert conv.source_key == "model.layers.{i}.self_attn.qkv_proj.weight"

    def test_mlp_source_key(self, adapter: Phi3ArchitectureAdapter) -> None:
        """Gate and up projections both source from fused gate_up_proj."""
        for key in ["blocks.{i}.mlp.in", "blocks.{i}.mlp.gate"]:
            conv = adapter.weight_processing_conversions[key]
            assert conv.source_key == "model.layers.{i}.mlp.gate_up_proj.weight"


# ---------------------------------------------------------------------------
# _SizedSplitConversion numerical correctness tests
# ---------------------------------------------------------------------------


class TestSizedSplitConversion:
    """Numerical correctness of Phi-3's GQA split conversion."""

    def test_extracts_q_slice(self) -> None:
        """Index 0 should return the first (Q) chunk."""
        q_size, kv_size = 8, 4
        sizes = [q_size, kv_size, kv_size]
        tensor = torch.arange(float(q_size + 2 * kv_size)).unsqueeze(1)  # [16, 1]
        conv = _SizedSplitConversion(sizes=sizes, index=0)
        out = conv.handle_conversion(tensor)
        assert out.shape[0] == q_size
        assert torch.allclose(out, tensor[:q_size])

    def test_extracts_k_slice(self) -> None:
        """Index 1 should return the second (K) chunk."""
        q_size, kv_size = 8, 4
        sizes = [q_size, kv_size, kv_size]
        tensor = torch.arange(float(q_size + 2 * kv_size)).unsqueeze(1)
        conv = _SizedSplitConversion(sizes=sizes, index=1)
        out = conv.handle_conversion(tensor)
        assert out.shape[0] == kv_size
        assert torch.allclose(out, tensor[q_size : q_size + kv_size])

    def test_extracts_v_slice(self) -> None:
        """Index 2 should return the third (V) chunk."""
        q_size, kv_size = 8, 4
        sizes = [q_size, kv_size, kv_size]
        tensor = torch.arange(float(q_size + 2 * kv_size)).unsqueeze(1)
        conv = _SizedSplitConversion(sizes=sizes, index=2)
        out = conv.handle_conversion(tensor)
        assert out.shape[0] == kv_size
        assert torch.allclose(out, tensor[q_size + kv_size :])

    def test_dim_1_split(self) -> None:
        """Splitting along dim=1 returns the correct column slice."""
        sizes = [3, 5]
        tensor = torch.ones(4, 8)
        conv = _SizedSplitConversion(sizes=sizes, index=1, dim=1)
        out = conv.handle_conversion(tensor)
        assert out.shape == (4, 5)


# ---------------------------------------------------------------------------
# preprocess_weights: LN folding
# ---------------------------------------------------------------------------


class TestPhi3PreprocessWeights:
    """Tests that preprocess_weights correctly folds RMS-norm scales."""

    def _make_state_dict(self, n_layers: int = 2, d_model: int = D_MODEL, d_mlp: int = D_MLP):
        """Build a minimal state dict matching what weight_processing would see."""
        sd = {}
        for i in range(n_layers):
            sd[f"blocks.{i}.ln1.weight"] = torch.full((d_model,), 2.0)
            sd[f"blocks.{i}.ln2.weight"] = torch.full((d_model,), 3.0)
            sd[f"blocks.{i}.attn.q.weight"] = torch.ones(N_HEADS * D_HEAD, d_model)
            sd[f"blocks.{i}.attn.k.weight"] = torch.ones(N_KV_HEADS * D_HEAD, d_model)
            sd[f"blocks.{i}.attn.v.weight"] = torch.ones(N_KV_HEADS * D_HEAD, d_model)
            sd[f"blocks.{i}.mlp.gate.weight"] = torch.ones(d_mlp, d_model)
            sd[f"blocks.{i}.mlp.in.weight"] = torch.ones(d_mlp, d_model)
        sd["ln_final.weight"] = torch.full((d_model,), 4.0)
        sd["unembed.weight"] = torch.ones(D_VOCAB, d_model)
        return sd

    def test_ln1_folded_into_qkv(self, adapter: Phi3ArchitectureAdapter) -> None:
        """ln1 scale should be multiplied into Q/K/V weights."""
        sd = self._make_state_dict()
        adapter._fold_ln_requested = True
        out = adapter.preprocess_weights(sd)
        # ln1.weight was 2.0, QKV weights were 1.0 → expect 2.0
        for key in ["blocks.0.attn.q.weight", "blocks.0.attn.k.weight", "blocks.0.attn.v.weight"]:
            assert torch.allclose(out[key], torch.full_like(out[key], 2.0)), key

    def test_ln1_set_to_ones_after_fold(self, adapter: Phi3ArchitectureAdapter) -> None:
        sd = self._make_state_dict()
        adapter._fold_ln_requested = True
        out = adapter.preprocess_weights(sd)
        assert torch.allclose(out["blocks.0.ln1.weight"], torch.ones(D_MODEL))

    def test_ln2_folded_into_mlp(self, adapter: Phi3ArchitectureAdapter) -> None:
        """ln2 scale should be multiplied into gate and up projection weights."""
        sd = self._make_state_dict()
        adapter._fold_ln_requested = True
        out = adapter.preprocess_weights(sd)
        for key in ["blocks.0.mlp.gate.weight", "blocks.0.mlp.in.weight"]:
            assert torch.allclose(out[key], torch.full_like(out[key], 3.0)), key

    def test_ln2_set_to_ones_after_fold(self, adapter: Phi3ArchitectureAdapter) -> None:
        sd = self._make_state_dict()
        adapter._fold_ln_requested = True
        out = adapter.preprocess_weights(sd)
        assert torch.allclose(out["blocks.0.ln2.weight"], torch.ones(D_MODEL))

    def test_fold_skipped_when_not_requested(self, adapter: Phi3ArchitectureAdapter) -> None:
        """When _fold_ln_requested=False the state dict is returned unchanged."""
        sd = self._make_state_dict()
        adapter._fold_ln_requested = False
        out = adapter.preprocess_weights(sd)
        assert torch.allclose(out["blocks.0.ln1.weight"], torch.full((D_MODEL,), 2.0))
        assert torch.allclose(out["blocks.0.attn.q.weight"], torch.ones(N_HEADS * D_HEAD, D_MODEL))


# ---------------------------------------------------------------------------
# Explicit head_dim: d_head decoupled from d_model // n_heads
# ---------------------------------------------------------------------------


class _FakeAttention(torch.nn.Module):
    """Minimal stand-in for HF Phi3Attention exposing only qkv_proj."""

    def __init__(self, d_model: int, qkv_rows: int, bias: bool) -> None:
        super().__init__()
        self.qkv_proj = torch.nn.Linear(d_model, qkv_rows, bias=bias)
        with torch.no_grad():
            self.qkv_proj.weight.copy_(
                torch.arange(qkv_rows * d_model, dtype=torch.float32).reshape(qkv_rows, d_model)
            )
            if bias:
                self.qkv_proj.bias.copy_(torch.arange(qkv_rows, dtype=torch.float32))


class TestPhi3ExplicitHeadDim:
    """HF configs may set head_dim explicitly, decoupled from d_model // n_heads.

    Geometry is chosen so d_model // n_heads never equals d_head — the old
    derived-d_head formula cannot accidentally produce the right split sizes.
    """

    # MHA: d_model=16, n_heads=3 → derived would be 5, explicit d_head=4
    MHA = dict(d_model=16, n_heads=3, n_kv_heads=3, d_head=4)
    # GQA: d_model=16, n_heads=4 → derived would be 4, explicit d_head=3
    GQA = dict(d_model=16, n_heads=4, n_kv_heads=2, d_head=3)

    def _split_sizes(self, adapter: Phi3ArchitectureAdapter) -> list[int]:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.q"]
        assert isinstance(conv.tensor_conversion, _SizedSplitConversion)
        return conv.tensor_conversion.sizes

    def test_mha_conversion_split_sizes(self) -> None:
        """Q/K/V each get n_heads * d_head = 12 rows, not 3 * (16 // 3) = 15."""
        adapter = Phi3ArchitectureAdapter(_make_cfg(**self.MHA))
        assert self._split_sizes(adapter) == [12, 12, 12]

    def test_gqa_conversion_split_sizes(self) -> None:
        """Q gets n_heads * d_head = 12 rows; K/V get n_kv_heads * d_head = 6."""
        adapter = Phi3ArchitectureAdapter(_make_cfg(**self.GQA))
        assert self._split_sizes(adapter) == [12, 6, 6]

    def test_conversion_split_sizes_consistent_across_qkv(self) -> None:
        """All three conversions must share the same size list."""
        adapter = Phi3ArchitectureAdapter(_make_cfg(**self.GQA))
        for key in ["blocks.{i}.attn.q", "blocks.{i}.attn.k", "blocks.{i}.attn.v"]:
            conv = adapter.weight_processing_conversions[key]
            assert isinstance(conv.tensor_conversion, _SizedSplitConversion)
            assert conv.tensor_conversion.sizes == [12, 6, 6]

    def test_split_phi3_qkv_mha_with_bias(self) -> None:
        """Live split of a fused 36-row qkv_proj into 12/12/12 with biases."""
        adapter = Phi3ArchitectureAdapter(_make_cfg(**self.MHA))
        fake = _FakeAttention(d_model=16, qkv_rows=36, bias=True)
        q, k, v = adapter._split_phi3_qkv(fake)
        assert q.weight.shape == (12, 16)
        assert k.weight.shape == (12, 16)
        assert v.weight.shape == (12, 16)
        fused_w = fake.qkv_proj.weight
        fused_b = fake.qkv_proj.bias
        assert torch.equal(q.weight, fused_w[:12])
        assert torch.equal(k.weight, fused_w[12:24])
        assert torch.equal(v.weight, fused_w[24:])
        assert torch.equal(q.bias, fused_b[:12])
        assert torch.equal(k.bias, fused_b[12:24])
        assert torch.equal(v.bias, fused_b[24:])

    def test_split_phi3_qkv_gqa_without_bias(self) -> None:
        """Live split of a fused 24-row GQA qkv_proj into 12/6/6 without biases."""
        adapter = Phi3ArchitectureAdapter(_make_cfg(**self.GQA))
        fake = _FakeAttention(d_model=16, qkv_rows=24, bias=False)
        q, k, v = adapter._split_phi3_qkv(fake)
        assert q.weight.shape == (12, 16)
        assert k.weight.shape == (6, 16)
        assert v.weight.shape == (6, 16)
        fused_w = fake.qkv_proj.weight
        assert torch.equal(q.weight, fused_w[:12])
        assert torch.equal(k.weight, fused_w[12:18])
        assert torch.equal(v.weight, fused_w[18:])
        assert q.bias is None
        assert k.bias is None
        assert v.bias is None

    def test_derived_geometry_unchanged(self) -> None:
        """Ordinary configs (d_head == d_model // n_heads) keep the same sizes."""
        adapter = Phi3ArchitectureAdapter(_make_cfg())
        assert self._split_sizes(adapter) == [
            N_HEADS * D_HEAD,
            N_KV_HEADS * D_HEAD,
            N_KV_HEADS * D_HEAD,
        ]


class TestPhi3FusedSplitRefusesQuantizedWeights:
    """The splitters Phi-3/GLM/GLM-4V actually install must refuse packed weights:
    every in-tree user overrides the guarded defaults, and FP8 slices silently
    into scale-less halves.
    """

    QUANTIZED = [
        pytest.param(torch.int8, "packed integer storage", id="int8"),
        pytest.param(torch.uint8, "packed integer storage", id="uint8"),
        pytest.param(torch.float8_e4m3fn, "narrow float", id="fp8-e4m3fn"),
    ]

    @pytest.mark.parametrize("dtype,reason", QUANTIZED)
    def test_qkv_split_refuses(self, dtype, reason) -> None:
        adapter = Phi3ArchitectureAdapter(_make_cfg(d_model=16, n_heads=3, n_kv_heads=3, d_head=4))
        fake = _FakeAttention(d_model=16, qkv_rows=36, bias=False)
        fake.qkv_proj.weight = torch.nn.Parameter(
            fake.qkv_proj.weight.detach().to(dtype), requires_grad=False
        )
        with pytest.raises(NotImplementedError, match=reason):
            adapter._split_phi3_qkv(fake)

    @pytest.mark.parametrize("dtype,reason", QUANTIZED)
    def test_gate_up_split_refuses(self, dtype, reason) -> None:
        class _FakeMLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_up_proj = torch.nn.Linear(16, 32, bias=False)
                self.gate_up_proj.weight = torch.nn.Parameter(
                    torch.zeros(32, 16, dtype=dtype), requires_grad=False
                )

        with pytest.raises(NotImplementedError, match=reason):
            Phi3ArchitectureAdapter._split_gate_up(_FakeMLP())

    def test_float_weights_still_split(self) -> None:
        """Positive control: the guard must not break ordinary Phi-3 boot."""
        adapter = Phi3ArchitectureAdapter(_make_cfg(d_model=16, n_heads=3, n_kv_heads=3, d_head=4))
        q, k, v = adapter._split_phi3_qkv(_FakeAttention(d_model=16, qkv_rows=36, bias=False))
        assert q.weight.shape == (12, 16)
