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
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Phi-3 adapter tests."""
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
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

    def test_normalization_type(self, adapter: Phi3ArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "RMS"

    def test_positional_embedding_type(self, adapter: Phi3ArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp(self, adapter: Phi3ArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_final_rms(self, adapter: Phi3ArchitectureAdapter) -> None:
        assert adapter.cfg.final_rms is False

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

    def test_values_are_correct_not_just_shape(self) -> None:
        """Returned slice should contain the correct values, not just the right shape."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        conv = _SizedSplitConversion(sizes=[2, 2, 2], index=1)
        out = conv.handle_conversion(tensor)
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(out, expected)

    def test_three_slices_reconstruct_original(self) -> None:
        """Concatenating all three slices should recover the original tensor."""
        torch.manual_seed(42)
        q_size, kv_size = 16, 8
        sizes = [q_size, kv_size, kv_size]
        tensor = torch.randn(q_size + 2 * kv_size, 32)
        parts = [_SizedSplitConversion(sizes, i).handle_conversion(tensor) for i in range(3)]
        assert torch.allclose(torch.cat(parts, dim=0), tensor)


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
