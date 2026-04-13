"""Unit tests for CohereArchitectureAdapter — Phases A, B, C.

Covers:
- All cfg.* attributes set correctly in __init__
- logit_scale None-check behaviour
- RoPE theta extraction from rope_parameters dict
- GQA n_key_value_heads forwarded to cfg
- Factory registration (CohereForCausalLM maps to CohereArchitectureAdapter)
- weight_processing_conversions: GQA-aware Q/K/V/O rearrangements
- preprocess_weights: logit_scale folded into unembed.weight
"""

import pytest
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.cohere import (
    CohereArchitectureAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 1000,
    n_ctx: int = 512,
    n_key_value_heads: int | None = 2,
    logit_scale: float | None = 0.0625,
    rope_parameters: dict | None = None,
) -> TransformerBridgeConfig:
    """Return a minimal TransformerBridgeConfig for Cohere adapter tests."""
    cfg = TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        default_prepend_bos=True,
        architecture="CohereForCausalLM",
    )
    # Set Cohere-specific fields that sources/transformers.py would populate.
    # logit_scale and rope_parameters are not declared on TransformerBridgeConfig;
    # use setattr so mypy doesn't flag attr-defined errors in test helpers.
    if n_key_value_heads is not None:
        cfg.n_key_value_heads = n_key_value_heads
    setattr(cfg, "logit_scale", logit_scale)
    if rope_parameters is not None:
        setattr(cfg, "rope_parameters", rope_parameters)
    return cfg


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> CohereArchitectureAdapter:
    return CohereArchitectureAdapter(cfg)


# ---------------------------------------------------------------------------
# Config attribute tests
# ---------------------------------------------------------------------------


class TestCohereAdapterConfig:
    """Verify every cfg.* attribute set by CohereArchitectureAdapter.__init__."""

    def test_normalization_type_is_ln(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.cfg.normalization_type == "LN"

    def test_uses_rms_norm_is_false(self, adapter: CohereArchitectureAdapter) -> None:
        # CohereLayerNorm subtracts the mean — NOT RMSNorm
        assert adapter.cfg.uses_rms_norm is False

    def test_eps_attr_is_variance_epsilon(self, adapter: CohereArchitectureAdapter) -> None:
        # CohereLayerNorm stores epsilon as self.variance_epsilon
        assert adapter.cfg.eps_attr == "variance_epsilon"

    def test_final_rms_is_false(self, adapter: CohereArchitectureAdapter) -> None:
        # Final norm is also CohereLayerNorm, not RMSNorm
        assert adapter.cfg.final_rms is False

    def test_positional_embedding_type_is_rotary(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_gated_mlp_is_true(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.cfg.gated_mlp is True

    def test_attn_only_is_false(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.cfg.attn_only is False

    def test_parallel_attn_mlp_is_true(self, adapter: CohereArchitectureAdapter) -> None:
        # Single input_layernorm; attn and MLP run in parallel on same normed input
        assert adapter.cfg.parallel_attn_mlp is True

    def test_default_prepend_bos_is_true(self, adapter: CohereArchitectureAdapter) -> None:
        # CohereTokenizerFast prepends BOS by default via add_special_tokens=True
        assert adapter.cfg.default_prepend_bos is True

    def test_n_key_value_heads_forwarded(self, adapter: CohereArchitectureAdapter) -> None:
        # GQA: n_key_value_heads=2 from the test cfg should be on adapter.cfg
        assert adapter.cfg.n_key_value_heads == 2

    def test_logit_scale_default(self, adapter: CohereArchitectureAdapter) -> None:
        # Default logit_scale is 0.0625 (1/16)
        # logit_scale is a Cohere-specific dynamic attribute on cfg
        assert getattr(adapter.cfg, "logit_scale") == pytest.approx(0.0625)

    def test_logit_scale_is_float(self, adapter: CohereArchitectureAdapter) -> None:
        assert isinstance(getattr(adapter.cfg, "logit_scale"), float)

    def test_rotary_base_extracted(self) -> None:
        # rope_parameters dict with explicit rope_theta
        # TransformerBridgeConfig stores rotary_base as int; 80000 == 80000.0
        cfg = _make_cfg(rope_parameters={"rope_theta": 80000.0, "rope_type": "default"})
        adapter = CohereArchitectureAdapter(cfg)
        assert adapter.cfg.rotary_base == 80000

    def test_rotary_base_default_when_no_rope_parameters(self) -> None:
        # When rope_parameters is absent, fall back via default_theta or 10000.0
        # TransformerBridgeConfig stores rotary_base as int
        cfg = _make_cfg()  # no rope_parameters key set
        adapter = CohereArchitectureAdapter(cfg)
        assert isinstance(adapter.cfg.rotary_base, int)
        assert adapter.cfg.rotary_base > 0


# ---------------------------------------------------------------------------
# logit_scale None-check tests
# ---------------------------------------------------------------------------


class TestCohereLogitScaleNoneCheck:
    """Verify the explicit None-check for logit_scale (HF type is float | None)."""

    def test_none_logit_scale_falls_back_to_default(self) -> None:
        cfg = _make_cfg(logit_scale=None)
        adapter = CohereArchitectureAdapter(cfg)
        assert getattr(adapter.cfg, "logit_scale") == pytest.approx(0.0625)

    def test_explicit_logit_scale_used_when_set(self) -> None:
        cfg = _make_cfg(logit_scale=0.5)
        adapter = CohereArchitectureAdapter(cfg)
        assert getattr(adapter.cfg, "logit_scale") == pytest.approx(0.5)

    def test_logit_scale_one_preserved(self) -> None:
        cfg = _make_cfg(logit_scale=1.0)
        adapter = CohereArchitectureAdapter(cfg)
        assert getattr(adapter.cfg, "logit_scale") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestCohereFactoryRegistration:
    """Verify factory maps 'CohereForCausalLM' to CohereArchitectureAdapter."""

    def test_factory_returns_cohere_adapter(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(
            adapter, CohereArchitectureAdapter
        ), f"Expected CohereArchitectureAdapter, got {type(adapter).__name__}"

    def test_factory_key_is_cohere_for_causal_lm(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert (
            "CohereForCausalLM" in SUPPORTED_ARCHITECTURES
        ), "CohereForCausalLM must be registered in SUPPORTED_ARCHITECTURES"

    def test_factory_maps_to_correct_class(self) -> None:
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        assert SUPPORTED_ARCHITECTURES["CohereForCausalLM"] is CohereArchitectureAdapter


# ---------------------------------------------------------------------------
# Component mapping tests (Phase B)
# ---------------------------------------------------------------------------


class TestCohereAdapterComponentMapping:
    """Verify component_mapping has the correct bridge types and HF module paths.

    Plan reference: Phase B — module paths table.
    Block structure: Falcon parallel-attn pattern (ln1 only, no ln2).
    Submodule shapes: Llama-style separate Q/K/V/O and SwiGLU gate/in/out.
    """

    # -- Top-level components --

    def test_embed_is_embedding_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["embed"], EmbeddingBridge)

    def test_embed_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["embed"].name == "model.embed_tokens"

    def test_rotary_emb_is_rotary_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        # rotary_emb is top-level (not inside blocks), matching llama.py:75 / falcon.py:154
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["rotary_emb"], RotaryEmbeddingBridge)

    def test_rotary_emb_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["rotary_emb"].name == "model.rotary_emb"

    def test_rotary_emb_is_top_level_not_in_blocks(
        self, adapter: CohereArchitectureAdapter
    ) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert "rotary_emb" not in blocks.submodules

    def test_blocks_is_block_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["blocks"], BlockBridge)

    def test_blocks_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["blocks"].name == "model.layers"

    def test_ln_final_is_normalization_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["ln_final"], NormalizationBridge)

    def test_ln_final_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["ln_final"].name == "model.norm"

    def test_unembed_is_unembedding_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert isinstance(adapter.component_mapping["unembed"], UnembeddingBridge)

    def test_unembed_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        assert adapter.component_mapping["unembed"].name == "lm_head"

    # -- Block submodules --

    def test_blocks_has_ln1(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert "ln1" in blocks.submodules

    def test_blocks_ln1_is_normalization_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["ln1"], NormalizationBridge)

    def test_blocks_ln1_name(self, adapter: CohereArchitectureAdapter) -> None:
        # Cohere uses input_layernorm (same HF name as Llama, unlike Falcon's ln_attn)
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["ln1"].name == "input_layernorm"

    def test_no_ln2_in_blocks(self, adapter: CohereArchitectureAdapter) -> None:
        # Parallel block: single norm feeds both attn and MLP — no post_attention_layernorm
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert (
            "ln2" not in blocks.submodules
        ), "Cohere parallel block must NOT have ln2 (no post_attention_layernorm)"

    # -- Attention submodules --

    def test_attn_is_position_embeddings_attention_bridge(
        self, adapter: CohereArchitectureAdapter
    ) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["attn"], PositionEmbeddingsAttentionBridge)

    def test_attn_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["attn"].name == "self_attn"

    def test_attn_q_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.submodules["q"].name == "q_proj"

    def test_attn_k_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.submodules["k"].name == "k_proj"

    def test_attn_v_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.submodules["v"].name == "v_proj"

    def test_attn_o_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.submodules["o"].name == "o_proj"

    def test_attn_q_is_linear_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert isinstance(attn.submodules["q"], LinearBridge)

    def test_attn_k_is_linear_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert isinstance(attn.submodules["k"], LinearBridge)

    def test_attn_v_is_linear_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert isinstance(attn.submodules["v"], LinearBridge)

    def test_attn_o_is_linear_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        attn = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert isinstance(attn.submodules["o"], LinearBridge)

    # -- MLP submodules --

    def test_mlp_is_gated_mlp_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert isinstance(blocks.submodules["mlp"], GatedMLPBridge)

    def test_mlp_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        blocks = adapter.component_mapping["blocks"]
        assert blocks.submodules["mlp"].name == "mlp"

    def test_mlp_gate_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.submodules["gate"].name == "gate_proj"

    def test_mlp_in_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.submodules["in"].name == "up_proj"

    def test_mlp_out_name(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert mlp.submodules["out"].name == "down_proj"

    def test_mlp_gate_is_linear_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert isinstance(mlp.submodules["gate"], LinearBridge)

    def test_mlp_in_is_linear_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert isinstance(mlp.submodules["in"], LinearBridge)

    def test_mlp_out_is_linear_bridge(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        mlp = adapter.component_mapping["blocks"].submodules["mlp"]
        assert isinstance(mlp, GatedMLPBridge)
        assert isinstance(mlp.submodules["out"], LinearBridge)

    # -- Full component_mapping key set --

    def test_all_expected_top_level_keys_present(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.component_mapping is not None
        expected = {"embed", "rotary_emb", "blocks", "ln_final", "unembed"}
        actual = set(adapter.component_mapping.keys())
        assert (
            expected == actual
        ), f"Unexpected top-level keys: {actual.symmetric_difference(expected)}"


# ---------------------------------------------------------------------------
# Weight processing conversions tests (Phase C)
# ---------------------------------------------------------------------------


class TestCohereAdapterWeightConversions:
    """Verify weight_processing_conversions has the expected GQA-aware Q/K/V/O keys."""

    def test_weight_processing_conversions_not_none(
        self, adapter: CohereArchitectureAdapter
    ) -> None:
        assert adapter.weight_processing_conversions is not None

    def test_weight_processing_conversions_has_exactly_four_keys(
        self, adapter: CohereArchitectureAdapter
    ) -> None:
        assert adapter.weight_processing_conversions is not None
        assert len(adapter.weight_processing_conversions) == 4

    def test_q_weight_key_present(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.q.weight" in adapter.weight_processing_conversions

    def test_k_weight_key_present(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.k.weight" in adapter.weight_processing_conversions

    def test_v_weight_key_present(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.v.weight" in adapter.weight_processing_conversions

    def test_o_weight_key_present(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        assert "blocks.{i}.attn.o.weight" in adapter.weight_processing_conversions

    def test_exact_key_set(self, adapter: CohereArchitectureAdapter) -> None:
        assert adapter.weight_processing_conversions is not None
        expected = {
            "blocks.{i}.attn.q.weight",
            "blocks.{i}.attn.k.weight",
            "blocks.{i}.attn.v.weight",
            "blocks.{i}.attn.o.weight",
        }
        assert set(adapter.weight_processing_conversions.keys()) == expected

    def test_gqa_adapter_kv_heads_in_conversions(self) -> None:
        # GQA: K/V conversions must carry n=n_kv_heads (2), Q/O must carry n=n_heads (8).
        # Verified by inspecting RearrangeTensorConversion.axes_lengths["n"].
        from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
            RearrangeTensorConversion,
        )
        from transformer_lens.conversion_utils.param_processing_conversion import (
            ParamProcessingConversion,
        )

        cfg = _make_cfg(n_heads=8, d_model=64, n_key_value_heads=2)
        adapter = CohereArchitectureAdapter(cfg)
        assert adapter.weight_processing_conversions is not None
        assert len(adapter.weight_processing_conversions) == 4

        def _n(key: str) -> int:
            conv = adapter.weight_processing_conversions[key]  # type: ignore[index]
            assert isinstance(conv, ParamProcessingConversion)
            assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
            return int(conv.tensor_conversion.axes_lengths["n"])

        assert _n("blocks.{i}.attn.q.weight") == 8  # full n_heads
        assert _n("blocks.{i}.attn.k.weight") == 2  # n_kv_heads
        assert _n("blocks.{i}.attn.v.weight") == 2  # n_kv_heads
        assert _n("blocks.{i}.attn.o.weight") == 8  # full n_heads


# ---------------------------------------------------------------------------
# preprocess_weights tests (Phase C)
# ---------------------------------------------------------------------------


class TestCoherePreprocessWeights:
    """Verify preprocess_weights folds logit_scale into unembed.weight."""

    def _make_state_dict(self, d_model: int = 64, d_vocab: int = 1000) -> dict[str, torch.Tensor]:
        """Minimal state dict with unembed.weight and unembed.bias."""
        return {
            "embed.weight": torch.ones(d_vocab, d_model),
            "unembed.weight": torch.ones(d_vocab, d_model),
            "unembed.bias": torch.zeros(d_vocab),
        }

    def test_unembed_weight_scaled_by_logit_scale(self) -> None:
        cfg = _make_cfg(logit_scale=0.5)
        adapter = CohereArchitectureAdapter(cfg)
        sd = self._make_state_dict()
        sd = adapter.preprocess_weights(sd)
        # All values in unembed.weight should be 1.0 * 0.5 = 0.5
        assert torch.allclose(sd["unembed.weight"], torch.full_like(sd["unembed.weight"], 0.5))

    def test_unembed_bias_scaled_by_logit_scale(self) -> None:
        cfg = _make_cfg(logit_scale=0.5)
        adapter = CohereArchitectureAdapter(cfg)
        sd = self._make_state_dict()
        sd["unembed.bias"] = torch.ones(1000)
        sd = adapter.preprocess_weights(sd)
        assert torch.allclose(sd["unembed.bias"], torch.full_like(sd["unembed.bias"], 0.5))

    def test_embed_weight_unchanged_when_tied(self) -> None:
        # Simulate the tied-weight state: both keys reference the same storage.
        # bridge.py lines 726-732 clone unembed.weight before calling preprocess_weights,
        # so the fold must NOT corrupt embed.weight. This test fails if the fold is
        # ever changed to an in-place op like mul_().
        cfg = _make_cfg(logit_scale=0.0625)
        adapter = CohereArchitectureAdapter(cfg)
        shared = torch.ones(1000, 64)
        sd: dict[str, torch.Tensor] = {
            "embed.weight": shared,
            "unembed.weight": shared,  # same tensor — genuinely tied
        }
        assert sd["embed.weight"].data_ptr() == sd["unembed.weight"].data_ptr()
        adapter.preprocess_weights(sd)
        # embed.weight storage must be unscaled (all 1.0)
        assert torch.allclose(sd["embed.weight"], torch.ones_like(sd["embed.weight"]))

    def test_logit_scale_one_is_noop(self) -> None:
        cfg = _make_cfg(logit_scale=1.0)
        adapter = CohereArchitectureAdapter(cfg)
        sd = self._make_state_dict()
        original_unembed = sd["unembed.weight"].clone()
        adapter.preprocess_weights(sd)
        assert torch.allclose(sd["unembed.weight"], original_unembed)

    def test_missing_unembed_bias_no_error(self) -> None:
        # Guard: if unembed.bias is absent, no KeyError should be raised
        cfg = _make_cfg(logit_scale=0.0625)
        adapter = CohereArchitectureAdapter(cfg)
        sd = {
            "unembed.weight": torch.ones(1000, 64),
        }
        result = adapter.preprocess_weights(sd)
        assert "unembed.weight" in result
        assert torch.allclose(
            result["unembed.weight"], torch.full_like(result["unembed.weight"], 0.0625)
        )

    def test_default_logit_scale_applied(self) -> None:
        # Default logit_scale is 0.0625; verify 1.0 input becomes 0.0625
        cfg = _make_cfg(logit_scale=None)  # None triggers default 0.0625
        adapter = CohereArchitectureAdapter(cfg)
        sd = self._make_state_dict()
        sd = adapter.preprocess_weights(sd)
        expected = pytest.approx(0.0625, abs=1e-6)
        assert float(sd["unembed.weight"][0, 0].item()) == expected

    def test_dtype_preserved_float32(self) -> None:
        cfg = _make_cfg(logit_scale=0.5)
        adapter = CohereArchitectureAdapter(cfg)
        sd = {"unembed.weight": torch.ones(100, 64, dtype=torch.float32)}
        sd = adapter.preprocess_weights(sd)
        assert sd["unembed.weight"].dtype == torch.float32

    def test_dtype_preserved_float16(self) -> None:
        cfg = _make_cfg(logit_scale=0.5)
        adapter = CohereArchitectureAdapter(cfg)
        sd = {"unembed.weight": torch.ones(100, 64, dtype=torch.float16)}
        sd = adapter.preprocess_weights(sd)
        assert sd["unembed.weight"].dtype == torch.float16

    def test_returns_state_dict(self) -> None:
        cfg = _make_cfg(logit_scale=0.0625)
        adapter = CohereArchitectureAdapter(cfg)
        sd = self._make_state_dict()
        result = adapter.preprocess_weights(sd)
        assert isinstance(result, dict)
