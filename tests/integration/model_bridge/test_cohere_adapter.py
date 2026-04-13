"""Integration tests for Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, ~8M params, CPU-safe, no gating required
  - tie_word_embeddings=True by default
  - logit_scale=0.0625 (1/16)

NOTE: The tiny model has use_qk_norm=False, so QK-norm is not exercised here.
Cohere's QK-norm is a per-head LayerNorm inside CohereAttention.forward; it is
handled via HF delegation (PositionEmbeddingsAttentionBridge calls the original
CohereAttention.forward directly), so functional correctness for that path relies
on the same delegation mechanism verified in test_forward_matches_hf.
"""

from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"


@pytest.fixture(scope="module")
def cohere_bridge():
    """Load tiny Cohere bridge once per module (no weight processing)."""
    return TransformerBridge.boot_transformers(MODEL, device="cpu")


@pytest.fixture(scope="module")
def cohere_bridge_processed():
    """Bridge with preprocess_weights applied (fold only, no centering).

    process_weights must be called explicitly — boot_transformers does not call
    it automatically. We disable all ProcessWeights options so only the adapter's
    preprocess_weights (logit_scale fold + untie) runs.
    """
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu")
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def cohere_hf() -> Any:
    """Load the raw HF model for side-by-side comparisons."""
    return AutoModelForCausalLM.from_pretrained(MODEL).eval()


# ---------------------------------------------------------------------------
# 1. Bridge creation — exercises all 4 registration points + component_mapping
# ---------------------------------------------------------------------------


class TestCohereBridgeCreation:
    """Verify the bridge loads cleanly and exposes the expected structure."""

    def test_boot_transformers_succeeds(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge is not None

    def test_block_count(self, cohere_bridge: TransformerBridge) -> None:
        # tiny model has 2 layers
        assert len(cohere_bridge.blocks) == 2

    def test_parallel_attn_mlp_flag(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.cfg.parallel_attn_mlp is True

    def test_has_core_components(self, cohere_bridge: TransformerBridge) -> None:
        assert hasattr(cohere_bridge, "embed")
        assert hasattr(cohere_bridge, "unembed")
        assert hasattr(cohere_bridge, "ln_final")
        assert hasattr(cohere_bridge, "rotary_emb")

    def test_no_ln2_in_blocks(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block: no post_attention_layernorm
        for block in cohere_bridge.blocks:
            assert not hasattr(block, "ln2"), "Parallel block must not have ln2"

    def test_cfg_normalization_type(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.cfg.normalization_type == "LN"

    def test_cfg_uses_rms_norm_false(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.cfg.uses_rms_norm is False

    def test_cfg_logit_scale_is_float(self, cohere_bridge: TransformerBridge) -> None:
        assert isinstance(getattr(cohere_bridge.cfg, "logit_scale"), float)

    def test_cfg_logit_scale_value(self, cohere_bridge: TransformerBridge) -> None:
        assert getattr(cohere_bridge.cfg, "logit_scale") == pytest.approx(0.0625)


# ---------------------------------------------------------------------------
# 2. Forward equivalence — HF logits ≈ bridge logits
# ---------------------------------------------------------------------------


class TestCohereForwardEquivalence:
    """Verify bridge and HF produce identical logits for the same input."""

    def test_forward_returns_logits(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = cohere_bridge(tokens)
        assert output.shape[0] == 1
        assert output.shape[1] == 4
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_matches_hf(self, cohere_bridge: TransformerBridge, cohere_hf: Any) -> None:
        """Bridge delegates to HF native forward — logits should be identical."""
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff:.6f}"

    def test_forward_shape_matches_hf(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        assert bridge_out.shape == hf_out.shape


# ---------------------------------------------------------------------------
# 3. Logit scale applied end-to-end
# ---------------------------------------------------------------------------


class TestCohereLogitScaleEndToEnd:
    """Verify logit_scale is correctly folded into the loaded model.

    process_weights must be called before the fold takes effect — boot_transformers
    alone does NOT call process_weights. cohere_bridge_processed uses a fixture that
    calls process_weights with all standard options disabled so only preprocess_weights
    (the logit_scale fold) runs.
    """

    def test_unembed_weight_is_scaled_relative_to_hf(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After preprocess_weights, lm_head.weight inside the bridge should equal
        # HF lm_head.weight * logit_scale (both [d_vocab, d_model]).
        logit_scale = getattr(cohere_bridge_processed.cfg, "logit_scale")
        tl_weight = cohere_bridge_processed.unembed.original_component.weight  # [d_vocab, d_model]
        hf_weight = cohere_hf.lm_head.weight.detach()  # [d_vocab, d_model]
        expected = hf_weight * logit_scale
        max_diff = (tl_weight - expected).abs().max().item()
        assert max_diff < 1e-5, (
            f"unembed.weight not correctly scaled: max_diff={max_diff:.6f}, "
            f"logit_scale={logit_scale}"
        )

    def test_logit_scale_folded_not_applied_twice(
        self, cohere_bridge: TransformerBridge, cohere_hf: Any
    ) -> None:
        """Confirm logit_scale isn't double-applied.

        The bridge (before process_weights) delegates to HF's forward, which includes
        the logit_scale multiply. If forward still matches HF, the fold hasn't been
        applied a second time on top of HF's own scale.
        """
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            bridge_out = cohere_bridge(tokens)
            hf_out = cohere_hf(tokens).logits
        # If logit_scale were applied twice, outputs would differ by a factor of 16
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Possible double-application of logit_scale; diff={max_diff:.6f}"


# ---------------------------------------------------------------------------
# 4. Tied embedding preserved — embed.W_E must NOT be scaled
# ---------------------------------------------------------------------------


class TestCohereTiedEmbedding:
    """Verify preprocess_weights does not corrupt embed.W_E in the tied case.

    Both tests use cohere_bridge_processed (process_weights called with fold-only
    options) because the untie + fold only happens inside process_weights.
    """

    def test_embed_weight_equals_hf_embed_tokens(
        self, cohere_bridge_processed: TransformerBridge, cohere_hf: Any
    ) -> None:
        # After the fold, embed.W_E must still equal HF's unscaled embed_tokens.weight.
        # If the fold corrupted embed (in-place on the shared tensor), this fails.
        hf_embed = cohere_hf.model.embed_tokens.weight.detach()  # [d_vocab, d_model]
        tl_embed = cohere_bridge_processed.embed.W_E  # [d_vocab, d_model]
        max_diff = (tl_embed - hf_embed).abs().max().item()
        assert (
            max_diff < 1e-6
        ), f"embed.W_E was corrupted (possibly by logit_scale fold): max_diff={max_diff:.6f}"

    def test_embed_and_unembed_weights_differ(
        self, cohere_bridge_processed: TransformerBridge
    ) -> None:
        # After the logit_scale fold, embed.W_E and unembed.weight must NOT be identical.
        # If they are, the untie or fold did not take effect.
        logit_scale = getattr(cohere_bridge_processed.cfg, "logit_scale")
        if logit_scale == 1.0:
            pytest.skip("logit_scale=1.0 — fold is a no-op, skip this check")
        tl_embed = cohere_bridge_processed.embed.W_E
        tl_unembed = cohere_bridge_processed.unembed.original_component.weight
        assert not torch.allclose(tl_embed, tl_unembed), (
            "embed.W_E and unembed.weight are identical — "
            "logit_scale fold may not have been applied or untied correctly"
        )


# ---------------------------------------------------------------------------
# 5. HF delegation — RoPE, attention, normalization go through HF modules
# ---------------------------------------------------------------------------


class TestCohereHFDelegation:
    """Spot-check that bridge submodules delegate to actual HF modules.

    This confirms setup_component_testing wired rotary_emb correctly and that
    NormalizationBridge and PositionEmbeddingsAttentionBridge hold live HF objects.
    """

    def test_ln1_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        # bridge.blocks[0].ln1.original_component should be the live HF CohereLayerNorm
        ln1 = cohere_bridge.blocks[0].ln1
        assert isinstance(ln1, NormalizationBridge)
        assert ln1.original_component is not None
        # CohereLayerNorm has variance_epsilon (not eps)
        assert hasattr(
            ln1.original_component, "variance_epsilon"
        ), "original_component is not CohereLayerNorm (missing variance_epsilon)"

    def test_attn_original_component_is_hf_attention(
        self, cohere_bridge: TransformerBridge
    ) -> None:
        attn = cohere_bridge.blocks[0].attn
        assert isinstance(attn, PositionEmbeddingsAttentionBridge)
        assert attn.original_component is not None
        # CohereAttention has q_proj
        assert hasattr(
            attn.original_component, "q_proj"
        ), "original_component is not CohereAttention (missing q_proj)"

    def test_ln_final_original_component_is_hf_norm(self, cohere_bridge: TransformerBridge) -> None:
        assert cohere_bridge.ln_final.original_component is not None
        assert hasattr(cohere_bridge.ln_final.original_component, "variance_epsilon")


# ---------------------------------------------------------------------------
# 6. Parallel-attn hooks fire correctly
# ---------------------------------------------------------------------------


class TestCohereParallelHooks:
    """Verify hook placement for the parallel attention+MLP block."""

    def test_no_hook_resid_mid(self, cohere_bridge: TransformerBridge) -> None:
        # Parallel block has no intermediate residual stream
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        assert not any("hook_resid_mid" in k for k in cache.keys())

    def test_attn_and_mlp_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.attn.hook_in" in cache
            assert f"blocks.{i}.attn.hook_out" in cache
            assert f"blocks.{i}.mlp.hook_in" in cache
            assert f"blocks.{i}.mlp.hook_out" in cache

    def test_residual_hooks_fire(self, cohere_bridge: TransformerBridge) -> None:
        tokens = torch.tensor([[1, 2, 3, 4]])
        _, cache = cohere_bridge.run_with_cache(tokens)
        for i in range(2):
            assert f"blocks.{i}.hook_resid_pre" in cache
            assert f"blocks.{i}.hook_resid_post" in cache
