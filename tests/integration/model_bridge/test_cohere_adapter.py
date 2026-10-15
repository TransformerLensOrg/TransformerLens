"""Integration tests for Cohere architecture adapter (CohereForCausalLM).

Model: trl-internal-testing/tiny-CohereForCausalLM
  - 2 layers, ~8M params, CPU-safe, no gating required
  - tie_word_embeddings=True by default
  - logit_scale is read from the checkpoint config (not an adapter default)

The pinned checkpoint has use_qk_norm=False. Processed-path tests additionally
exercise a deterministic synthetic Cohere model with QK normalization enabled.
"""

from copy import deepcopy
from typing import Any

import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_OFFLINE
from transformers import AutoModelForCausalLM

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import NormalizationBridge
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

MODEL = "trl-internal-testing/tiny-CohereForCausalLM"
MODEL_REVISION = "90bc56bd12c642f6bf5e710ebd7c626dd7c0201c"


@pytest.fixture(scope="module")
def cohere_checkpoint():
    """Pin config, tokenizer, and weights together, including bridge boot reads."""
    return snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=HF_HUB_OFFLINE)


@pytest.fixture(scope="module")
def cohere_bridge(cohere_checkpoint):
    """Load tiny Cohere bridge once per module (no weight processing)."""
    return TransformerBridge.boot_transformers(cohere_checkpoint, device="cpu")


@pytest.fixture(scope="module")
def cohere_bridge_processed(cohere_checkpoint):
    """Bridge with preprocess_weights applied (fold only, no centering).

    process_weights must be called explicitly — boot_transformers does not call
    it automatically. We disable all ProcessWeights options so only the adapter's
    preprocess_weights (logit_scale fold + untie) runs.
    """
    bridge = TransformerBridge.boot_transformers(cohere_checkpoint, device="cpu")
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    return bridge


@pytest.fixture(scope="module")
def cohere_hf(cohere_checkpoint) -> Any:
    """Load the raw HF model for side-by-side comparisons."""
    return AutoModelForCausalLM.from_pretrained(cohere_checkpoint).eval()


@pytest.fixture(
    scope="module",
    params=[(False, False), (False, True), (True, False), (True, True)],
    ids=["raw-logits", "centered-log-probs", "qk-raw-logits", "qk-centered-log-probs"],
)
def cohere_folded_pair(request, cohere_hf, cohere_checkpoint):
    """Independent reference and processed model, with non-identity norm scales.

    Tiny checkpoints can have identity LayerNorm weights, hiding broken folding.
    Change only a private reference copy, then clone it before processing so the
    reference cannot inherit mutations made by the bridge.
    """
    use_qk_norm, centered = request.param
    if use_qk_norm:
        config = deepcopy(cohere_hf.config)
        config.use_qk_norm = True
        with torch.random.fork_rng():
            torch.manual_seed(1124)
            reference = AutoModelForCausalLM.from_config(config).float().eval()
        assert reference.model.layers[0].self_attn.q_norm is not None
    else:
        reference = deepcopy(cohere_hf).float().eval()
    with torch.no_grad():
        for module in reference.modules():
            if hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                module.weight.copy_(
                    torch.linspace(0.5, 1.5, module.weight.numel()).reshape_as(module.weight)
                )
    bridge = TransformerBridge.boot_transformers(
        cohere_checkpoint,
        device="cpu",
        dtype=torch.float32,
        hf_model=deepcopy(reference),
        hf_config_overrides={"use_qk_norm": use_qk_norm},
    ).eval()
    bridge.enable_compatibility_mode(
        fold_ln=True,
        center_writing_weights=centered,
        center_unembed=centered,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    )
    return bridge, reference, centered


@pytest.mark.parametrize(
    "token_ids",
    [
        [[1, 2, 3, 4]],
        [[1, 7, 7, 7, 2, 19, 3, 5]],
        [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
        [[1] + [2 + (i * 17) % 127 for i in range(63)]],
    ],
    ids=["short", "repeated", "batch", "long-64"],
)
def test_cohere_folded_forward_matches_hf(cohere_folded_pair, token_ids):
    """Exercise normalization folding, including the default centered mode.

    Centering the unembedding changes raw logits by an additive constant, so
    compare log-probabilities in that mode. No relative tolerance is allowed.
    """
    bridge, reference, centered = cohere_folded_pair
    tokens = torch.tensor(token_ids)
    with torch.no_grad():
        actual = bridge(tokens)
        expected = reference(tokens).logits
        if centered:
            actual = actual.float().log_softmax(dim=-1)
            expected = expected.float().log_softmax(dim=-1)
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    max_diff = (actual - expected).abs().max().item()
    assert max_diff < 1e-4, f"Processed bridge vs HF max diff = {max_diff:.8g}"


def test_cohere_folded_norm_epsilon_placement(cohere_folded_pair):
    """Low-variance activations expose epsilon moved outside the square root.

    Folding moves the learned norm scale into downstream projections. Restore
    that scale only on the observed norm output when comparing with native HF.
    This checks the live processed normalization, not a copied formula.
    """
    bridge, reference, _ = cohere_folded_pair
    native_norm = reference.model.layers[0].input_layernorm
    x = torch.linspace(-1e-4, 1e-4, bridge.cfg.d_model).reshape(1, 1, -1)
    with torch.no_grad():
        actual = bridge.blocks[0].ln1(x) * native_norm.weight
        expected = native_norm(x)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=0)


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

    def test_cfg_logit_scale_value(self, cohere_bridge: TransformerBridge, cohere_hf: Any) -> None:
        assert getattr(cohere_bridge.cfg, "logit_scale") == pytest.approx(
            cohere_hf.config.logit_scale
        )


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
