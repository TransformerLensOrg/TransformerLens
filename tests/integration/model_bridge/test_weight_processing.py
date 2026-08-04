#!/usr/bin/env python3
"""Consolidated weight processing tests for TransformerBridge.

Tests flag combinations, regression anchors, and parity against the frozen
HookedTransformer goldens (tests/goldens.py). Consolidates:
- test_weight_processing_combinations.py (flag matrix + ablation effects)
- compatibility/test_weight_processing_compatibility.py (Main Demo regression anchors)

Uses distilgpt2 for fast flag matrix tests and gpt2 for Main Demo regression anchors.
"""

import pytest
import torch
from jaxtyping import Float

from tests import goldens
from transformer_lens import utils
from transformer_lens.model_bridge import TransformerBridge

# ---------------------------------------------------------------------------
# Flag combination matrix (distilgpt2 for speed)
# ---------------------------------------------------------------------------


# The four golden processing configs (see scripts/capture_ht_goldens.py). The
# former @slow flag combinations are covered by the per-function math invariant
# tests instead — they had no independent reference once HT is frozen.
GOLDEN_FLAG_CONFIGS = {
    "no_processing": dict(
        fold_ln=False, center_writing_weights=False, center_unembed=False, fold_value_biases=False
    ),
    "fold_ln_only": dict(
        fold_ln=True, center_writing_weights=False, center_unembed=False, fold_value_biases=False
    ),
    "fold_ln_center_writing": dict(
        fold_ln=True, center_writing_weights=True, center_unembed=False, fold_value_biases=False
    ),
    "full_defaults": dict(
        fold_ln=True, center_writing_weights=True, center_unembed=True, fold_value_biases=True
    ),
}


@pytest.mark.parametrize("config_name", list(GOLDEN_FLAG_CONFIGS))
def test_weight_processing_flag_combinations(config_name):
    """Each golden processing config must reproduce the frozen HT loss and ablation effect."""
    if not goldens.goldens_available("distilgpt2", config_name):
        pytest.skip("TL goldens dataset unavailable (set TL_GOLDENS_DIR or enable network)")
    golden = goldens.GoldenCell("distilgpt2", config_name)
    anchor = golden.scalars["ablation"]
    flags = GOLDEN_FLAG_CONFIGS[config_name]

    # enable_compatibility_mode() calls process_weights() internally,
    # so pass flags there directly (not via separate process_weights call).
    bridge = TransformerBridge.boot_transformers("distilgpt2", device="cpu")
    bridge.enable_compatibility_mode(refactor_factored_attn_matrices=False, **flags)

    hook_name = anchor["hook"]
    head = anchor["head"]

    def ablation_hook(activation, hook):
        activation[:, :, head, :] = 0
        return activation

    bridge_loss = bridge(anchor["text"], return_type="loss")
    bridge_ablated_loss = bridge.run_with_hooks(
        anchor["text"], return_type="loss", fwd_hooks=[(hook_name, ablation_hook)]
    )
    bridge_effect = (bridge_ablated_loss - bridge_loss).item()
    golden_effect = anchor["ablated_loss"] - anchor["orig_loss"]

    # Observed values (distilgpt2, 2026-04-07):
    #   Loss diffs: all < 0.00002 across all flag combos
    #   Effect diffs: ~0.133 for partial processing, ~0.000001 for full processing
    #   The partial-processing effect mismatch is due to different V hook capture
    #   points between bridge and HookedTransformer in non-fully-processed mode.
    loss_diff = abs(bridge_loss.item() - anchor["orig_loss"])
    effect_diff = abs(bridge_effect - golden_effect)
    assert loss_diff < 0.01, f"Baseline loss difference too large: {loss_diff:.6f}"
    assert effect_diff < 0.5, f"Ablation effect difference too large: {effect_diff:.6f}"

    assert not torch.isnan(bridge_loss), "Bridge produced NaN loss"
    assert not torch.isinf(bridge_loss), "Bridge produced infinite loss"


def test_no_processing_matches_unprocessed_hooked_transformer(
    distilgpt2_goldens_unprocessed, distilgpt2_bridge_compat_no_processing
):
    """No-processing bridge logits must match the frozen unprocessed HT logits."""
    golden = distilgpt2_goldens_unprocessed
    golden_logits = golden.tensors("logits_short")["logits"]

    with torch.no_grad():
        bridge_logits = distilgpt2_bridge_compat_no_processing(
            golden.scalars["short_prompt"], return_type="logits"
        )

    max_diff = (bridge_logits - golden_logits).abs().max().item()
    assert max_diff < 1e-2, f"Unprocessed logits should match the golden: {max_diff:.6f}"


def test_all_processing_matches_default_hooked_transformer(
    distilgpt2_goldens_processed, distilgpt2_bridge_compat
):
    """Fully processed bridge logits must match the frozen processed HT logits."""
    golden = distilgpt2_goldens_processed
    golden_logits = golden.tensors("logits_short")["logits"]

    with torch.no_grad():
        bridge_logits = distilgpt2_bridge_compat(
            golden.scalars["short_prompt"], return_type="logits"
        )

    max_diff = (bridge_logits - golden_logits).abs().max().item()
    assert max_diff < 1e-2, f"Processed logits should match the golden: {max_diff:.6f}"


# ---------------------------------------------------------------------------
# Main Demo regression anchors (gpt2 — matches published demo values)
# ---------------------------------------------------------------------------

# Expected values from the TransformerLens Main Demo notebook
MAIN_DEMO_TEXT = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
MAIN_DEMO_LAYER = 0
MAIN_DEMO_HEAD = 8
EXPECTED_PROCESSED_ORIG = 3.999
EXPECTED_PROCESSED_ABLATED = 5.453
EXPECTED_UNPROCESSED_ORIG = 3.999
EXPECTED_UNPROCESSED_ABLATED = 4.117
REGRESSION_TOLERANCE = 0.01


def _run_ablation(model, text, layer, head):
    """Run baseline + ablation and return (orig_loss, ablated_loss)."""
    tokens = model.to_tokens(text)

    def ablation_hook(
        value: Float[torch.Tensor, "batch pos head_index d_head"], hook
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        value[:, :, head, :] = 0.0
        return value

    hook_name = utils.get_act_name("v", layer)
    orig = model(tokens, return_type="loss").item()
    ablated = model.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(hook_name, ablation_hook)]
    ).item()
    return orig, ablated


class TestMainDemoRegression:
    """Regression anchors from the TransformerLens Main Demo.

    These tests pin the exact loss values produced by gpt2 with and without
    weight processing, ensuring that changes to weight processing code don't
    silently shift the numbers that published notebooks depend on. The frozen
    HT side lives in the goldens; the bridge replays the same intervention.
    """

    def test_golden_processed_matches_main_demo(self, gpt2_goldens_processed):
        """The frozen processed-HT anchors must match the published Main Demo values."""
        anchor = gpt2_goldens_processed.scalars["ablation"]
        assert anchor["text"] == MAIN_DEMO_TEXT
        assert anchor["layer"] == MAIN_DEMO_LAYER and anchor["head"] == MAIN_DEMO_HEAD
        assert (
            abs(anchor["orig_loss"] - EXPECTED_PROCESSED_ORIG) < REGRESSION_TOLERANCE
        ), f"Golden processed orig {anchor['orig_loss']:.6f} != expected {EXPECTED_PROCESSED_ORIG}"
        assert (
            abs(anchor["ablated_loss"] - EXPECTED_PROCESSED_ABLATED) < REGRESSION_TOLERANCE
        ), f"Golden processed ablated {anchor['ablated_loss']:.6f} != expected {EXPECTED_PROCESSED_ABLATED}"

    def test_golden_unprocessed_matches_expected(self, gpt2_goldens_unprocessed):
        """The frozen unprocessed-HT anchors must match the published expected values."""
        anchor = gpt2_goldens_unprocessed.scalars["ablation"]
        assert (
            abs(anchor["orig_loss"] - EXPECTED_UNPROCESSED_ORIG) < REGRESSION_TOLERANCE
        ), f"Golden unprocessed orig {anchor['orig_loss']:.6f} != expected {EXPECTED_UNPROCESSED_ORIG}"
        assert (
            abs(anchor["ablated_loss"] - EXPECTED_UNPROCESSED_ABLATED) < REGRESSION_TOLERANCE
        ), f"Golden unprocessed ablated {anchor['ablated_loss']:.6f} != expected {EXPECTED_UNPROCESSED_ABLATED}"

    def test_processing_preserves_baseline(self, gpt2_goldens_processed, gpt2_goldens_unprocessed):
        """Processing should not change baseline loss (mathematical equivalence)."""
        proc_orig = gpt2_goldens_processed.scalars["ablation"]["orig_loss"]
        unproc_orig = gpt2_goldens_unprocessed.scalars["ablation"]["orig_loss"]
        assert (
            abs(proc_orig - unproc_orig) < 0.001
        ), f"Baseline not mathematically equivalent: {proc_orig:.6f} vs {unproc_orig:.6f}"

    def test_processing_enhances_ablation_signal(
        self, gpt2_goldens_processed, gpt2_goldens_unprocessed
    ):
        """Processing should increase the ablation effect (better interpretability)."""
        proc_ablated = gpt2_goldens_processed.scalars["ablation"]["ablated_loss"]
        unproc_ablated = gpt2_goldens_unprocessed.scalars["ablation"]["ablated_loss"]
        diff = abs(proc_ablated - unproc_ablated)
        assert diff > 0.5, (
            f"Processing should significantly change ablation: "
            f"processed={proc_ablated:.6f}, unprocessed={unproc_ablated:.6f}, diff={diff:.6f}"
        )

    def test_bridge_processed_matches_hooked_processed(
        self, gpt2_bridge_compat, gpt2_goldens_processed
    ):
        """TransformerBridge with processing should match the frozen HT anchors."""
        anchor = gpt2_goldens_processed.scalars["ablation"]
        br_orig, br_ablated = _run_ablation(
            gpt2_bridge_compat, anchor["text"], anchor["layer"], anchor["head"]
        )
        # Observed: 0.000000 diff for gpt2 (2026-04-07)
        assert (
            abs(br_orig - anchor["orig_loss"]) < REGRESSION_TOLERANCE
        ), f"Bridge processed orig {br_orig:.6f} != golden {anchor['orig_loss']:.6f}"
        assert (
            abs(br_ablated - anchor["ablated_loss"]) < REGRESSION_TOLERANCE
        ), f"Bridge processed ablated {br_ablated:.6f} != golden {anchor['ablated_loss']:.6f}"

    def test_bridge_unprocessed_matches_hooked_unprocessed(
        self, gpt2_bridge_compat_no_processing, gpt2_goldens_unprocessed
    ):
        """TransformerBridge without processing should match the frozen HT anchors."""
        anchor = gpt2_goldens_unprocessed.scalars["ablation"]
        br_orig, br_ablated = _run_ablation(
            gpt2_bridge_compat_no_processing, anchor["text"], anchor["layer"], anchor["head"]
        )
        # Observed: 0.000000 diff for gpt2 (2026-04-07)
        assert (
            abs(br_orig - anchor["orig_loss"]) < REGRESSION_TOLERANCE
        ), f"Bridge unprocessed orig {br_orig:.6f} != golden {anchor['orig_loss']:.6f}"
        assert (
            abs(br_ablated - anchor["ablated_loss"]) < REGRESSION_TOLERANCE
        ), f"Bridge unprocessed ablated {br_ablated:.6f} != golden {anchor['ablated_loss']:.6f}"
