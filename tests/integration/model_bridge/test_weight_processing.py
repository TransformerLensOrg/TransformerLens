#!/usr/bin/env python3
"""Consolidated weight processing tests for TransformerBridge.

Tests flag combinations, regression anchors, and bridge-vs-HT parity.
Consolidates:
- test_weight_processing_combinations.py (flag matrix + ablation effects)
- compatibility/test_weight_processing_compatibility.py (Main Demo regression anchors)

Uses distilgpt2 for fast flag matrix tests and gpt2 for Main Demo regression anchors.
"""

import pytest
import torch
from jaxtyping import Float

from transformer_lens import HookedTransformer, utils
from transformer_lens.model_bridge import TransformerBridge

# ---------------------------------------------------------------------------
# Flag combination matrix (distilgpt2 for speed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fold_ln,center_writing_weights,center_unembed,fold_value_biases,expected_close_match",
    [
        # Test critical combinations only to speed up CI
        (False, False, False, False, True),  # No processing
        (True, False, False, False, True),  # Only fold_ln (most important)
        (True, True, False, False, True),  # fold_ln + center_writing (common combo)
        (True, True, True, True, True),  # All processing (default)
        # Extended flag combinations (mark test with @pytest.mark.slow to skip in fast CI runs)
        pytest.param(
            False, True, False, False, True, marks=pytest.mark.slow
        ),  # Only center_writing
        pytest.param(
            False, False, True, False, True, marks=pytest.mark.slow
        ),  # Only center_unembed
        pytest.param(
            False, False, False, True, True, marks=pytest.mark.slow
        ),  # Only fold_value_biases
        pytest.param(
            True, False, True, False, True, marks=pytest.mark.slow
        ),  # fold_ln + center_unembed
        pytest.param(
            True, False, False, True, True, marks=pytest.mark.slow
        ),  # fold_ln + fold_value_biases
        pytest.param(
            False, True, True, False, True, marks=pytest.mark.slow
        ),  # center_writing + center_unembed
        pytest.param(
            True, True, True, False, True, marks=pytest.mark.slow
        ),  # All except fold_value_biases
        pytest.param(
            True, True, False, True, True, marks=pytest.mark.slow
        ),  # All except center_unembed
        pytest.param(
            True, False, True, True, True, marks=pytest.mark.slow
        ),  # All except center_writing
        pytest.param(False, True, True, True, True, marks=pytest.mark.slow),  # All except fold_ln
    ],
)
def test_weight_processing_flag_combinations(
    fold_ln, center_writing_weights, center_unembed, fold_value_biases, expected_close_match
):
    """Test that different combinations of weight processing flags work correctly."""
    device = "cpu"
    model_name = "distilgpt2"
    test_text = "Natural language processing"

    # Get reference values from HookedTransformer with same settings
    reference_ht = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        fold_value_biases=fold_value_biases,
        refactor_factored_attn_matrices=False,
    )

    ref_loss = reference_ht(test_text, return_type="loss")

    # Test ablation effect
    hook_name = utils.get_act_name("v", 0)

    def ablation_hook(activation, hook):
        activation[:, :, 8, :] = 0  # Ablate head 8 in layer 0
        return activation

    ref_ablated_loss = reference_ht.run_with_hooks(
        test_text, return_type="loss", fwd_hooks=[(hook_name, ablation_hook)]
    )
    ref_ablation_effect = ref_ablated_loss - ref_loss

    # Create TransformerBridge and apply weight processing
    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.process_weights(
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        fold_value_biases=fold_value_biases,
        refactor_factored_attn_matrices=False,
    )
    bridge.enable_compatibility_mode()

    # Test baseline inference
    bridge_loss = bridge(test_text, return_type="loss")

    # Test ablation with bridge
    bridge_ablated_loss = bridge.run_with_hooks(
        test_text, return_type="loss", fwd_hooks=[(hook_name, ablation_hook)]
    )
    bridge_ablation_effect = bridge_ablated_loss - bridge_loss

    # Compare results
    loss_diff = abs(bridge_loss - ref_loss)
    effect_diff = abs(bridge_ablation_effect - ref_ablation_effect)

    # Assertions
    # Observed values (distilgpt2, 2026-04-07):
    #   Loss diffs: all < 0.00002 across all flag combos
    #   Effect diffs: ~0.133 for partial processing, ~0.000001 for full processing
    #   The partial-processing effect mismatch is due to different V hook capture
    #   points between bridge and HookedTransformer in non-fully-processed mode.
    if expected_close_match:
        assert loss_diff < 0.01, f"Baseline loss difference too large: {loss_diff:.6f}"
        assert effect_diff < 0.5, f"Ablation effect difference too large: {effect_diff:.6f}"

    # Ensure model produces reasonable outputs
    assert not torch.isnan(bridge_loss), "Bridge produced NaN loss"
    assert not torch.isinf(bridge_loss), "Bridge produced infinite loss"


def test_no_processing_matches_unprocessed_hooked_transformer():
    """Test that no processing flag matches HookedTransformer loaded without processing."""
    device = "cpu"
    model_name = "distilgpt2"
    test_text = "Natural language processing"

    unprocessed_ht = HookedTransformer.from_pretrained_no_processing(model_name, device=device)
    unprocessed_loss = unprocessed_ht(test_text, return_type="loss")

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    bridge.enable_compatibility_mode()
    bridge_loss = bridge(test_text, return_type="loss")

    # Observed: < 0.00002 for distilgpt2
    loss_diff = abs(bridge_loss - unprocessed_loss)
    assert loss_diff < 0.01, f"Unprocessed models should match closely: {loss_diff:.6f}"


def test_all_processing_matches_default_hooked_transformer():
    """Test that all processing flags match default HookedTransformer behavior."""
    device = "cpu"
    model_name = "distilgpt2"
    test_text = "Natural language processing"

    default_ht = HookedTransformer.from_pretrained(model_name, device=device)
    default_loss = default_ht(test_text, return_type="loss")

    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode()
    bridge_loss = bridge(test_text, return_type="loss")

    loss_diff = abs(bridge_loss - default_loss)
    assert loss_diff < 0.01, f"Fully processed models should match closely: {loss_diff:.6f}"


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
    silently shift the numbers that published notebooks depend on.
    """

    @pytest.fixture(scope="class")
    def hooked_processed(self):
        return HookedTransformer.from_pretrained("gpt2", device="cpu")

    @pytest.fixture(scope="class")
    def hooked_unprocessed(self):
        return HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")

    @pytest.fixture(scope="class")
    def bridge_processed(self):
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        bridge.enable_compatibility_mode()
        return bridge

    @pytest.fixture(scope="class")
    def bridge_unprocessed(self):
        bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
        bridge.enable_compatibility_mode(no_processing=True)
        return bridge

    def test_hooked_processed_matches_main_demo(self, hooked_processed):
        """HookedTransformer with processing should match Main Demo values."""
        orig, ablated = _run_ablation(
            hooked_processed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        assert (
            abs(orig - EXPECTED_PROCESSED_ORIG) < REGRESSION_TOLERANCE
        ), f"Processed orig {orig:.6f} != expected {EXPECTED_PROCESSED_ORIG}"
        assert (
            abs(ablated - EXPECTED_PROCESSED_ABLATED) < REGRESSION_TOLERANCE
        ), f"Processed ablated {ablated:.6f} != expected {EXPECTED_PROCESSED_ABLATED}"

    def test_hooked_unprocessed_matches_expected(self, hooked_unprocessed):
        """HookedTransformer without processing should match expected values."""
        orig, ablated = _run_ablation(
            hooked_unprocessed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        assert (
            abs(orig - EXPECTED_UNPROCESSED_ORIG) < REGRESSION_TOLERANCE
        ), f"Unprocessed orig {orig:.6f} != expected {EXPECTED_UNPROCESSED_ORIG}"
        assert (
            abs(ablated - EXPECTED_UNPROCESSED_ABLATED) < REGRESSION_TOLERANCE
        ), f"Unprocessed ablated {ablated:.6f} != expected {EXPECTED_UNPROCESSED_ABLATED}"

    def test_processing_preserves_baseline(self, hooked_processed, hooked_unprocessed):
        """Processing should not change baseline loss (mathematical equivalence)."""
        proc_orig, _ = _run_ablation(
            hooked_processed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        unproc_orig, _ = _run_ablation(
            hooked_unprocessed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        assert (
            abs(proc_orig - unproc_orig) < 0.001
        ), f"Baseline not mathematically equivalent: {proc_orig:.6f} vs {unproc_orig:.6f}"

    def test_processing_enhances_ablation_signal(self, hooked_processed, hooked_unprocessed):
        """Processing should increase the ablation effect (better interpretability)."""
        _, proc_ablated = _run_ablation(
            hooked_processed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        _, unproc_ablated = _run_ablation(
            hooked_unprocessed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        diff = abs(proc_ablated - unproc_ablated)
        assert diff > 0.5, (
            f"Processing should significantly change ablation: "
            f"processed={proc_ablated:.6f}, unprocessed={unproc_ablated:.6f}, diff={diff:.6f}"
        )

    def test_bridge_processed_matches_hooked_processed(self, bridge_processed, hooked_processed):
        """TransformerBridge with processing should match HookedTransformer."""
        br_orig, br_ablated = _run_ablation(
            bridge_processed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        ht_orig, ht_ablated = _run_ablation(
            hooked_processed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        # Observed: 0.000000 diff for gpt2 (2026-04-07)
        assert (
            abs(br_orig - ht_orig) < REGRESSION_TOLERANCE
        ), f"Bridge processed orig {br_orig:.6f} != HT {ht_orig:.6f}"
        assert (
            abs(br_ablated - ht_ablated) < REGRESSION_TOLERANCE
        ), f"Bridge processed ablated {br_ablated:.6f} != HT {ht_ablated:.6f}"

    def test_bridge_unprocessed_matches_hooked_unprocessed(
        self, bridge_unprocessed, hooked_unprocessed
    ):
        """TransformerBridge without processing should match HookedTransformer."""
        br_orig, br_ablated = _run_ablation(
            bridge_unprocessed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        ht_orig, ht_ablated = _run_ablation(
            hooked_unprocessed, MAIN_DEMO_TEXT, MAIN_DEMO_LAYER, MAIN_DEMO_HEAD
        )
        # Observed: 0.000000 diff for gpt2 (2026-04-07)
        assert (
            abs(br_orig - ht_orig) < REGRESSION_TOLERANCE
        ), f"Bridge unprocessed orig {br_orig:.6f} != HT {ht_orig:.6f}"
        assert (
            abs(br_ablated - ht_ablated) < REGRESSION_TOLERANCE
        ), f"Bridge unprocessed ablated {br_ablated:.6f} != HT {ht_ablated:.6f}"
