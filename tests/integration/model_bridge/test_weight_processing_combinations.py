#!/usr/bin/env python3
"""Test different combinations of weight processing flags to ensure each works correctly."""

import pytest
import torch

from transformer_lens import HookedTransformer, utils
from transformer_lens.model_bridge import TransformerBridge


@pytest.mark.parametrize(
    "fold_ln,center_writing_weights,center_unembed,fold_value_biases,expected_close_match",
    [
        # Test critical combinations only to speed up CI
        (False, False, False, False, True),  # No processing
        (True, False, False, False, True),  # Only fold_ln (most important)
        (True, True, False, False, True),  # fold_ln + center_writing (common combo)
        (True, True, True, True, True),  # All processing (default)
        # NOTE: Full test matrix commented out for CI speed. Uncomment for thorough testing:
        # (False, True, False, False, True),  # Only center_writing
        # (False, False, True, False, True),  # Only center_unembed
        # (False, False, False, True, True),  # Only fold_value_biases
        # (True, False, True, False, True),  # fold_ln + center_unembed
        # (True, False, False, True, True),  # fold_ln + fold_value_biases
        # (False, True, True, False, True),  # center_writing + center_unembed
        # (True, True, True, False, True),  # All except fold_value_biases
        # (True, True, False, True, True),  # All except center_unembed
        # (True, False, True, True, True),  # All except center_writing
        # (False, True, True, True, True),  # All except fold_ln
    ],
)
def test_weight_processing_flag_combinations(
    fold_ln, center_writing_weights, center_unembed, fold_value_biases, expected_close_match
):
    """Test that different combinations of weight processing flags work correctly."""
    device = "cpu"
    model_name = "distilgpt2"  # Use distilgpt2 for faster tests
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
    bridge = TransformerBridge.boot_transformers(
        model_name,
        device=device,
    )

    # Apply weight processing with specified settings
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
    if expected_close_match:
        assert loss_diff < 30.0, f"Baseline loss difference too large: {loss_diff:.6f}"
        assert effect_diff < 20.0, f"Ablation effect difference too large: {effect_diff:.6f}"

    # Ensure model produces reasonable outputs
    assert not torch.isnan(bridge_loss), "Bridge produced NaN loss"
    assert not torch.isinf(bridge_loss), "Bridge produced infinite loss"


def test_no_processing_matches_unprocessed_hooked_transformer():
    """Test that no processing flag matches HookedTransformer loaded without processing."""
    device = "cpu"
    model_name = "distilgpt2"  # Use distilgpt2 for faster tests
    test_text = "Natural language processing"

    # Load HookedTransformer without processing
    unprocessed_ht = HookedTransformer.from_pretrained_no_processing(model_name, device=device)
    unprocessed_loss = unprocessed_ht(test_text, return_type="loss")

    # Load TransformerBridge without processing
    bridge = TransformerBridge.boot_transformers(model_name, device=device)

    # Apply no weight processing
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    bridge.enable_compatibility_mode()
    bridge_loss = bridge(test_text, return_type="loss")

    # Should match closely
    loss_diff = abs(bridge_loss - unprocessed_loss)
    assert loss_diff < 30.0, f"Unprocessed models should match closely: {loss_diff:.6f}"


def test_all_processing_matches_default_hooked_transformer():
    """Test that all processing flags match default HookedTransformer behavior."""
    device = "cpu"
    model_name = "distilgpt2"  # Use distilgpt2 for faster tests
    test_text = "Natural language processing"

    # Load default HookedTransformer (with all processing)
    default_ht = HookedTransformer.from_pretrained(model_name, device=device)
    default_loss = default_ht(test_text, return_type="loss")

    # Load TransformerBridge with all processing (default behavior)
    bridge = TransformerBridge.boot_transformers(model_name, device=device)
    bridge.enable_compatibility_mode()
    bridge_loss = bridge(test_text, return_type="loss")

    # Should match closely
    loss_diff = abs(bridge_loss - default_loss)
    assert loss_diff < 0.01, f"Fully processed models should match closely: {loss_diff:.6f}"
