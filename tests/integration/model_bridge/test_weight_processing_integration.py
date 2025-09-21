#!/usr/bin/env python3
"""
Integration Compatibility Test for Weight Processing
====================================================

This test verifies that:
1. HookedTransformer with processing matches expected Main Demo values (3.999 → 5.453)
2. HookedTransformer without processing matches expected unprocessed values (~3.999 → ~4.117)
3. TransformerBridge with processing matches HookedTransformer with processing
4. TransformerBridge without processing matches HookedTransformer without processing
5. Processing maintains mathematical equivalence for baseline computation
6. Processing changes ablation results as expected (for better interpretability)
"""

import torch
from jaxtyping import Float

from transformer_lens import HookedTransformer, utils
from transformer_lens.model_bridge.bridge import TransformerBridge


def test_integration_compatibility():
    """Test integration compatibility between HookedTransformer and TransformerBridge."""
    model_name = "gpt2"
    device = "cpu"

    # Test text from Main Demo
    test_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."

    # Ablation parameters from Main Demo
    layer_to_ablate = 0
    head_index_to_ablate = 8

    print("=== INTEGRATION COMPATIBILITY TEST ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Test text: {test_text[:50]}...")
    print(f"Ablating layer {layer_to_ablate}, head {head_index_to_ablate}")

    # ===========================================
    # STEP 1: HookedTransformer with processing
    # ===========================================
    print("\n1. Loading HookedTransformer with processing...")
    hooked_processed = HookedTransformer.from_pretrained(model_name, device=device)
    tokens = hooked_processed.to_tokens(test_text)

    print("\n   Testing baseline performance...")
    hooked_processed_baseline = hooked_processed(tokens, return_type="loss")
    print(f"   HookedTransformer (processed) baseline: {hooked_processed_baseline.item():.6f}")

    print("\n   Testing ablation performance...")

    def head_ablation_hook(value: Float[torch.Tensor, "batch pos head_index d_head"], hook):
        value[:, :, head_index_to_ablate, :] = 0.0
        return value

    hook_name = utils.get_act_name("v", layer_to_ablate)
    hooked_processed_ablated = hooked_processed.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook)]
    )
    print(f"   HookedTransformer (processed) ablated: {hooked_processed_ablated.item():.6f}")

    hooked_processed_gain = hooked_processed_ablated.item() - hooked_processed_baseline.item()
    print(f"   HookedTransformer (processed) gain: {hooked_processed_gain:.6f}")

    # ===========================================
    # STEP 2: HookedTransformer without processing
    # ===========================================
    print("\n2. Loading HookedTransformer without processing...")
    hooked_unprocessed = HookedTransformer.from_pretrained_no_processing(model_name, device=device)

    print("\n   Testing baseline performance...")
    hooked_unprocessed_baseline = hooked_unprocessed(tokens, return_type="loss")
    print(f"   HookedTransformer (unprocessed) baseline: {hooked_unprocessed_baseline.item():.6f}")

    print("\n   Testing ablation performance...")
    hooked_unprocessed_ablated = hooked_unprocessed.run_with_hooks(
        tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook)]
    )
    print(f"   HookedTransformer (unprocessed) ablated: {hooked_unprocessed_ablated.item():.6f}")

    hooked_unprocessed_gain = hooked_unprocessed_ablated.item() - hooked_unprocessed_baseline.item()
    print(f"   HookedTransformer (unprocessed) gain: {hooked_unprocessed_gain:.6f}")

    # ===========================================
    # STEP 3: TransformerBridge without processing
    # ===========================================
    print("\n3. Loading TransformerBridge without processing...")
    try:
        bridge_unprocessed = TransformerBridge.boot_transformers(
            model_name, device=device, apply_weight_processing=False
        )

        print("\n   Testing baseline performance...")
        bridge_unprocessed_baseline = bridge_unprocessed(tokens, return_type="loss")
        print(
            f"   TransformerBridge (unprocessed) baseline: {bridge_unprocessed_baseline.item():.6f}"
        )

        print("\n   Testing ablation performance...")
        bridge_unprocessed_ablated = bridge_unprocessed.run_with_hooks(
            tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook)]
        )
        print(
            f"   TransformerBridge (unprocessed) ablated: {bridge_unprocessed_ablated.item():.6f}"
        )

        bridge_unprocessed_gain = (
            bridge_unprocessed_ablated.item() - bridge_unprocessed_baseline.item()
        )
        print(f"   TransformerBridge (unprocessed) gain: {bridge_unprocessed_gain:.6f}")

        bridge_unprocessed_success = True

    except Exception as e:
        print(f"   ❌ TransformerBridge (unprocessed) failed: {e}")
        bridge_unprocessed_success = False

    # ===========================================
    # STEP 4: TransformerBridge with processing
    # ===========================================
    print("\n4. Loading TransformerBridge with processing...")
    try:
        bridge_processed = TransformerBridge.boot_transformers(
            model_name, device=device, apply_weight_processing=True
        )

        print("\n   Testing baseline performance...")
        bridge_processed_baseline = bridge_processed(tokens, return_type="loss")
        print(f"   TransformerBridge (processed) baseline: {bridge_processed_baseline.item():.6f}")

        print("\n   Testing ablation performance...")
        bridge_processed_ablated = bridge_processed.run_with_hooks(
            tokens, return_type="loss", fwd_hooks=[(hook_name, head_ablation_hook)]
        )
        print(f"   TransformerBridge (processed) ablated: {bridge_processed_ablated.item():.6f}")

        bridge_processed_gain = bridge_processed_ablated.item() - bridge_processed_baseline.item()
        print(f"   TransformerBridge (processed) gain: {bridge_processed_gain:.6f}")

        bridge_processed_success = True

    except Exception as e:
        print(f"   ❌ TransformerBridge (processed) failed: {e}")
        bridge_processed_success = False

    # ===========================================
    # ANALYSIS
    # ===========================================
    print("\n" + "=" * 60)
    print("COMPATIBILITY ANALYSIS")
    print("=" * 60)

    # Expected values from Main Demo
    expected_processed_baseline = 3.999
    expected_processed_ablated = 5.453
    expected_unprocessed_baseline = 3.999
    expected_unprocessed_ablated = 4.117

    tolerance_strict = 0.01
    tolerance_loose = 0.1

    print("\n1. HookedTransformer Validation:")
    processed_baseline_match = (
        abs(hooked_processed_baseline.item() - expected_processed_baseline) < tolerance_strict
    )
    processed_ablated_match = (
        abs(hooked_processed_ablated.item() - expected_processed_ablated) < tolerance_strict
    )
    unprocessed_baseline_match = (
        abs(hooked_unprocessed_baseline.item() - expected_unprocessed_baseline) < tolerance_strict
    )
    unprocessed_ablated_match = (
        abs(hooked_unprocessed_ablated.item() - expected_unprocessed_ablated) < tolerance_loose
    )

    print(
        f"   Processed baseline: {'✅' if processed_baseline_match else '❌'} {hooked_processed_baseline.item():.6f} (expected ~{expected_processed_baseline})"
    )
    print(
        f"   Processed ablated:  {'✅' if processed_ablated_match else '❌'} {hooked_processed_ablated.item():.6f} (expected ~{expected_processed_ablated})"
    )
    print(
        f"   Unprocessed baseline: {'✅' if unprocessed_baseline_match else '❌'} {hooked_unprocessed_baseline.item():.6f} (expected ~{expected_unprocessed_baseline})"
    )
    print(
        f"   Unprocessed ablated:  {'✅' if unprocessed_ablated_match else '❌'} {hooked_unprocessed_ablated.item():.6f} (expected ~{expected_unprocessed_ablated})"
    )

    if bridge_unprocessed_success:
        print("\n2. Bridge vs HookedTransformer (Unprocessed) Compatibility:")
        bridge_hooked_baseline_diff = abs(
            bridge_unprocessed_baseline.item() - hooked_unprocessed_baseline.item()
        )
        bridge_hooked_ablated_diff = abs(
            bridge_unprocessed_ablated.item() - hooked_unprocessed_ablated.item()
        )
        bridge_hooked_gain_diff = abs(bridge_unprocessed_gain - hooked_unprocessed_gain)

        baseline_compatible = bridge_hooked_baseline_diff < tolerance_strict
        ablated_compatible = bridge_hooked_ablated_diff < tolerance_strict
        gain_compatible = bridge_hooked_gain_diff < tolerance_strict

        print(
            f"   Baseline diff: {'✅' if baseline_compatible else '❌'} {bridge_hooked_baseline_diff:.6f}"
        )
        print(
            f"   Ablated diff:  {'✅' if ablated_compatible else '❌'} {bridge_hooked_ablated_diff:.6f}"
        )
        print(f"   Gain diff:     {'✅' if gain_compatible else '❌'} {bridge_hooked_gain_diff:.6f}")

    if bridge_processed_success:
        print("\n3. Bridge vs HookedTransformer (Processed) Compatibility:")
        bridge_hooked_processed_baseline_diff = abs(
            bridge_processed_baseline.item() - hooked_processed_baseline.item()
        )
        bridge_hooked_processed_ablated_diff = abs(
            bridge_processed_ablated.item() - hooked_processed_ablated.item()
        )
        bridge_hooked_processed_gain_diff = abs(bridge_processed_gain - hooked_processed_gain)

        processed_baseline_compatible = bridge_hooked_processed_baseline_diff < tolerance_strict
        processed_ablated_compatible = bridge_hooked_processed_ablated_diff < tolerance_strict
        processed_gain_compatible = bridge_hooked_processed_gain_diff < tolerance_strict

        print(
            f"   Baseline diff: {'✅' if processed_baseline_compatible else '❌'} {bridge_hooked_processed_baseline_diff:.6f}"
        )
        print(
            f"   Ablated diff:  {'✅' if processed_ablated_compatible else '❌'} {bridge_hooked_processed_ablated_diff:.6f}"
        )
        print(
            f"   Gain diff:     {'✅' if processed_gain_compatible else '❌'} {bridge_hooked_processed_gain_diff:.6f}"
        )

    print("\n4. Processing Effect Analysis:")
    processing_improves_interpretability = hooked_processed_gain > hooked_unprocessed_gain
    print(
        f"   Processing improves interpretability: {'✅' if processing_improves_interpretability else '❌'}"
    )
    print(f"   Processed gain: {hooked_processed_gain:.6f}")
    print(f"   Unprocessed gain: {hooked_unprocessed_gain:.6f}")
    print(f"   Improvement: {hooked_processed_gain - hooked_unprocessed_gain:.6f}")

    # ===========================================
    # FINAL VERDICT
    # ===========================================
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    hooked_valid = (
        processed_baseline_match
        and processed_ablated_match
        and unprocessed_baseline_match
        and unprocessed_ablated_match
    )
    bridge_unprocessed_compatible = (
        bridge_unprocessed_success
        and baseline_compatible
        and ablated_compatible
        and gain_compatible
        if bridge_unprocessed_success
        else False
    )
    bridge_processed_compatible = (
        bridge_processed_success
        and processed_baseline_compatible
        and processed_ablated_compatible
        and processed_gain_compatible
        if bridge_processed_success
        else False
    )

    print(f"HookedTransformer validation: {'✅' if hooked_valid else '❌'}")
    print(f"Bridge (unprocessed) compatibility: {'✅' if bridge_unprocessed_compatible else '❌'}")
    print(f"Bridge (processed) compatibility: {'✅' if bridge_processed_compatible else '❌'}")
    print(f"Processing effectiveness: {'✅' if processing_improves_interpretability else '❌'}")

    overall_success = (
        hooked_valid
        and bridge_unprocessed_compatible
        and bridge_processed_compatible
        and processing_improves_interpretability
    )

    if overall_success:
        print("\n🎉🎉🎉 FULL INTEGRATION COMPATIBILITY ACHIEVED! 🎉🎉🎉")
        print("TransformerBridge is fully compatible with HookedTransformer!")
        return True
    else:
        print("\n⚠️ Integration compatibility issues detected")
        return False


if __name__ == "__main__":
    success = test_integration_compatibility()
    if success:
        print("\n🚀 INTEGRATION READY FOR PRODUCTION! 🚀")
