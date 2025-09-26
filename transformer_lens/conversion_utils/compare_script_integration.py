"""Integration test for weight processing and round-trip conversion validation.

This module provides comprehensive testing to ensure that:
1. Weight processing (layer norm folding, etc.) works correctly
2. Round-trip HF ↔ TLens conversions preserve processed weights
3. All model variants produce equivalent outputs
"""

import torch
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.conversion_utils.round_trip_validator import RoundTripValidator


def run_weight_processing_integration_test(
    model_name: str = "gpt2",
    device: str = "cpu",
    test_tokens: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """Run comprehensive integration test for weight processing workflow.

    Args:
        model_name: Model to test (default: "gpt2")
        device: Device to run tests on
        test_tokens: Input tokens for testing (auto-generated if None)

    Returns:
        Dictionary containing all test results
    """
    print(f"=== Weight Processing Integration Test: {model_name} ===")

    if test_tokens is None:
        # Create standard test tokens
        test_tokens = torch.randint(0, 50257, (1, 33), device=device)

    results = {
        "model_name": model_name,
        "device": device,
        "models": {},
        "loss_comparisons": {},
        "ablation_tests": {},
        "round_trip_validation": {},
        "overall_success": False
    }

    try:
        # Load all model variants
        print("Loading models...")
        models = _load_all_model_variants(model_name, device)
        results["models"] = {name: "loaded" for name in models.keys()}

        # Test original losses
        print("Testing original losses...")
        loss_results = _test_original_losses(models, test_tokens)
        results["loss_comparisons"] = loss_results

        # Test ablations
        print("Testing ablations...")
        ablation_results = _test_ablations(models, test_tokens)
        results["ablation_tests"] = ablation_results

        # Test round-trip conversion
        print("Testing round-trip conversion...")
        if "bridge_processed" in models:
            round_trip_results = _test_round_trip_conversion(
                models["bridge_processed"], model_name
            )
            results["round_trip_validation"] = round_trip_results
        else:
            results["round_trip_validation"] = {"error": "No processed bridge model available"}

        # Determine overall success
        results["overall_success"] = _determine_overall_success(results)

        print(f"Integration test completed. Overall success: {results['overall_success']}")

    except Exception as e:
        results["error"] = str(e)
        results["overall_success"] = False
        print(f"Integration test failed: {e}")

    return results


def _load_all_model_variants(model_name: str, device: str) -> Dict[str, Any]:
    """Load all model variants for comparison."""
    models = {}

    # HookedTransformer with processing
    print("  Loading HookedTransformer with processing...")
    models["hooked_processed"] = HookedTransformer.from_pretrained(
        model_name, device=device, fold_ln=True, center_writing_weights=True,
        fold_value_biases=True
    )

    # TransformerBridge with processing
    print("  Loading TransformerBridge with processing...")
    models["bridge_processed"] = TransformerBridge.boot_transformers(
        model_name, device=device
    )

    # Manual processing for comparison
    print("  Creating manual processed model...")
    models["manual_processed"] = _create_manual_processed_model(model_name, device)

    return models


def _create_manual_processed_model(model_name: str, device: str) -> HookedTransformer:
    """Create manually processed model for comparison."""
    # Load raw HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_state = hf_model.state_dict()

    # Create bridge for processing
    bridge = TransformerBridge.boot_transformers(model_name, device=device)

    # Extract processed weights
    processed_weights = bridge._extract_weights_in_tl_format()

    # Create new HookedTransformer and load processed weights
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.load_state_dict(processed_weights, strict=False)

    return model


def _test_original_losses(models: Dict[str, Any], test_tokens: torch.Tensor) -> Dict[str, Any]:
    """Test that all models produce similar original losses."""
    losses = {}

    with torch.no_grad():
        for name, model in models.items():
            try:
                output = model(test_tokens)
                # Simple loss calculation (cross-entropy with shifted targets)
                targets = test_tokens[:, 1:].contiguous()
                logits = output[:, :-1, :].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                ).item()
                losses[name] = loss
                print(f"    {name}: {loss:.6f}")
            except Exception as e:
                losses[name] = f"ERROR: {e}"

    # Calculate differences
    loss_values = [v for v in losses.values() if isinstance(v, float)]
    if len(loss_values) >= 2:
        max_diff = max(loss_values) - min(loss_values)
        losses["max_difference"] = max_diff
        losses["losses_match"] = max_diff < 0.01

    return losses


def _test_ablations(models: Dict[str, Any], test_tokens: torch.Tensor) -> Dict[str, Any]:
    """Test that ablations work consistently across models."""
    ablation_results = {}
    hook_name = "blocks.0.attn.hook_v"  # Standard hook for testing

    def ablation_fn(activation, hook):
        return torch.zeros_like(activation)

    with torch.no_grad():
        for name, model in models.items():
            try:
                # Original output
                original_output = model(test_tokens)
                original_loss = torch.nn.functional.cross_entropy(
                    original_output[:, :-1, :].contiguous().view(-1, original_output.size(-1)),
                    test_tokens[:, 1:].contiguous().view(-1)
                ).item()

                # Ablated output
                ablated_output = model.run_with_hooks(test_tokens, fwd_hooks=[(hook_name, ablation_fn)])
                ablated_loss = torch.nn.functional.cross_entropy(
                    ablated_output[:, :-1, :].contiguous().view(-1, ablated_output.size(-1)),
                    test_tokens[:, 1:].contiguous().view(-1)
                ).item()

                ablation_results[name] = {
                    "original_loss": original_loss,
                    "ablated_loss": ablated_loss,
                    "difference": ablated_loss - original_loss
                }
                print(f"    {name}: original={original_loss:.6f}, ablated={ablated_loss:.6f}")

            except Exception as e:
                ablation_results[name] = {"error": str(e)}

    return ablation_results


def _test_round_trip_conversion(bridge_model: TransformerBridge, model_name: str) -> Dict[str, Any]:
    """Test round-trip conversion for processed weights."""
    try:
        # Get processed weights from bridge
        processed_weights = bridge_model._extract_weights_in_tl_format()

        # Load original HF weights for reference
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        original_hf_weights = hf_model.state_dict()

        # Run validation
        validator = RoundTripValidator(tolerance=1e-6)
        results = validator.validate_processed_weight_conversion(
            original_hf_weights, processed_weights, bridge_model.cfg, model_name
        )

        # Add sample validation for debugging
        if results["success"]:
            sample_results = validator.validate_conversion_with_sample_keys(
                processed_weights,
                validator.converter.hf_to_tlens(
                    validator.converter.tlens_to_hf(processed_weights, bridge_model.cfg, model_name),
                    bridge_model.cfg, model_name
                )
            )
            results.update(sample_results)

        return results

    except Exception as e:
        return {"error": str(e), "success": False}


def _determine_overall_success(results: Dict[str, Any]) -> bool:
    """Determine if the overall integration test passed."""
    # Check if models loaded
    if not results.get("models"):
        return False

    # Check if losses match
    loss_results = results.get("loss_comparisons", {})
    if not loss_results.get("losses_match", False):
        return False

    # Check if ablations worked
    ablation_results = results.get("ablation_tests", {})
    ablation_success = all(
        isinstance(result, dict) and "error" not in result
        for result in ablation_results.values()
    )
    if not ablation_success:
        return False

    # Check round-trip validation
    round_trip_results = results.get("round_trip_validation", {})
    if not round_trip_results.get("success", False):
        return False

    return True


if __name__ == "__main__":
    # Run the integration test
    results = run_weight_processing_integration_test("gpt2", "cpu")

    print("\n=== Final Results ===")
    print(f"Overall Success: {results['overall_success']}")

    if results["overall_success"]:
        print("✅ All weight processing and conversion tests passed!")
    else:
        print("❌ Some tests failed. Check detailed results above.")