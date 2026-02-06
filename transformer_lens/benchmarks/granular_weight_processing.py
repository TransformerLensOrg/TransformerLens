"""Granular weight processing benchmarks.

This module provides detailed benchmarks that test each weight processing operation
individually and in combination to isolate which processing steps cause issues.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch

from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity


@dataclass
class WeightProcessingConfig:
    """Configuration for a specific weight processing test."""

    name: str
    fold_ln: bool
    center_writing_weights: bool
    center_unembed: bool
    fold_value_biases: bool
    refactor_factored_attn_matrices: bool

    def __str__(self) -> str:
        """Get a short string representation."""
        flags = []
        if self.fold_ln:
            flags.append("fold_ln")
        if self.center_writing_weights:
            flags.append("center_weights")
        if self.center_unembed:
            flags.append("center_unembed")
        if self.fold_value_biases:
            flags.append("fold_value_bias")
        if self.refactor_factored_attn_matrices:
            flags.append("refactor_attn")
        return "+".join(flags) if flags else "none"


# Phase 4: Individual weight processing operations (test each flag in isolation)
# NOTE: Centering operations (center_writing_weights, center_unembed) require fold_ln=True
# as they rely on LayerNorm ignoring the mean. Testing them without fold_ln produces
# invalid/misleading results, so we test them with fold_ln enabled.
INDIVIDUAL_CONFIGS = [
    # Test fold_ln alone
    WeightProcessingConfig(
        name="only_fold_ln",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    # Test center_writing_weights (requires fold_ln)
    WeightProcessingConfig(
        name="only_center_weights",
        fold_ln=False,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    # Test center_unembed (requires fold_ln)
    WeightProcessingConfig(
        name="only_center_unembed",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=True,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    # Test fold_value_biases alone
    WeightProcessingConfig(
        name="only_fold_value_biases",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
]

# Phase 5: Combinations of weight processing operations
COMBINATION_CONFIGS = [
    # Two-way combinations (fold_ln + one other)
    WeightProcessingConfig(
        name="fold_ln+center_weights",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="fold_ln+center_unembed",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=True,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="fold_ln+fold_value_biases",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
    # Three-way combinations (commonly used together)
    WeightProcessingConfig(
        name="fold_ln+center_weights+center_unembed",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="fold_ln+center_weights+fold_value_biases",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=False,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
    WeightProcessingConfig(
        name="fold_ln+center_unembed+fold_value_biases",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
    # Standard configuration (all enabled except refactor)
    WeightProcessingConfig(
        name="standard_all",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    ),
]

# Experimental configurations that test refactor_factored_attn_matrices
# These are only run when explicitly requested via include_refactor_tests=True
REFACTOR_ATTN_CONFIGS = [
    WeightProcessingConfig(
        name="only_refactor_attn",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=True,
    ),
    WeightProcessingConfig(
        name="fold_ln+refactor_attn",
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=True,
    ),
    WeightProcessingConfig(
        name="all_with_refactor",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=True,
    ),
]


def run_granular_weight_processing_benchmarks(
    model_name: str,
    device: str,
    test_text: str,
    verbose: bool = True,
    include_refactor_tests: bool = False,
    phase: int | None = None,
) -> Dict[str, List[BenchmarkResult]]:
    """Run benchmarks with each weight processing configuration.

    This function tests each weight processing flag individually (Phase 4) and
    in combination (Phase 5) to identify which specific processing steps cause issues.

    Args:
        model_name: Name of the model to benchmark
        device: Device to run on ("cpu" or "cuda")
        test_text: Test text for generation/inference
        verbose: Whether to print detailed output
        include_refactor_tests: Whether to include experimental refactor_factored_attn_matrices tests
        phase: Optional phase number (4 for individual, 5 for combinations). If None, runs both.

    Returns:
        Dictionary mapping config name to list of benchmark results
    """
    from transformer_lens import HookedTransformer
    from transformer_lens.benchmarks.forward_pass import (
        benchmark_logits_equivalence,
        benchmark_loss_equivalence,
    )
    from transformer_lens.benchmarks.hook_registration import (
        benchmark_critical_forward_hooks,
        benchmark_forward_hooks,
        benchmark_hook_functionality,
    )
    from transformer_lens.model_bridge.bridge import TransformerBridge

    all_results: Dict[str, List[BenchmarkResult]] = {}

    # Check if HookedTransformer is available for this model before running any tests
    ht_available = False
    try:
        test_ht = HookedTransformer.from_pretrained(model_name, device=device)
        ht_available = True
        del test_ht
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        if verbose:
            print("\n" + "=" * 80)
            print("GRANULAR WEIGHT PROCESSING BENCHMARKS")
            print(f"Model: {model_name}")
            print("=" * 80)
            print(f"⚠ HookedTransformer not available for {model_name}: {str(e)}")
            print(
                "⚠ Skipping granular weight processing tests (requires HookedTransformer reference)"
            )
            print("=" * 80 + "\n")

        # Return a single SKIPPED result for all tests
        skip_result = BenchmarkResult(
            name="granular_weight_processing",
            passed=True,
            severity=BenchmarkSeverity.SKIPPED,
            message=f"HookedTransformer not available for {model_name} - tests skipped",
            details={"reason": "HookedTransformer unavailable", "error": str(e)},
        )
        all_results["skipped"] = [skip_result]
        return all_results

    # Determine which configurations to test based on phase
    configs_to_test = []
    phase_name = ""

    if phase is None or phase == 4:
        configs_to_test.extend(INDIVIDUAL_CONFIGS)
        if phase == 4:
            phase_name = "PHASE 4: Individual Weight Processing Flags"

    if phase is None or phase == 5:
        configs_to_test.extend(COMBINATION_CONFIGS)
        if phase == 5:
            phase_name = "PHASE 5: Combined Weight Processing Flags"

    if phase is None:
        phase_name = "PHASE 4 & 5: Granular Weight Processing"

    if include_refactor_tests:
        configs_to_test.extend(REFACTOR_ATTN_CONFIGS)

    if verbose:
        print("\n" + "=" * 80)
        print(phase_name)
        print(f"Model: {model_name}")
        print(f"Testing {len(configs_to_test)} configurations")
        if phase is None or phase == 4:
            print(f"  Individual flags: {len(INDIVIDUAL_CONFIGS)}")
        if phase is None or phase == 5:
            print(f"  Combinations: {len(COMBINATION_CONFIGS)}")
        if include_refactor_tests:
            print(f"  Refactor tests: {len(REFACTOR_ATTN_CONFIGS)}")
        print("=" * 80)

    for config in configs_to_test:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {config.name}")
            print(f"Flags: {config}")
            print(f"{'='*80}\n")

        results: List[BenchmarkResult] = []

        try:
            # Load HookedTransformer reference with same processing
            if verbose:
                print(f"Loading HookedTransformer ({config})...")
            ht_ref = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=config.fold_ln,
                center_writing_weights=config.center_writing_weights,
                center_unembed=config.center_unembed,
                fold_value_biases=config.fold_value_biases,
                refactor_factored_attn_matrices=config.refactor_factored_attn_matrices,
            )

            # Load TransformerBridge and apply same processing
            if verbose:
                print(f"Loading TransformerBridge ({config})...")
            bridge = TransformerBridge.boot_transformers(model_name, device=device)
            bridge.enable_compatibility_mode(
                disable_warnings=True,
                fold_ln=config.fold_ln,
                center_writing_weights=config.center_writing_weights,
                center_unembed=config.center_unembed,
                fold_value_biases=config.fold_value_biases,
                refactor_factored_attn_matrices=config.refactor_factored_attn_matrices,
            )

            # Run core benchmarks
            if verbose:
                print("Running benchmarks...\n")

            # Logits/loss equivalence
            logits_result = benchmark_logits_equivalence(bridge, test_text, reference_model=ht_ref)
            results.append(logits_result)
            if verbose:
                status = "🟢 [PASS]" if logits_result.passed else "🔴 [FAIL]"
                print(f"{status} logits_equivalence: {logits_result.message}")
                if logits_result.details:
                    for key, value in logits_result.details.items():
                        print(f"  {key}: {value}")

            loss_result = benchmark_loss_equivalence(bridge, test_text, reference_model=ht_ref)
            results.append(loss_result)
            if verbose:
                status = "🟢 [PASS]" if loss_result.passed else "🔴 [FAIL]"
                print(f"{status} loss_equivalence: {loss_result.message}")
                if loss_result.details:
                    for key, value in loss_result.details.items():
                        print(f"  {key}: {value}")

            # Hook functionality
            hook_func_result = benchmark_hook_functionality(
                bridge, test_text, reference_model=ht_ref
            )
            results.append(hook_func_result)
            if verbose:
                status = "🟢 [PASS]" if hook_func_result.passed else "🔴 [FAIL]"
                print(f"{status} hook_functionality: {hook_func_result.message}")
                if hook_func_result.details:
                    for key, value in hook_func_result.details.items():
                        print(f"  {key}: {value}")

            critical_hooks_result = benchmark_critical_forward_hooks(
                bridge, test_text, reference_model=ht_ref
            )
            results.append(critical_hooks_result)
            if verbose:
                status = "🟢 [PASS]" if critical_hooks_result.passed else "🔴 [FAIL]"
                print(f"{status} critical_forward_hooks: {critical_hooks_result.message}")
                if critical_hooks_result.details:
                    for key, value in critical_hooks_result.details.items():
                        print(f"  {key}: {value}")

            forward_hooks_result = benchmark_forward_hooks(
                bridge, test_text, reference_model=ht_ref
            )
            results.append(forward_hooks_result)
            if verbose:
                status = "🟢 [PASS]" if forward_hooks_result.passed else "🔴 [FAIL]"
                print(f"{status} forward_hooks: {forward_hooks_result.message}")
                if forward_hooks_result.details:
                    for key, value in forward_hooks_result.details.items():
                        print(f"  {key}: {value}")

            # Clean up
            del bridge
            del ht_ref
            # Force garbage collection (multiple passes to break circular references)
            import gc

            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            # Record failure
            results.append(
                BenchmarkResult(
                    name=f"{config.name}_error",
                    passed=False,
                    severity=BenchmarkSeverity.ERROR,
                    message=f"Failed to run configuration: {str(e)}",
                    details={"error": str(e), "config": str(config)},
                )
            )

        # Store results
        all_results[config.name] = results

        # Print summary for this config
        if verbose:
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            print(f"\n{config.name}: {passed}/{total} passed")

    # Print overall summary
    if verbose:
        print("\n" + "=" * 80)
        print("GRANULAR WEIGHT PROCESSING SUMMARY")
        print("=" * 80)
        for config_name, results in all_results.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            status = "✅" if passed == total else "❌" if passed == 0 else "⚠️"
            print(f"{status} {config_name}: {passed}/{total} passed")

    return all_results
