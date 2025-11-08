"""Main benchmark runner for TransformerBridge.

This module provides the main benchmark suite that compares TransformerBridge
against reference implementations in an optimized 3-phase approach:
Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing
"""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer
from transformer_lens.benchmarks.activation_cache import (
    benchmark_activation_cache,
    benchmark_run_with_cache,
)
from transformer_lens.benchmarks.backward_gradients import (
    benchmark_backward_hooks,
    benchmark_critical_backward_hooks,
    benchmark_gradient_computation,
)
from transformer_lens.benchmarks.component_benchmark import benchmark_all_components
from transformer_lens.benchmarks.forward_pass import (
    benchmark_forward_pass,
    benchmark_logits_equivalence,
    benchmark_loss_equivalence,
)
from transformer_lens.benchmarks.generation import (
    benchmark_generation,
    benchmark_generation_with_kv_cache,
    benchmark_multiple_generation_calls,
)
from transformer_lens.benchmarks.hook_registration import (
    benchmark_critical_forward_hooks,
    benchmark_forward_hooks,
    benchmark_hook_functionality,
    benchmark_hook_registry,
)
from transformer_lens.benchmarks.utils import BenchmarkResult, format_results
from transformer_lens.benchmarks.weight_processing import (
    benchmark_weight_modification,
    benchmark_weight_processing,
    benchmark_weight_sharing,
)
from transformer_lens.model_bridge import TransformerBridge


def run_benchmark_suite(
    model_name: str,
    device: str = "cpu",
    test_text: Optional[str] = None,
    use_hf_reference: bool = True,
    use_ht_reference: bool = True,
    enable_compatibility_mode: bool = True,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Run comprehensive benchmark suite for TransformerBridge.

    This function implements an optimized 3-phase approach to minimize model reloading:
    Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
    Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
    Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing

    Args:
        model_name: Name of the model to benchmark (e.g., "gpt2")
        device: Device to run on ("cpu" or "cuda")
        test_text: Optional test text (default: standard test prompt)
        use_hf_reference: Whether to compare against HuggingFace model
        use_ht_reference: Whether to compare against HookedTransformer
        enable_compatibility_mode: Whether to enable compatibility mode on bridge
        verbose: Whether to print results to console

    Returns:
        List of BenchmarkResult objects
    """
    if test_text is None:
        test_text = (
            "Natural language processing tasks, such as question answering, "
            "machine translation, reading comprehension, and summarization, "
            "are typically approached with supervised learning."
        )

    results: List[BenchmarkResult] = []

    if verbose:
        print(f"\n{'='*80}")
        print(f"Running TransformerBridge Benchmark Suite")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"{'='*80}\n")

    def cleanup_model(model, model_name_str: str):
        """Free up memory by deleting a model and forcing garbage collection."""
        import gc

        if verbose:
            print(f"Cleaning up {model_name_str}...")
        del model
        gc.collect()

        # Clear CUDA cache if using GPU
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================================
    # PHASE 1: HuggingFace + Bridge (unprocessed)
    # ========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 1: HuggingFace + TransformerBridge (unprocessed)")
        print(f"{'='*80}\n")

    bridge_unprocessed = None
    hf_model = None

    # Load models for Phase 1
    if use_hf_reference:
        try:
            if verbose:
                print("Loading HuggingFace reference model...")
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[arg-type]
            hf_model.to(device)  # type: ignore[arg-type]
            hf_model.eval()
            if verbose:
                print("✓ HuggingFace model loaded\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load HuggingFace model: {str(e)}\n")

    try:
        if verbose:
            print("Loading TransformerBridge (unprocessed)...")
        bridge_unprocessed = TransformerBridge.boot_transformers(model_name, device=device)  # type: ignore[attr-defined]
        if verbose:
            print("✓ TransformerBridge loaded (unprocessed)\n")
    except Exception as e:
        from transformer_lens.benchmarks.utils import BenchmarkSeverity

        results.append(
            BenchmarkResult(
                name="load_bridge_unprocessed",
                severity=BenchmarkSeverity.ERROR,
                message=f"Failed to load unprocessed TransformerBridge: {str(e)}",
                passed=False,
            )
        )
        if verbose:
            print(f"✗ Failed to load TransformerBridge: {str(e)}\n")
        return results

    # Run Phase 1 benchmarks
    if hf_model and bridge_unprocessed:
        if verbose:
            print("Running Phase 1 benchmarks...\n")

        # Component-level benchmarks
        if verbose:
            print("1. Component-Level Benchmarks")
        try:
            component_result = benchmark_all_components(bridge_unprocessed, hf_model)
            results.append(component_result)
            if verbose:
                status = "✓" if component_result.passed else "✗"
                print(f"{status} {component_result.message}\n")
        except Exception as e:
            if verbose:
                print(f"✗ Component benchmark failed: {e}\n")

        # Forward pass benchmarks
        if verbose:
            print("2. Forward Pass Benchmarks")
        try:
            results.append(
                benchmark_forward_pass(bridge_unprocessed, test_text, reference_model=hf_model)
            )
        except Exception as e:
            if verbose:
                print(f"✗ Forward pass benchmark failed: {e}\n")

    # Clean up HF model - no longer needed
    if hf_model is not None:
        cleanup_model(hf_model, "HuggingFace model")
        hf_model = None

    # ========================================================================
    # PHASE 2: Bridge (unprocessed) + HookedTransformer (unprocessed)
    # ========================================================================
    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 2: TransformerBridge (unprocessed) + HookedTransformer (unprocessed)")
        print(f"{'='*80}\n")

    ht_model_unprocessed = None

    # Load HookedTransformer (unprocessed) for Phase 2
    if use_ht_reference:
        try:
            if verbose:
                print("Loading HookedTransformer (unprocessed)...")
            ht_model_unprocessed = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
                fold_value_biases=False,
                refactor_factored_attn_matrices=False,
            )
            if verbose:
                print("✓ HookedTransformer loaded (unprocessed)\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load unprocessed HookedTransformer: {str(e)}\n")

    # Run Phase 2 benchmarks
    if bridge_unprocessed:
        if verbose:
            print("Running Phase 2 benchmarks...\n")

        # Unprocessed model comparison
        if ht_model_unprocessed:
            if verbose:
                print("1. Unprocessed Model Equivalence")
            try:
                results.append(
                    benchmark_loss_equivalence(
                        bridge_unprocessed, test_text, reference_model=ht_model_unprocessed
                    )
                )
                results.append(
                    benchmark_logits_equivalence(
                        bridge_unprocessed, test_text, reference_model=ht_model_unprocessed
                    )
                )
            except Exception as e:
                if verbose:
                    print(f"✗ Unprocessed equivalence benchmark failed: {e}\n")
        else:
            if verbose:
                print(
                    "⚠ No unprocessed HookedTransformer available - skipping unprocessed comparisons\n"
                )

        # Generation benchmarks (unprocessed only)
        if verbose:
            print("2. Generation Benchmarks (unprocessed)")
        try:
            results.append(benchmark_generation(bridge_unprocessed, test_text, max_new_tokens=10))
            results.append(
                benchmark_generation_with_kv_cache(bridge_unprocessed, test_text, max_new_tokens=10)
            )
            results.append(
                benchmark_multiple_generation_calls(
                    bridge_unprocessed,
                    test_prompts=[
                        "The quick brown fox",
                        "Hello world",
                        "Machine learning is",
                    ],
                    max_new_tokens=5,
                )
            )
        except Exception as e:
            if verbose:
                print(f"✗ Generation benchmark failed: {e}\n")

    # Clean up unprocessed models - no longer needed
    if ht_model_unprocessed is not None:
        cleanup_model(ht_model_unprocessed, "HookedTransformer (unprocessed)")
        ht_model_unprocessed = None
    if bridge_unprocessed is not None:
        cleanup_model(bridge_unprocessed, "TransformerBridge (unprocessed)")
        bridge_unprocessed = None

    # ========================================================================
    # PHASE 3: Bridge (processed) + HookedTransformer (processed)
    # ========================================================================
    if not enable_compatibility_mode:
        if verbose:
            print("\n⚠ Compatibility mode disabled - skipping Phase 3\n")
        if verbose:
            print("\n" + format_results(results))
        return results

    if verbose:
        print(f"\n{'='*80}")
        print("PHASE 3: TransformerBridge (processed) + HookedTransformer (processed)")
        print(f"{'='*80}\n")

    bridge_processed = None
    ht_model_processed = None

    # Load processed models for Phase 3
    try:
        if verbose:
            print("Loading TransformerBridge (processed)...")
        bridge_processed = TransformerBridge.boot_transformers(model_name, device=device)  # type: ignore[attr-defined]
        bridge_processed.enable_compatibility_mode(disable_warnings=True)
        if verbose:
            print("✓ TransformerBridge compatibility mode enabled (processed)\n")
    except Exception as e:
        from transformer_lens.benchmarks.utils import BenchmarkSeverity

        results.append(
            BenchmarkResult(
                name="load_bridge_processed",
                severity=BenchmarkSeverity.ERROR,
                message=f"Failed to load processed TransformerBridge: {str(e)}",
                passed=False,
            )
        )
        if verbose:
            print(f"✗ Failed to load processed TransformerBridge: {str(e)}\n")
        if verbose:
            print("\n" + format_results(results))
        return results

    if use_ht_reference:
        try:
            if verbose:
                print("Loading HookedTransformer (processed)...")
            ht_model_processed = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,
            )
            if verbose:
                print("✓ HookedTransformer loaded (processed)\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load processed HookedTransformer: {str(e)}\n")

    # Run Phase 3 benchmarks
    if bridge_processed:
        if verbose:
            print("Running Phase 3 benchmarks...\n")

        # Processed model equivalence
        if verbose:
            print("1. Processed Model Equivalence")
        try:
            results.append(
                benchmark_loss_equivalence(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
            results.append(
                benchmark_logits_equivalence(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
        except Exception as e:
            if verbose:
                print(f"✗ Processed equivalence benchmark failed: {e}\n")

        # Hook registration benchmarks
        if verbose:
            print("2. Hook Registration Benchmarks")
        try:
            results.append(
                benchmark_hook_registry(bridge_processed, reference_model=ht_model_processed)
            )
            results.append(
                benchmark_hook_functionality(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
            results.append(
                benchmark_critical_forward_hooks(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )

            # Only run full forward hooks if HT reference is available (computationally expensive)
            if ht_model_processed is not None:
                results.append(
                    benchmark_forward_hooks(
                        bridge_processed, test_text, reference_model=ht_model_processed
                    )
                )
        except Exception as e:
            if verbose:
                print(f"✗ Hook benchmark failed: {e}\n")

        # Gradient benchmarks
        if verbose:
            print("3. Backward Gradient Benchmarks")
        try:
            results.append(
                benchmark_gradient_computation(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
            results.append(
                benchmark_critical_backward_hooks(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )

            # Only run full backward hooks if HT reference is available (computationally expensive)
            if ht_model_processed is not None:
                results.append(
                    benchmark_backward_hooks(
                        bridge_processed, test_text, reference_model=ht_model_processed
                    )
                )
        except Exception as e:
            if verbose:
                print(f"✗ Gradient benchmark failed: {e}\n")

        # Weight processing benchmarks
        if verbose:
            print("4. Weight Processing Benchmarks")
        try:
            results.append(
                benchmark_weight_processing(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
            results.append(
                benchmark_weight_sharing(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
            results.append(benchmark_weight_modification(bridge_processed, test_text))
        except Exception as e:
            if verbose:
                print(f"✗ Weight processing benchmark failed: {e}\n")

        # Activation cache benchmarks
        if verbose:
            print("5. Activation Cache Benchmarks")
        try:
            results.append(
                benchmark_run_with_cache(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
            results.append(
                benchmark_activation_cache(
                    bridge_processed, test_text, reference_model=ht_model_processed
                )
            )
        except Exception as e:
            if verbose:
                print(f"✗ Activation cache benchmark failed: {e}\n")

    # Print results
    if verbose:
        print("\n" + format_results(results))

    return results


def main():
    """Run benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run TransformerBridge benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name to benchmark (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--no-hf-reference",
        action="store_true",
        help="Disable HuggingFace reference comparison",
    )
    parser.add_argument(
        "--no-ht-reference",
        action="store_true",
        help="Disable HookedTransformer reference comparison",
    )
    parser.add_argument(
        "--no-compat",
        action="store_true",
        help="Disable compatibility mode",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    run_benchmark_suite(
        model_name=args.model,
        device=args.device,
        use_hf_reference=not args.no_hf_reference,
        use_ht_reference=not args.no_ht_reference,
        enable_compatibility_mode=not args.no_compat,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
