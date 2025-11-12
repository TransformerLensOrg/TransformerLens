"""Main benchmark runner for TransformerBridge.

This module provides the main benchmark suite that compares TransformerBridge
against reference implementations in an optimized 3-phase approach:
Phase 1: HF + Bridge (unprocessed) - Compare against raw HuggingFace model
Phase 2: Bridge (unprocessed) + HT (unprocessed) - Compare unprocessed models
Phase 3: Bridge (processed) + HT (processed) - Full compatibility mode testing
"""

import gc
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
    benchmark_attention_output_centering,
    benchmark_layer_norm_folding,
    benchmark_mlp_output_centering,
    benchmark_no_nan_inf,
    benchmark_unembed_centering,
    benchmark_value_bias_folding,
    benchmark_weight_magnitudes,
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
    track_memory: bool = False,
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
        track_memory: Whether to track and report memory usage (requires psutil)

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

    # Memory tracking setup
    memory_tracker = None
    if track_memory:
        try:
            import psutil

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            def get_memory_mb():
                return process.memory_info().rss / 1024 / 1024

            memory_tracker = {"initial": initial_memory, "checkpoints": []}
            if verbose:
                print(f"Memory tracking enabled (initial: {initial_memory:.1f} MB)")
        except ImportError:
            if verbose:
                print("⚠ psutil not available - memory tracking disabled")
            track_memory = False

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

        # Track memory before cleanup
        if track_memory and memory_tracker is not None:
            memory_before = get_memory_mb()

        # Explicitly remove all hooks to prevent memory leaks
        if hasattr(model, "modules"):
            try:
                for module in model.modules():
                    # Clear PyTorch hooks
                    if hasattr(module, "_forward_hooks"):
                        module._forward_hooks.clear()
                    if hasattr(module, "_backward_hooks"):
                        module._backward_hooks.clear()
                    if hasattr(module, "_forward_pre_hooks"):
                        module._forward_pre_hooks.clear()
                    if hasattr(module, "_backward_pre_hooks"):
                        module._backward_pre_hooks.clear()
                    if hasattr(module, "_state_dict_hooks"):
                        module._state_dict_hooks.clear()
                    if hasattr(module, "_state_dict_pre_hooks"):
                        module._state_dict_pre_hooks.clear()
                    if hasattr(module, "_load_state_dict_pre_hooks"):
                        module._load_state_dict_pre_hooks.clear()
                    if hasattr(module, "_load_state_dict_post_hooks"):
                        module._load_state_dict_post_hooks.clear()

                    # Clear TransformerLens-specific hooks
                    if hasattr(module, "remove_all_hooks"):
                        module.remove_all_hooks()
            except Exception:
                # If hook cleanup fails, continue anyway
                pass

        # Clear top-level hooks
        if hasattr(model, "_forward_hooks"):
            model._forward_hooks.clear()
        if hasattr(model, "_backward_hooks"):
            model._backward_hooks.clear()
        if hasattr(model, "_forward_pre_hooks"):
            model._forward_pre_hooks.clear()

        # OPTIMIZATION: Break circular references more aggressively
        # Clear all submodule references to help GC
        if hasattr(model, "_modules"):
            # Clear each submodule's __dict__ to break circular references
            for name, submodule in list(model._modules.items()):
                if submodule is not None:
                    # Clear submodule hooks
                    if hasattr(submodule, "_forward_hooks"):
                        submodule._forward_hooks.clear()
                    if hasattr(submodule, "_backward_hooks"):
                        submodule._backward_hooks.clear()
                    # Break reference
                    model._modules[name] = None
            model._modules.clear()

        # Clear parameters dict
        if hasattr(model, "_parameters"):
            model._parameters.clear()

        # Clear buffers dict
        if hasattr(model, "_buffers"):
            model._buffers.clear()

        del model

        # Aggressive garbage collection (multiple passes to break circular references)
        for _ in range(3):
            gc.collect()

        # Clear CUDA cache if using GPU
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Track memory after cleanup
        if track_memory and memory_tracker is not None:
            memory_after = get_memory_mb()
            freed_mb = memory_before - memory_after
            memory_tracker["checkpoints"].append(
                {
                    "label": f"Cleanup: {model_name_str}",
                    "memory_mb": memory_after,
                    "freed_mb": freed_mb,
                }
            )
            if verbose and freed_mb > 0:
                print(f"  Freed {freed_mb:.1f} MB")

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
        # Detect dtype from HF model if available, otherwise use float32
        bridge_dtype = torch.float32
        if hf_model is not None:
            try:
                bridge_dtype = next(hf_model.parameters()).dtype
            except StopIteration:
                pass
        bridge_unprocessed = TransformerBridge.boot_transformers(model_name, device=device, dtype=bridge_dtype)  # type: ignore[attr-defined]
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

    # OPTIMIZATION: Run generation benchmarks first (only bridge in memory)
    # Then cleanup bridge before loading HT to reduce peak memory
    if bridge_unprocessed:
        if verbose:
            print("Running Phase 2 benchmarks...\n")

        # Generation benchmarks (unprocessed only) - RUN FIRST
        if verbose:
            print("1. Generation Benchmarks (unprocessed)")
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
            gc.collect()  # Force cleanup after generation benchmarks
        except Exception as e:
            if verbose:
                print(f"✗ Generation benchmark failed: {e}\n")

    # Load HookedTransformer for comparison (after generation benchmarks)
    ht_model_unprocessed = None
    if use_ht_reference:
        try:
            if verbose:
                print("Loading HookedTransformer (unprocessed) for comparison...")
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

    # Run comparison benchmarks
    if bridge_unprocessed:
        # Unprocessed model comparison
        if ht_model_unprocessed:
            if verbose:
                print("2. Unprocessed Model Equivalence")
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
                gc.collect()  # Force cleanup after equivalence benchmarks
            except Exception as e:
                if verbose:
                    print(f"✗ Unprocessed equivalence benchmark failed: {e}\n")
        else:
            # Skip unprocessed comparison tests when HT is not available
            from transformer_lens.benchmarks.utils import BenchmarkSeverity

            if verbose:
                print(
                    "⚠ No unprocessed HookedTransformer available - skipping unprocessed comparisons\n"
                )
            results.append(
                BenchmarkResult(
                    name="loss_equivalence",
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )
            results.append(
                BenchmarkResult(
                    name="logits_equivalence",
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )

        # Generation benchmarks already run above (before loading HT)

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
        # Use same dtype detection as Phase 1
        bridge_dtype = torch.float32
        if hf_model is not None:
            try:
                bridge_dtype = next(hf_model.parameters()).dtype
            except StopIteration:
                pass
        bridge_processed = TransformerBridge.boot_transformers(model_name, device=device, dtype=bridge_dtype)  # type: ignore[attr-defined]
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

        # Check if HT is available for Phase 3 tests
        ht_available = ht_model_processed is not None

        # Processed model equivalence
        if verbose:
            print("1. Processed Model Equivalence")
        if ht_available:
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
                gc.collect()  # Force cleanup after equivalence benchmarks
            except Exception as e:
                if verbose:
                    print(f"✗ Processed equivalence benchmark failed: {e}\n")
        else:
            from transformer_lens.benchmarks.utils import BenchmarkSeverity

            if verbose:
                print("⏭️ Skipped (no HookedTransformer reference)\n")
            results.append(
                BenchmarkResult(
                    name="loss_equivalence",
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )
            results.append(
                BenchmarkResult(
                    name="logits_equivalence",
                    severity=BenchmarkSeverity.SKIPPED,
                    message="Skipped (HookedTransformer not available for this model)",
                    passed=True,
                )
            )

        # Hook registration benchmarks
        if verbose:
            print("2. Hook Registration Benchmarks")
        if ht_available:
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
                results.append(
                    benchmark_forward_hooks(
                        bridge_processed, test_text, reference_model=ht_model_processed
                    )
                )
                # Reset hooks to prevent handle leaks
                if hasattr(bridge_processed, "reset_hooks"):
                    bridge_processed.reset_hooks()
                if ht_model_processed is not None and hasattr(ht_model_processed, "reset_hooks"):
                    ht_model_processed.reset_hooks()
                gc.collect()  # Force cleanup after hook benchmarks
            except Exception as e:
                if verbose:
                    print(f"✗ Hook benchmark failed: {e}\n")
        else:
            from transformer_lens.benchmarks.utils import BenchmarkSeverity

            if verbose:
                print("⏭️ Skipped (no HookedTransformer reference)\n")
            for benchmark_name in [
                "hook_registry",
                "hook_functionality",
                "critical_forward_hooks",
                "forward_hooks",
            ]:
                results.append(
                    BenchmarkResult(
                        name=benchmark_name,
                        severity=BenchmarkSeverity.SKIPPED,
                        message="Skipped (HookedTransformer not available for this model)",
                        passed=True,
                    )
                )

        # Gradient benchmarks
        if verbose:
            print("3. Backward Gradient Benchmarks")
        if ht_available:
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
                results.append(
                    benchmark_backward_hooks(
                        bridge_processed, test_text, reference_model=ht_model_processed
                    )
                )
                # Reset hooks to prevent handle leaks
                if hasattr(bridge_processed, "reset_hooks"):
                    bridge_processed.reset_hooks()
                if ht_model_processed is not None and hasattr(ht_model_processed, "reset_hooks"):
                    ht_model_processed.reset_hooks()
                gc.collect()  # Force cleanup after gradient benchmarks
            except Exception as e:
                if verbose:
                    print(f"✗ Gradient benchmark failed: {e}\n")
        else:
            from transformer_lens.benchmarks.utils import BenchmarkSeverity

            if verbose:
                print("⏭️ Skipped (no HookedTransformer reference)\n")
            for benchmark_name in [
                "gradient_computation",
                "critical_backward_hooks",
                "backward_hooks",
            ]:
                results.append(
                    BenchmarkResult(
                        name=benchmark_name,
                        severity=BenchmarkSeverity.SKIPPED,
                        message="Skipped (HookedTransformer not available for this model)",
                        passed=True,
                    )
                )

        # Weight processing benchmarks
        if verbose:
            print("4. Weight Processing Benchmarks")
        try:
            if ht_available:
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
            else:
                from transformer_lens.benchmarks.utils import BenchmarkSeverity

                if verbose:
                    print("⏭️ weight_processing and weight_sharing skipped (no HT reference)")
                for benchmark_name in ["weight_processing", "weight_sharing"]:
                    results.append(
                        BenchmarkResult(
                            name=benchmark_name,
                            severity=BenchmarkSeverity.SKIPPED,
                            message="Skipped (HookedTransformer not available for this model)",
                            passed=True,
                        )
                    )

            # weight_modification doesn't need reference model
            results.append(benchmark_weight_modification(bridge_processed, test_text))

            # Detailed weight processing validation benchmarks (don't need reference model)
            results.append(benchmark_layer_norm_folding(bridge_processed, test_text))
            results.append(benchmark_attention_output_centering(bridge_processed, test_text))
            results.append(benchmark_mlp_output_centering(bridge_processed, test_text))
            results.append(benchmark_unembed_centering(bridge_processed, test_text))
            results.append(benchmark_value_bias_folding(bridge_processed, test_text))
            results.append(benchmark_no_nan_inf(bridge_processed, test_text))
            results.append(benchmark_weight_magnitudes(bridge_processed, test_text))
            gc.collect()  # Force cleanup after weight processing benchmarks
        except Exception as e:
            if verbose:
                print(f"✗ Weight processing benchmark failed: {e}\n")

        # Activation cache benchmarks
        if verbose:
            print("5. Activation Cache Benchmarks")
        if ht_available:
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
                # Reset hooks to prevent handle leaks
                if hasattr(bridge_processed, "reset_hooks"):
                    bridge_processed.reset_hooks()
                if ht_model_processed is not None and hasattr(ht_model_processed, "reset_hooks"):
                    ht_model_processed.reset_hooks()
                gc.collect()  # Force cleanup after cache benchmarks
            except Exception as e:
                if verbose:
                    print(f"✗ Activation cache benchmark failed: {e}\n")
        else:
            from transformer_lens.benchmarks.utils import BenchmarkSeverity

            if verbose:
                print("⏭️ Skipped (no HookedTransformer reference)\n")
            for benchmark_name in ["run_with_cache", "activation_cache"]:
                results.append(
                    BenchmarkResult(
                        name=benchmark_name,
                        severity=BenchmarkSeverity.SKIPPED,
                        message="Skipped (HookedTransformer not available for this model)",
                        passed=True,
                    )
                )

    # Print results
    if verbose:
        print("\n" + format_results(results))

    # Print memory summary
    if track_memory and memory_tracker is not None:
        final_memory = get_memory_mb()
        total_increase = final_memory - memory_tracker["initial"]

        if verbose:
            print("\n" + "=" * 80)
            print("MEMORY USAGE SUMMARY")
            print("=" * 80)
            print(f"Initial memory:  {memory_tracker['initial']:>8.1f} MB")
            print(f"Final memory:    {final_memory:>8.1f} MB")
            print(f"Net increase:    {total_increase:>+8.1f} MB")

            if memory_tracker["checkpoints"]:
                print("\nCleanup operations:")
                for cp in memory_tracker["checkpoints"]:
                    if cp.get("freed_mb", 0) > 0:
                        print(
                            f"  {cp['label']:<40} freed {cp['freed_mb']:>7.1f} MB "
                            f"(after: {cp['memory_mb']:.1f} MB)"
                        )
            print("=" * 80)

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
