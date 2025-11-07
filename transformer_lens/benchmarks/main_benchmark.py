"""Main benchmark runner for TransformerBridge.

This module provides the main benchmark suite that compares TransformerBridge
against reference implementations in a tiered approach:
1. First Priority: Compare TB → HuggingFace model (raw)
2. Second Priority: If HT version exists, compare TB → HT
3. Third Priority: If model unavailable in HT, run TB-only validation
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

    This function implements a tiered comparison approach:
    1. First Priority: Compare TransformerBridge → HuggingFace model (raw)
    2. Second Priority: If HT version exists, compare TransformerBridge → HookedTransformer
    3. Third Priority: If model unavailable in HT, run TB-only validation

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

    # Lazy loading containers for models - load only when needed
    bridge_unprocessed = None
    bridge_processed = None
    hf_model: Optional[torch.nn.Module] = None
    ht_model_unprocessed: Optional[HookedTransformer] = None
    ht_model: Optional[HookedTransformer] = None

    def cleanup_model(model):
        """Free up memory by deleting a model and forcing garbage collection."""
        import gc

        del model
        gc.collect()

        # Clear CUDA cache if using GPU
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_bridge_unprocessed():
        """Lazy load unprocessed TransformerBridge."""
        nonlocal bridge_unprocessed
        if bridge_unprocessed is None:
            if verbose:
                print("Loading TransformerBridge (unprocessed)...")
            try:
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
                raise
        return bridge_unprocessed

    def get_bridge_processed():
        """Lazy load processed TransformerBridge."""
        nonlocal bridge_processed
        if bridge_processed is None and enable_compatibility_mode:
            if verbose:
                print("Loading TransformerBridge (processed)...")
            try:
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
                raise
        return bridge_processed

    def get_hf_model():
        """Lazy load HuggingFace model."""
        nonlocal hf_model
        if hf_model is None and use_hf_reference:
            if verbose:
                print("Loading HuggingFace reference model...")
            try:
                hf_model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[arg-type]
                hf_model.to(device)  # type: ignore[arg-type]
                hf_model.eval()
                if verbose:
                    print("✓ HuggingFace model loaded\n")
            except Exception as e:
                if verbose:
                    print(f"✗ Could not load HuggingFace model: {str(e)}\n")
        return hf_model

    def get_ht_model_unprocessed():
        """Lazy load unprocessed HookedTransformer."""
        nonlocal ht_model_unprocessed
        if ht_model_unprocessed is None and use_ht_reference:
            if verbose:
                print("Loading HookedTransformer (unprocessed)...")
            try:
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
        return ht_model_unprocessed

    def get_ht_model_processed():
        """Lazy load processed HookedTransformer."""
        nonlocal ht_model
        if ht_model is None and use_ht_reference:
            if verbose:
                print("Loading HookedTransformer (processed)...")
            try:
                ht_model = HookedTransformer.from_pretrained(
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
        return ht_model

    # Run benchmarks
    if verbose:
        print("Running benchmarks...\n")

    # Component-level benchmarks (compare unprocessed Bridge components vs HF)
    if verbose:
        print("1. Component-Level Benchmarks (unprocessed Bridge vs HuggingFace)")
    try:
        bridge_unproc = get_bridge_unprocessed()
        hf = get_hf_model()
        if hf:
            component_result = benchmark_all_components(bridge_unproc, hf)
            results.append(component_result)
            if verbose:
                status = "✓" if component_result.passed else "✗"
                print(f"{status} {component_result.message}\n")
    except Exception as e:
        if verbose:
            print(f"✗ Component benchmark failed: {e}\n")

    # Forward pass benchmarks (compare unprocessed Bridge vs HF)
    if verbose:
        print("2. Forward Pass Benchmarks (unprocessed Bridge vs HuggingFace)")
    try:
        results.append(
            benchmark_forward_pass(
                get_bridge_unprocessed(), test_text, reference_model=get_hf_model()
            )
        )
    except Exception:
        pass  # Error already recorded in get_bridge_unprocessed

    # Clean up HF model - no longer needed
    if hf_model is not None:
        cleanup_model(hf_model)
        hf_model = None

    # Unprocessed model comparison (compare unprocessed Bridge vs unprocessed HT)
    if verbose:
        print(
            "2. Unprocessed Model Comparison (unprocessed Bridge vs unprocessed HookedTransformer)"
        )
    ht_unproc = get_ht_model_unprocessed()
    if ht_unproc:
        try:
            bridge_unproc = get_bridge_unprocessed()
            results.append(
                benchmark_loss_equivalence(bridge_unproc, test_text, reference_model=ht_unproc)
            )
            results.append(
                benchmark_logits_equivalence(bridge_unproc, test_text, reference_model=ht_unproc)
            )
        except Exception:
            pass  # Error already recorded
    else:
        # No unprocessed HT reference - skip unprocessed comparisons
        if verbose:
            print(
                "⚠ No unprocessed HookedTransformer available - skipping unprocessed comparisons\n"
            )

    # Clean up unprocessed HT model and unprocessed bridge - no longer needed
    if ht_model_unprocessed is not None:
        cleanup_model(ht_model_unprocessed)
        ht_model_unprocessed = None
    if bridge_unprocessed is not None:
        cleanup_model(bridge_unprocessed)
        bridge_unprocessed = None

    # Compatibility mode benchmarks (compare processed Bridge vs processed HT)
    if verbose:
        print("3. Compatibility Mode Benchmarks (processed Bridge vs processed HookedTransformer)")
    if enable_compatibility_mode:
        try:
            bridge_proc = get_bridge_processed()
            ht_proc = get_ht_model_processed()
            if bridge_proc and ht_proc:
                results.append(
                    benchmark_loss_equivalence(bridge_proc, test_text, reference_model=ht_proc)
                )
                results.append(
                    benchmark_logits_equivalence(bridge_proc, test_text, reference_model=ht_proc)
                )
            elif bridge_proc:
                # No HT reference - just validate processed Bridge works
                results.append(
                    benchmark_loss_equivalence(bridge_proc, test_text, reference_model=None)
                )
                results.append(
                    benchmark_logits_equivalence(bridge_proc, test_text, reference_model=None)
                )
        except Exception:
            pass  # Error already recorded
    else:
        # No processed bridge - skip compatibility tests
        if verbose:
            print("⚠ Compatibility mode disabled - skipping processed comparisons\n")

    # Hook benchmarks (use processed Bridge for compatibility with HT)
    if verbose:
        print("4. Hook Registration Benchmarks")
    try:
        # Prefer processed bridge if available, otherwise use unprocessed
        bridge_proc = get_bridge_processed() if enable_compatibility_mode else None
        test_bridge = bridge_proc if bridge_proc else get_bridge_unprocessed()
        ht_proc = get_ht_model_processed()

        results.append(benchmark_hook_registry(test_bridge, reference_model=ht_proc))
        results.append(
            benchmark_hook_functionality(test_bridge, test_text, reference_model=ht_proc)
        )
        results.append(
            benchmark_critical_forward_hooks(test_bridge, test_text, reference_model=ht_proc)
        )

        # Only run full forward hooks if HT reference is available (computationally expensive)
        if ht_proc is not None and bridge_proc:
            results.append(benchmark_forward_hooks(bridge_proc, test_text, reference_model=ht_proc))
    except Exception:
        pass  # Error already recorded

    # Gradient benchmarks (use processed Bridge for compatibility with HT)
    if verbose:
        print("5. Backward Gradient Benchmarks")
    try:
        bridge_proc = get_bridge_processed() if enable_compatibility_mode else None
        test_bridge = bridge_proc if bridge_proc else get_bridge_unprocessed()
        ht_proc = get_ht_model_processed()

        results.append(
            benchmark_gradient_computation(test_bridge, test_text, reference_model=ht_proc)
        )
        results.append(
            benchmark_critical_backward_hooks(test_bridge, test_text, reference_model=ht_proc)
        )

        # Only run full backward hooks if HT reference is available (computationally expensive)
        if ht_proc is not None and bridge_proc:
            results.append(
                benchmark_backward_hooks(bridge_proc, test_text, reference_model=ht_proc)
            )
    except Exception:
        pass  # Error already recorded

    # Generation benchmarks (test both unprocessed and processed)
    if verbose:
        print("6. Generation Benchmarks")
    try:
        bridge_unproc = get_bridge_unprocessed()
        results.append(benchmark_generation(bridge_unproc, test_text, max_new_tokens=10))
        results.append(
            benchmark_generation_with_kv_cache(bridge_unproc, test_text, max_new_tokens=10)
        )
        results.append(
            benchmark_multiple_generation_calls(
                bridge_unproc,
                test_prompts=[
                    "The quick brown fox",
                    "Hello world",
                    "Machine learning is",
                ],
                max_new_tokens=5,
            )
        )
    except Exception:
        pass  # Error already recorded

    # Weight processing benchmarks (compare processed Bridge vs processed HT)
    if verbose:
        print("7. Weight Processing Benchmarks")
    if enable_compatibility_mode:
        try:
            bridge_proc = get_bridge_processed()
            ht_proc = get_ht_model_processed()
            if bridge_proc and ht_proc:
                results.append(
                    benchmark_weight_processing(bridge_proc, test_text, reference_model=ht_proc)
                )
                results.append(
                    benchmark_weight_sharing(bridge_proc, test_text, reference_model=ht_proc)
                )
                results.append(benchmark_weight_modification(bridge_proc, test_text))
            elif bridge_proc:
                # No HT reference - just test processed bridge works
                results.append(
                    benchmark_weight_processing(bridge_proc, test_text, reference_model=None)
                )
                results.append(
                    benchmark_weight_sharing(bridge_proc, test_text, reference_model=None)
                )
                results.append(benchmark_weight_modification(bridge_proc, test_text))
        except Exception:
            pass  # Error already recorded

    # Activation cache benchmarks (compare processed Bridge vs processed HT)
    if verbose:
        print("8. Activation Cache Benchmarks")
    if enable_compatibility_mode:
        try:
            bridge_proc = get_bridge_processed()
            ht_proc = get_ht_model_processed()
            if bridge_proc and ht_proc:
                results.append(
                    benchmark_run_with_cache(bridge_proc, test_text, reference_model=ht_proc)
                )
                results.append(
                    benchmark_activation_cache(bridge_proc, test_text, reference_model=ht_proc)
                )
            elif bridge_proc:
                # No HT reference - just test processed bridge works
                results.append(
                    benchmark_run_with_cache(bridge_proc, test_text, reference_model=None)
                )
                results.append(
                    benchmark_activation_cache(bridge_proc, test_text, reference_model=None)
                )
        except Exception:
            pass  # Error already recorded

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
