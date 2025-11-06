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

    # Load TransformerBridge (without processing for raw HF comparison)
    if verbose:
        print("Loading TransformerBridge...")
    try:
        bridge_unprocessed = TransformerBridge.boot_transformers(model_name, device=device)  # type: ignore[attr-defined]
        if verbose:
            print("✓ TransformerBridge loaded (unprocessed)\n")

        # Also create a processed version for compatibility mode testing
        bridge_processed = None
        if enable_compatibility_mode:
            bridge_processed = TransformerBridge.boot_transformers(model_name, device=device)  # type: ignore[attr-defined]
            bridge_processed.enable_compatibility_mode(disable_warnings=True)
            if verbose:
                print("✓ TransformerBridge compatibility mode enabled (processed)\n")

        # For backward compatibility, use processed version as default if enabled
        bridge = bridge_processed if bridge_processed else bridge_unprocessed
    except Exception as e:
        from transformer_lens.benchmarks.utils import BenchmarkSeverity

        results.append(
            BenchmarkResult(
                name="load_bridge",
                severity=BenchmarkSeverity.ERROR,
                message=f"Failed to load TransformerBridge: {str(e)}",
                passed=False,
            )
        )
        if verbose:
            print(format_results(results))
        return results

    # Load reference models for different comparison purposes:
    # 1. HuggingFace: For comparing unprocessed Bridge implementation
    # 2. HookedTransformer: For comparing processed Bridge compatibility mode
    hf_model: Optional[torch.nn.Module] = None
    ht_model: Optional[HookedTransformer] = None

    # Load HuggingFace model for raw forward pass comparison
    if use_hf_reference:
        if verbose:
            print("Loading HuggingFace reference model...")
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[arg-type]
            hf_model.to(device)  # type: ignore[arg-type]
            hf_model.eval()
            if verbose:
                print("✓ HuggingFace model loaded (for raw forward pass comparison)\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load HuggingFace model: {str(e)}\n")

    # Load HookedTransformer for compatibility mode comparison
    if use_ht_reference:
        if verbose:
            print("Loading HookedTransformer reference model...")
        try:
            # Load with same processing as Bridge compatibility mode
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
                print("✓ HookedTransformer loaded (for compatibility mode comparison)\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load HookedTransformer: {str(e)}\n")

    # Check if we have at least one reference model
    if hf_model is None and ht_model is None:
        if verbose:
            print("⚠ No reference models available - running Bridge-only validation\n")

    # Run benchmarks
    if verbose:
        print("Running benchmarks...\n")

    # Forward pass benchmarks (compare unprocessed Bridge vs HF)
    if verbose:
        print("1. Forward Pass Benchmarks (unprocessed Bridge vs HuggingFace)")
    results.append(benchmark_forward_pass(bridge_unprocessed, test_text, reference_model=hf_model))

    # Compatibility mode benchmarks (compare processed Bridge vs processed HT)
    if verbose:
        print("2. Compatibility Mode Benchmarks (processed Bridge vs HookedTransformer)")
    if bridge_processed and ht_model:
        results.append(
            benchmark_loss_equivalence(bridge_processed, test_text, reference_model=ht_model)
        )
        results.append(
            benchmark_logits_equivalence(bridge_processed, test_text, reference_model=ht_model)
        )
    elif bridge_processed:
        # No HT reference - just validate processed Bridge works
        results.append(
            benchmark_loss_equivalence(bridge_processed, test_text, reference_model=None)
        )
        results.append(
            benchmark_logits_equivalence(bridge_processed, test_text, reference_model=None)
        )
    else:
        # No processed bridge - skip compatibility tests
        if verbose:
            print("⚠ Compatibility mode disabled - skipping processed comparisons\n")

    # Hook benchmarks (use processed Bridge for compatibility with HT)
    if verbose:
        print("3. Hook Registration Benchmarks")
    test_bridge = bridge_processed if bridge_processed and ht_model else bridge
    results.append(benchmark_hook_registry(test_bridge, reference_model=ht_model))
    results.append(benchmark_hook_functionality(test_bridge, test_text, reference_model=ht_model))
    results.append(
        benchmark_critical_forward_hooks(test_bridge, test_text, reference_model=ht_model)
    )

    # Only run full forward hooks if HT reference is available (computationally expensive)
    if ht_model is not None and bridge_processed:
        results.append(
            benchmark_forward_hooks(bridge_processed, test_text, reference_model=ht_model)
        )

    # Gradient benchmarks (use processed Bridge for compatibility with HT)
    if verbose:
        print("4. Backward Gradient Benchmarks")
    results.append(benchmark_gradient_computation(test_bridge, test_text, reference_model=ht_model))
    results.append(
        benchmark_critical_backward_hooks(test_bridge, test_text, reference_model=ht_model)
    )

    # Only run full backward hooks if HT reference is available (computationally expensive)
    if ht_model is not None and bridge_processed:
        results.append(
            benchmark_backward_hooks(bridge_processed, test_text, reference_model=ht_model)
        )

    # Generation benchmarks (test both unprocessed and processed)
    if verbose:
        print("5. Generation Benchmarks")
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

    # Weight processing benchmarks (compare processed Bridge vs processed HT)
    if verbose:
        print("6. Weight Processing Benchmarks")
    if bridge_processed and ht_model:
        results.append(
            benchmark_weight_processing(bridge_processed, test_text, reference_model=ht_model)
        )
        results.append(
            benchmark_weight_sharing(bridge_processed, test_text, reference_model=ht_model)
        )
        results.append(benchmark_weight_modification(bridge_processed, test_text))
    elif bridge_processed:
        # No HT reference - just test processed bridge works
        results.append(
            benchmark_weight_processing(bridge_processed, test_text, reference_model=None)
        )
        results.append(benchmark_weight_sharing(bridge_processed, test_text, reference_model=None))
        results.append(benchmark_weight_modification(bridge_processed, test_text))

    # Activation cache benchmarks (compare processed Bridge vs processed HT)
    if verbose:
        print("7. Activation Cache Benchmarks")
    if bridge_processed and ht_model:
        results.append(
            benchmark_run_with_cache(bridge_processed, test_text, reference_model=ht_model)
        )
        results.append(
            benchmark_activation_cache(bridge_processed, test_text, reference_model=ht_model)
        )
    elif bridge_processed:
        # No HT reference - just test processed bridge works
        results.append(benchmark_run_with_cache(bridge_processed, test_text, reference_model=None))
        results.append(
            benchmark_activation_cache(bridge_processed, test_text, reference_model=None)
        )

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
