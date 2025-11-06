"""Main benchmark runner for TransformerBridge.

This module provides the main benchmark suite that compares TransformerBridge
against reference implementations in a tiered approach:
1. First Priority: Compare TB → HuggingFace model (raw)
2. Second Priority: If HT version exists, compare TB → HT
3. Third Priority: If model unavailable in HT, run TB-only validation
"""

from typing import List, Optional, Union

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

    # Load TransformerBridge
    if verbose:
        print("Loading TransformerBridge...")
    try:
        bridge = TransformerBridge.boot_transformers(model_name, device=device)  # type: ignore[attr-defined]
        if enable_compatibility_mode:
            bridge.enable_compatibility_mode(disable_warnings=True)
        if verbose:
            print("✓ TransformerBridge loaded\n")
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

    # Determine reference model to use (tiered approach)
    hf_model: Optional[torch.nn.Module] = None
    ht_model: Optional[HookedTransformer] = None
    reference_model: Optional[Union[HookedTransformer, torch.nn.Module]] = None

    # Priority 1: Try to load HuggingFace model
    if use_hf_reference:
        if verbose:
            print("Loading HuggingFace reference model...")
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[arg-type]
            hf_model.to(device)  # type: ignore[arg-type]
            hf_model.eval()
            reference_model = hf_model
            # If using HF as reference, disable processing in bridge to match raw weights
            if enable_compatibility_mode and getattr(bridge, "_weights_processed", False):
                if verbose:
                    print("⚠ Disabling weight processing to match raw HuggingFace model\n")
                # Reload bridge without processing for fair comparison
                bridge = TransformerBridge.boot_transformers(model_name, device=device)  # type: ignore[attr-defined]
                bridge.enable_compatibility_mode(disable_warnings=True, no_processing=True)
            if verbose:
                print("✓ HuggingFace model loaded as primary reference\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load HuggingFace model: {str(e)}\n")

    # Priority 2: Try to load HookedTransformer model (if HF not available or requested)
    if use_ht_reference and (hf_model is None or not use_hf_reference):
        if verbose:
            print("Loading HookedTransformer reference model...")
        try:
            # Try with processing first (compatibility mode)
            ht_model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,
            )
            reference_model = ht_model
            if verbose:
                print("✓ HookedTransformer loaded as reference\n")
        except Exception as e:
            if verbose:
                print(f"✗ Could not load HookedTransformer: {str(e)}\n")

    # Priority 3: No reference model - run TB-only validation
    if reference_model is None:
        if verbose:
            print("⚠ No reference model available - running TB-only validation\n")

    # Run benchmarks
    if verbose:
        print("Running benchmarks...\n")

    # Forward pass benchmarks
    if verbose:
        print("1. Forward Pass Benchmarks")
    results.append(
        benchmark_forward_pass(
            bridge, test_text, reference_model=ht_model if ht_model else hf_model
        )
    )
    results.append(benchmark_loss_equivalence(bridge, test_text, reference_model=ht_model))
    results.append(benchmark_logits_equivalence(bridge, test_text, reference_model=ht_model))

    # Hook benchmarks
    if verbose:
        print("2. Hook Registration Benchmarks")
    results.append(benchmark_hook_registry(bridge, reference_model=ht_model))
    results.append(benchmark_hook_functionality(bridge, test_text, reference_model=ht_model))
    results.append(benchmark_critical_forward_hooks(bridge, test_text, reference_model=ht_model))

    # Only run full forward hooks if HT reference is available (computationally expensive)
    if ht_model is not None:
        results.append(benchmark_forward_hooks(bridge, test_text, reference_model=ht_model))

    # Gradient benchmarks
    if verbose:
        print("3. Backward Gradient Benchmarks")
    results.append(benchmark_gradient_computation(bridge, test_text, reference_model=ht_model))
    results.append(benchmark_critical_backward_hooks(bridge, test_text, reference_model=ht_model))

    # Only run full backward hooks if HT reference is available (computationally expensive)
    if ht_model is not None:
        results.append(benchmark_backward_hooks(bridge, test_text, reference_model=ht_model))

    # Generation benchmarks
    if verbose:
        print("4. Generation Benchmarks")
    results.append(benchmark_generation(bridge, test_text, max_new_tokens=10))
    results.append(benchmark_generation_with_kv_cache(bridge, test_text, max_new_tokens=10))
    results.append(
        benchmark_multiple_generation_calls(
            bridge,
            test_prompts=[
                "The quick brown fox",
                "Hello world",
                "Machine learning is",
            ],
            max_new_tokens=5,
        )
    )

    # Weight processing benchmarks
    if verbose:
        print("5. Weight Processing Benchmarks")
    results.append(benchmark_weight_processing(bridge, test_text, reference_model=ht_model))
    results.append(benchmark_weight_sharing(bridge, test_text, reference_model=ht_model))
    results.append(benchmark_weight_modification(bridge, test_text))

    # Activation cache benchmarks
    if verbose:
        print("6. Activation Cache Benchmarks")
    results.append(benchmark_run_with_cache(bridge, test_text, reference_model=ht_model))
    results.append(benchmark_activation_cache(bridge, test_text, reference_model=ht_model))

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
