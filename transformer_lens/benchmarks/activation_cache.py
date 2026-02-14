"""Activation cache benchmarks for TransformerBridge."""

from typing import Optional

import torch

from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.model_bridge import TransformerBridge


def benchmark_run_with_cache(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
) -> BenchmarkResult:
    """Benchmark run_with_cache functionality.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model

    Returns:
        BenchmarkResult with cache functionality details
    """
    try:
        output, cache = bridge.run_with_cache(test_text)

        # Verify output and cache
        if not isinstance(output, torch.Tensor):
            return BenchmarkResult(
                name="run_with_cache",
                severity=BenchmarkSeverity.DANGER,
                message="Output is not a tensor",
                passed=False,
            )

        if not isinstance(cache, ActivationCache):
            return BenchmarkResult(
                name="run_with_cache",
                severity=BenchmarkSeverity.DANGER,
                message="Cache is not an ActivationCache object",
                passed=False,
            )

        if len(cache) == 0:
            return BenchmarkResult(
                name="run_with_cache",
                severity=BenchmarkSeverity.DANGER,
                message="Cache is empty",
                passed=False,
            )

        # Verify cache contains expected keys
        cache_keys = list(cache.keys())
        expected_patterns = ["embed", "ln_final", "unembed"]

        missing_patterns = []
        for pattern in expected_patterns:
            if not any(pattern in key for key in cache_keys):
                missing_patterns.append(pattern)

        if missing_patterns:
            return BenchmarkResult(
                name="run_with_cache",
                severity=BenchmarkSeverity.WARNING,
                message=f"Cache missing expected patterns: {missing_patterns}",
                details={"missing": missing_patterns, "cache_keys_count": len(cache_keys)},
            )

        # Verify cached tensors are actually tensors
        non_tensor_keys = []
        for key, value in cache.items():
            if not isinstance(value, torch.Tensor):
                non_tensor_keys.append(key)

        if non_tensor_keys:
            return BenchmarkResult(
                name="run_with_cache",
                severity=BenchmarkSeverity.WARNING,
                message=f"Cache contains {len(non_tensor_keys)} non-tensor values",
                details={"non_tensor_keys": non_tensor_keys[:5]},
            )

        if reference_model is not None:
            # Compare cache size with reference
            reference_output, reference_cache = reference_model.run_with_cache(test_text)

            cache_diff = abs(len(cache) - len(reference_cache))
            if cache_diff > 0:
                return BenchmarkResult(
                    name="run_with_cache",
                    severity=BenchmarkSeverity.WARNING,
                    message=f"Cache sizes differ: Bridge={len(cache)}, Ref={len(reference_cache)}",
                    details={"bridge_size": len(cache), "ref_size": len(reference_cache)},
                )

        return BenchmarkResult(
            name="run_with_cache",
            severity=BenchmarkSeverity.INFO,
            message=f"run_with_cache successful with {len(cache)} cached activations",
            details={"cache_size": len(cache)},
        )

    except Exception as e:
        return BenchmarkResult(
            name="run_with_cache",
            severity=BenchmarkSeverity.ERROR,
            message=f"run_with_cache failed: {str(e)}",
            passed=False,
        )


def benchmark_activation_cache(
    bridge: TransformerBridge,
    test_text: str,
    reference_model: Optional[HookedTransformer] = None,
    tolerance: float = 1e-3,
) -> BenchmarkResult:
    """Benchmark activation cache values against reference model.

    Args:
        bridge: TransformerBridge model to test
        test_text: Input text for testing
        reference_model: Optional HookedTransformer reference model
        tolerance: Tolerance for activation comparison

    Returns:
        BenchmarkResult with cache value comparison details
    """
    try:
        bridge_output, bridge_cache = bridge.run_with_cache(test_text)

        if reference_model is None:
            # No reference - just verify cache structure
            return BenchmarkResult(
                name="activation_cache",
                severity=BenchmarkSeverity.INFO,
                message=f"Activation cache created with {len(bridge_cache)} entries",
                details={"cache_size": len(bridge_cache)},
            )

        reference_output, reference_cache = reference_model.run_with_cache(test_text)

        # Find common keys
        bridge_keys = set(bridge_cache.keys())
        reference_keys = set(reference_cache.keys())
        common_keys = bridge_keys & reference_keys

        if len(common_keys) == 0:
            return BenchmarkResult(
                name="activation_cache",
                severity=BenchmarkSeverity.DANGER,
                message="No common keys between Bridge and Reference caches",
                details={
                    "bridge_keys": len(bridge_keys),
                    "reference_keys": len(reference_keys),
                },
                passed=False,
            )

        # Compare activations for common keys
        mismatches = []
        for key in sorted(common_keys):
            bridge_tensor = bridge_cache[key]
            reference_tensor = reference_cache[key]

            # Check shapes
            if bridge_tensor.shape != reference_tensor.shape:
                mismatches.append(
                    f"{key}: Shape mismatch - Bridge{bridge_tensor.shape} vs Ref{reference_tensor.shape}"
                )
                continue

            # Check values
            if not torch.allclose(bridge_tensor, reference_tensor, atol=tolerance, rtol=0):
                max_diff = torch.max(torch.abs(bridge_tensor - reference_tensor)).item()
                mean_diff = torch.mean(torch.abs(bridge_tensor - reference_tensor)).item()
                mismatches.append(
                    f"{key}: Value mismatch - max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
                )

        if mismatches:
            return BenchmarkResult(
                name="activation_cache",
                severity=BenchmarkSeverity.WARNING,
                message=f"Found {len(mismatches)}/{len(common_keys)} cached activations with differences",
                details={
                    "total_keys": len(common_keys),
                    "mismatches": len(mismatches),
                    "sample_mismatches": mismatches[:5],
                },
            )

        return BenchmarkResult(
            name="activation_cache",
            severity=BenchmarkSeverity.INFO,
            message=f"All {len(common_keys)} cached activations match within tolerance",
            details={"cache_size": len(common_keys), "tolerance": tolerance},
        )

    except Exception as e:
        return BenchmarkResult(
            name="activation_cache",
            severity=BenchmarkSeverity.ERROR,
            message=f"Activation cache check failed: {str(e)}",
            passed=False,
        )
