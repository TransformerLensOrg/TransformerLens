#!/usr/bin/env python3
"""Unified benchmark test suite using run_benchmark_suite().

This test suite replaces individual benchmark tests by running the comprehensive
benchmark suite and asserting all results pass. This eliminates duplication and
ensures consistency between manual benchmark runs and pytest.
"""

import pytest

from transformer_lens.benchmarks import run_benchmark_suite
from transformer_lens.benchmarks.utils import BenchmarkSeverity

# Model list - start with gpt2, can be expanded with more models
BENCHMARK_MODELS = ["gpt2"]


class TestBenchmarkSuite:
    """Test suite that runs comprehensive benchmarks for each model."""

    @pytest.mark.parametrize("model_name", BENCHMARK_MODELS)
    def test_benchmark_suite(self, model_name):
        """Run full benchmark suite and assert all tests pass.

        This test runs the complete benchmark suite which includes:
        - Phase 1: HuggingFace + Bridge (unprocessed) comparison
        - Phase 2: Bridge + HookedTransformer (unprocessed) comparison
        - Phase 3: Bridge + HookedTransformer (processed) compatibility

        The benchmark suite covers ~28 tests including:
        - Component-level equivalence
        - Forward/backward pass equivalence
        - Hook functionality and registration
        - Weight processing verification
        - Generation capabilities
        - Activation caching
        """
        results = run_benchmark_suite(
            model_name=model_name,
            device="cpu",
            use_hf_reference=True,
            use_ht_reference=True,
            enable_compatibility_mode=True,
            verbose=False,  # Keep test output clean
            track_memory=False,
        )

        # Filter out skipped tests
        non_skipped_results = [r for r in results if r.severity != BenchmarkSeverity.SKIPPED]

        # Collect failures
        failures = [r for r in non_skipped_results if not r.passed]

        # Build detailed failure message
        if failures:
            failure_details = []
            for r in failures:
                details_str = ""
                if r.details:
                    # Format details nicely
                    details_items = [f"{k}={v}" for k, v in r.details.items()]
                    details_str = f" ({', '.join(details_items[:3])})"

                failure_details.append(f"  - {r.name}: {r.message}{details_str}")

            failure_msg = "\n".join(failure_details)

            # Show summary stats
            total = len(non_skipped_results)
            passed = len([r for r in non_skipped_results if r.passed])
            failed = len(failures)

            pytest.fail(
                f"Benchmark suite failed for {model_name}:\n"
                f"Results: {passed}/{total} passed, {failed}/{total} failed\n\n"
                f"Failures:\n{failure_msg}"
            )

        # If we get here, all tests passed
        total = len(non_skipped_results)
        assert total > 0, "No benchmarks were run"
