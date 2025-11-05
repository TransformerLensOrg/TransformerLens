"""Benchmark utilities for TransformerBridge testing.

This module provides reusable benchmark functions for comparing TransformerBridge
with HuggingFace models and HookedTransformer implementations.
"""

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
from transformer_lens.benchmarks.main_benchmark import run_benchmark_suite
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.benchmarks.weight_processing import (
    benchmark_weight_modification,
    benchmark_weight_processing,
    benchmark_weight_sharing,
)

__all__ = [
    # Main benchmark runner
    "run_benchmark_suite",
    # Result types
    "BenchmarkResult",
    "BenchmarkSeverity",
    # Forward pass benchmarks
    "benchmark_forward_pass",
    "benchmark_logits_equivalence",
    "benchmark_loss_equivalence",
    # Hook benchmarks
    "benchmark_forward_hooks",
    "benchmark_critical_forward_hooks",
    "benchmark_hook_functionality",
    "benchmark_hook_registry",
    # Gradient benchmarks
    "benchmark_backward_hooks",
    "benchmark_critical_backward_hooks",
    "benchmark_gradient_computation",
    # Generation benchmarks
    "benchmark_generation",
    "benchmark_generation_with_kv_cache",
    "benchmark_multiple_generation_calls",
    # Weight processing benchmarks
    "benchmark_weight_processing",
    "benchmark_weight_sharing",
    "benchmark_weight_modification",
    # Activation cache benchmarks
    "benchmark_activation_cache",
    "benchmark_run_with_cache",
]
