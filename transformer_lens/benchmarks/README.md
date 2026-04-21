# TransformerBridge Benchmarks

This directory contains a comprehensive benchmark suite for testing TransformerBridge compatibility with HuggingFace models and HookedTransformer.

## Overview

The benchmark system provides reusable functions for testing various aspects of TransformerBridge:

- **Forward Pass**: Compare model outputs, logits, and loss values
- **Hook System**: Test hook registration and behavior (forward and backward)
- **Gradients**: Verify backward pass gradient computation
- **Generation**: Test text generation and KV cache functionality
- **Weight Processing**: Verify weight transformations (folding, centering)
- **Activation Cache**: Test `run_with_cache` functionality

## Quick Start

### Running the Full Benchmark Suite

```python
from transformer_lens.benchmarks import run_benchmark_suite

# Run complete benchmark suite
results = run_benchmark_suite(
    model_name="gpt2",
    device="cpu",
    use_hf_reference=True,      # Compare against HuggingFace model
    use_ht_reference=True,      # Compare against HookedTransformer
    enable_compatibility_mode=True,
    verbose=True
)

# Check results
passed = sum(1 for r in results if r.passed)
print(f"Passed: {passed}/{len(results)} tests")
```

### Using Individual Benchmark Functions

```python
from transformer_lens.benchmarks import (
    benchmark_forward_pass,
    benchmark_hook_functionality,
    benchmark_generation,
)
from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

# Load models
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
bridge.enable_compatibility_mode()

ht = HookedTransformer.from_pretrained("gpt2")

# Run individual benchmarks
test_text = "The quick brown fox"

result1 = benchmark_forward_pass(bridge, test_text, reference_model=ht)
print(result1)  # 🟢 [PASS] forward_pass: ...

result2 = benchmark_hook_functionality(bridge, test_text, reference_model=ht)
print(result2)  # 🟢 [PASS] hook_functionality: ...

result3 = benchmark_generation(bridge, test_text, max_new_tokens=10)
print(result3)  # 🟢 [PASS] generation: ...
```

## Using Benchmarks in Tests

The benchmarks are designed to be used in pytest test suites. Here's how to integrate them:

```python
import pytest
from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.benchmarks import (
    benchmark_loss_equivalence,
    benchmark_logits_equivalence,
    benchmark_hook_functionality,
)


class TestTransformerBridgeCompatibility:
    @pytest.fixture
    def models(self):
        """Create models for testing."""
        ht = HookedTransformer.from_pretrained("gpt2")
        bridge = TransformerBridge.boot_transformers("gpt2")
        bridge.enable_compatibility_mode()
        return {"ht": ht, "bridge": bridge}

    def test_loss_equivalence(self, models):
        """Test loss computation matches."""
        test_text = "Natural language processing"
        result = benchmark_loss_equivalence(
            models["bridge"],
            test_text,
            reference_model=models["ht"],
            atol=1e-3
        )
        assert result.passed, result.message

    def test_logits_equivalence(self, models):
        """Test logits match within tolerance."""
        test_text = "Natural language processing"
        result = benchmark_logits_equivalence(
            models["bridge"],
            test_text,
            reference_model=models["ht"],
            atol=3e-2,
            rtol=3e-2
        )
        assert result.passed, result.message

    def test_hooks(self, models):
        """Test hook functionality."""
        test_text = "Natural language processing"
        result = benchmark_hook_functionality(
            models["bridge"],
            test_text,
            reference_model=models["ht"],
            atol=2e-3
        )
        assert result.passed, result.message
```

## Benchmark Modules

### `forward_pass.py`

Forward pass comparison benchmarks:
- `benchmark_forward_pass()` - Compare model outputs
- `benchmark_loss_equivalence()` - Compare loss values
- `benchmark_logits_equivalence()` - Compare logits outputs

### `hook_registration.py`

Hook system benchmarks:
- `benchmark_hook_registry()` - Check hook registry completeness
- `benchmark_forward_hooks()` - Compare all forward hook activations
- `benchmark_critical_forward_hooks()` - Compare key forward hooks
- `benchmark_hook_functionality()` - Test ablation hook effects

### `backward_gradients.py`

Gradient computation benchmarks:
- `benchmark_backward_hooks()` - Compare all backward hook gradients
- `benchmark_critical_backward_hooks()` - Compare key backward hooks
- `benchmark_gradient_computation()` - Basic gradient computation test

### `generation.py`

Text generation benchmarks:
- `benchmark_generation()` - Basic generation test
- `benchmark_generation_with_kv_cache()` - Generation with KV cache
- `benchmark_multiple_generation_calls()` - Multiple generation robustness

### `weight_processing.py`

Weight processing benchmarks:
- `benchmark_weight_processing()` - Verify folding and centering
- `benchmark_weight_sharing()` - Test weight modification effects
- `benchmark_weight_modification()` - Weight modification propagation

### `activation_cache.py`

Activation caching benchmarks:
- `benchmark_run_with_cache()` - Test cache functionality
- `benchmark_activation_cache()` - Compare cached activations

### `main_benchmark.py`

Main benchmark suite with tiered comparison logic:
- `run_benchmark_suite()` - Run complete benchmark suite

## Comparison Strategy

The benchmarks use a tiered approach for comparison:

1. **First Priority**: Compare TransformerBridge → HuggingFace model (raw)
   - Direct comparison with original HF implementation
   - Ensures bridge maintains model fidelity

2. **Second Priority**: Compare TransformerBridge → HookedTransformer
   - If HT version exists, compare processed outputs
   - Ensures compatibility with TransformerLens ecosystem

3. **Third Priority**: TransformerBridge-only validation
   - If model unavailable in HT, validate bridge independently
   - Ensures basic functionality and structural correctness

## Benchmark Results

Results are returned as `BenchmarkResult` objects with severity levels:

- **🟢 INFO**: Perfect match or expected minor differences
- **🟡 WARNING**: Acceptable differences but noteworthy
- **🔴 DANGER**: Significant mismatches or failures
- **❌ ERROR**: Test failed to run

Each result includes:
- `name`: Test name
- `severity`: Severity level
- `message`: Human-readable description
- `details`: Additional diagnostic information
- `passed`: Boolean pass/fail status

## Command Line Usage

Run benchmarks from the command line:

```bash
# Basic usage
python -m transformer_lens.benchmarks.main_benchmark --model gpt2

# With options
python -m transformer_lens.benchmarks.main_benchmark \
    --model gpt2 \
    --device cuda \
    --no-compat  # Disable compatibility mode

# Disable reference comparisons
python -m transformer_lens.benchmarks.main_benchmark \
    --model gpt2 \
    --no-hf-reference \
    --no-ht-reference \
    --quiet  # Suppress verbose output
```

## Example Output

```
================================================================================
Running TransformerBridge Benchmark Suite
Model: gpt2
Device: cpu
================================================================================

Loading TransformerBridge...
✓ TransformerBridge loaded

Loading HuggingFace reference model...
✓ HuggingFace model loaded as primary reference

Running benchmarks...

1. Forward Pass Benchmarks
2. Hook Registration Benchmarks
3. Backward Gradient Benchmarks
4. Generation Benchmarks
5. Weight Processing Benchmarks
6. Activation Cache Benchmarks

================================================================================
BENCHMARK RESULTS
================================================================================

Total: 16 tests
Passed: 15 (93.8%)
Failed: 1 (6.2%)

🟢 INFO: 15
🟡 WARNING: 0
🔴 DANGER: 1
❌ ERROR: 0

--------------------------------------------------------------------------------
🟢 [PASS] forward_pass: Tensors match within tolerance
🟢 [PASS] loss_equivalence: Scalars match: 5.607012 ≈ 5.607012
🟢 [PASS] hook_registry: All 301 hooks match
🔴 [FAIL] backward_hooks: Found 5 significant mismatches
  total_hooks: 289
  mismatches: 5
  sample_mismatches: ['blocks.0.hook_resid_pre', ...]
...
================================================================================
```

## Notes

- **Tolerances**: Different operations have different numerical precision requirements:
  - Forward pass: `atol=1e-3, rtol=3e-2` (relaxed due to accumulated differences)
  - Backward hooks: `atol=0.2, rtol=3e-4` (relaxed due to gradient magnitude variations)
  - Loss/scalar comparisons: `atol=1e-5`

- **Known Differences**: Some architectural differences are expected and filtered:
  - Hook shape differences (e.g., `hook_z` concatenation)
  - LayerNorm bridging numerical differences
  - Attention pattern computation differences

- **Performance**: Full hook comparison tests are computationally expensive and only run when a HookedTransformer reference is available.

## Contributing

When adding new test patterns:

1. Create the benchmark function in the appropriate module
2. Add it to `__init__.py` exports
3. Update `main_benchmark.py` to include it in the suite
4. Update this README with usage examples
5. Update existing tests to use the new benchmark function

## See Also

- [TransformerBridge Documentation](../model_bridge/README.md)
- [HookedTransformer API](../HookedTransformer.py)
- [Test Suite](../../tests/)
