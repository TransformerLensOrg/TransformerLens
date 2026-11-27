# TransformerBridge Benchmarks

This directory contains a comprehensive benchmark suite for testing TransformerBridge compatibility with HuggingFace models.

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
    use_hf_reference=True,      # Compare against the raw HuggingFace model
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
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM

# Load models
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
bridge.enable_compatibility_mode()
hf = AutoModelForCausalLM.from_pretrained("gpt2")  # raw HF reference

# Run individual benchmarks
test_text = "The quick brown fox"

result1 = benchmark_forward_pass(bridge, test_text, reference_model=hf)
print(result1)  # 🟢 [PASS] forward_pass: ...

result2 = benchmark_hook_functionality(bridge, test_text)  # reference-free ablation check
print(result2)  # 🟢 [PASS] hook_functionality: ...

result3 = benchmark_generation(bridge, test_text, max_new_tokens=10)
print(result3)  # 🟢 [PASS] generation: ...
```

## Using Benchmarks in Tests

The benchmarks are designed to be used in pytest test suites. Here's how to integrate them:

```python
import pytest
import torch
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.benchmarks import (
    benchmark_loss_equivalence,
    benchmark_logits_equivalence,
    benchmark_hook_functionality,
)

TEST_TEXT = "Natural language processing"


class TestTransformerBridgeCompatibility:
    @pytest.fixture
    def models(self):
        """Capture raw-weight references (== HF, per Phase 1), then enable compat mode."""
        bridge = TransformerBridge.boot_transformers("gpt2")
        with torch.no_grad():
            raw_logits = bridge(TEST_TEXT, return_type="logits")
            raw_loss = bridge(TEST_TEXT, labels=bridge.to_tokens(TEST_TEXT), return_type="loss").item()
        bridge.enable_compatibility_mode()
        return {"bridge": bridge, "raw_loss": raw_loss, "raw_logits": raw_logits}

    def test_loss_equivalence(self, models):
        """Weight processing must preserve the loss."""
        result = benchmark_loss_equivalence(
            models["bridge"],
            TEST_TEXT,
            reference_loss=models["raw_loss"],
            atol=1e-3
        )
        assert result.passed, result.message

    def test_logits_equivalence(self, models):
        """Weight processing must preserve the logits within tolerance."""
        result = benchmark_logits_equivalence(
            models["bridge"],
            TEST_TEXT,
            reference_logits=models["raw_logits"],
            atol=3e-2,
            rtol=3e-2
        )
        assert result.passed, result.message

    def test_hooks(self, models):
        """Reference-free ablation check of the hook system."""
        result = benchmark_hook_functionality(
            models["bridge"],
            TEST_TEXT,
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
- `benchmark_weight_modification()` - Weight modification propagation (the only one re-exported from `transformer_lens.benchmarks`)
- `benchmark_layer_norm_folding()`, `benchmark_value_bias_folding()` - Verify folding
- `benchmark_attention_output_centering()`, `benchmark_mlp_output_centering()`, `benchmark_unembed_centering()` - Verify centering

### `activation_cache.py`

Activation caching benchmarks:
- `benchmark_run_with_cache()` - Test cache functionality
- `benchmark_activation_cache()` - Compare cached activations

### `adapter_coverage.py`

Adapter coverage checks. Logit parity cannot tell whether an adapter maps and exposes the
whole architecture, because a bridge that delegates its forward to HuggingFace reproduces
the logits whatever its hooks describe. These checks are report-only: they are not part of
`run_benchmark_suite()` and do not affect a model's registry status.

A finding is one of two kinds. A **misleading** adapter (wrong hooks or logits) fails; an
**incomplete** one passes, with what it leaves out listed as gaps and counted in a coverage
figure.

- `benchmark_residual_floor()` - Every block's change to the residual stream is accounted for by hooked outputs (blocking)
- `benchmark_hook_consistency()` - Bridge logits match HuggingFace, the residual stream adds up, the final residual gives the logits, and hook edits match edits on the HF module (blocking)
- `benchmark_layer_type_coverage()` - The same holds on layer kinds the config allows but the checkpoint lacks (blocking)
- `benchmark_scaling_factor_coverage()` - The same holds when a multiplier the checkpoint leaves at 1.0 takes another value (blocking)
- `benchmark_structural_coverage()` - Which HF parameters are owned by the bridge component their module kind calls for (`MODULE_KIND_OWNERS`); the rest are gaps
- `benchmark_expected_hooks()` - Which hooks the block classes promise are absent or silent; those are gaps
- `benchmark_reference_comparison()` - Which of a reference adapter's hook and weight names the adapter lacks; those are gaps

`adapter_coverage_runner.py` runs them on one adapter file in a fresh subprocess; see
[Checking an Adapter's Coverage](#checking-an-adapters-coverage).

### `main_benchmark.py`

Main benchmark suite with tiered comparison logic:
- `run_benchmark_suite()` - Run complete benchmark suite

## Comparison Strategy

The benchmarks use a tiered approach for comparison:

1. **First Priority**: Compare TransformerBridge → HuggingFace model (raw)
   - Direct comparison with original HF implementation
   - Ensures bridge maintains model fidelity

2. **Second Priority**: HF-anchored equivalence in both weight modes
   - Phases 2 and 3 compare unprocessed and compatibility-mode outputs against the saved Phase 1 HF reference
   - Ensures weight processing preserves model behavior

3. **Third Priority**: TransformerBridge-only validation
   - Reference-free structural self-checks for hooks, cache, and gradients
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

# Disable the HF reference comparison (structural-only; can only mint PROVISIONAL)
python -m transformer_lens.benchmarks.main_benchmark \
    --model gpt2 \
    --no-hf-reference \
    --quiet  # Suppress verbose output
```

### Checking an Adapter's Coverage

```bash
# The adapter already registered for a checkpoint
python -m transformer_lens.benchmarks.adapter_coverage_runner --checkpoint gpt2

# An adapter file that is not registered, used for this run only
python -m transformer_lens.benchmarks.adapter_coverage_runner \
    --adapter path/to/my_adapter.py \
    --checkpoint org/tiny-random-FooForCausalLM \
    --output coverage.json

# The same, compared with a reference adapter for the architecture
python -m transformer_lens.benchmarks.adapter_coverage_runner \
    --adapter path/to/my_adapter.py \
    --reference-adapter transformer_lens.model_bridge.supported_architectures.gpt2 \
    --checkpoint gpt2
```

The exit code is 0 when there is no blocking finding and 1 otherwise. The JSON report has:

| Field | Meaning |
| --- | --- |
| `passed` | No blocking finding |
| `blocking_findings` | One entry per finding: `check`, `part`, `message` |
| `n_blocking_parts`, `n_blocking_failed` | How many blocking parts ran and how many failed, counting variant runs |
| `coverage`, `coverage_basis` | A figure from 0 to 1 and what it is measured against: `reference` (share of a reference adapter's hook and weight names the adapter has) or `ownership` (share of parameter-bearing module path patterns owned by the component for their kind) |
| `gaps` | One entry per gap: `kind` (`unmapped`, `generic`, `absent_hook`, `silent_hook`, `missing_from_reference`), `where`, `declared`, `reason` |
| `checks` | One entry per check: `name`, `status` (`pass`, `gaps`, `fail` or `skip`), `message`, `details` |

A check that does not apply is a `skip`; one that could not run is a blocking finding,
because it certifies nothing. `--hf-class` is read from the checkpoint's config when omitted.

From Python, `run_adapter_coverage()` is the same run as a function, and
`run_coverage_checks(bridge)` runs the checks on a bridge you have already built, with no
checkpoint name, adapter file or subprocess:

```python
from transformer_lens.benchmarks.adapter_coverage_runner import run_coverage_checks

report = run_coverage_checks(bridge)
print(report["passed"], report["coverage"], report["gaps"])
```

The layer-type and scaling checks rebuild the architecture from an edited config. Above
`--max-variant-params` (20M by default) they first rebuild it small, from the checkpoint's
config with the size fields reduced and random weights.

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

- **Performance**: Full hook comparison tests are computationally expensive.

## Contributing

When adding new test patterns:

1. Create the benchmark function in the appropriate module
2. Add it to `__init__.py` exports
3. Update `main_benchmark.py` to include it in the suite
4. Update this README with usage examples
5. Update existing tests to use the new benchmark function

## See Also

- [TransformerBridge overview (AGENTS.md §2)](../../AGENTS.md#2-two-systems-live-in-this-repo)
- [Test Suite](../../tests/)
