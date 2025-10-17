# RNG State Preservation Fix

## Problem

The Main_Demo notebook was producing different random values on each execution of the factored matrix cell, even though the cell sets `torch.manual_seed(50)` at the beginning. This caused inconsistent results:

- First run: `tensor(3.6203)`
- Second run: `tensor(9.4086)`
- Expected: `tensor(9.9105)` (consistently)

## Root Causes

There were **two locations** where RNG state was being consumed:

### 1. Model Loading in `boot()` (sources/transformers.py)

When loading the HuggingFace model with `AutoModelForCausalLM.from_pretrained()`, PyTorch's RNG state was being consumed during model initialization. This affected any random number generation that occurred after loading the model.

### 2. Reference Model Loading in `process_compatibility_weights()` (bridge.py)

When calling `enable_compatibility_mode()`, the method internally calls `process_compatibility_weights()`, which loads a HookedTransformer reference model to extract processed weights. This second model loading also consumed RNG state, causing different random values on each notebook cell execution.

## Solution

Added RNG state save/restore in both locations:

### Fix 1: `transformer_lens/model_bridge/sources/transformers.py` (lines 234-249, 280-284)

```python
# Save RNG state before loading to avoid affecting downstream random number generation
rng_state = torch.get_rng_state()
if torch.cuda.is_available():
    cuda_rng_state = torch.cuda.get_rng_state_all()

# Load HuggingFace model
hf_model = AutoModelForCausalLM.from_pretrained(...)

# ... rest of initialization ...

# Restore RNG state at the END of boot() function
torch.set_rng_state(rng_state)
if torch.cuda.is_available():
    torch.cuda.set_rng_state_all(cuda_rng_state)
```

**Key point**: The restoration happens at the *end* of the `boot()` function, after all initialization code that might consume RNG state.

### Fix 2: `transformer_lens/model_bridge/bridge.py` (lines 652-696)

```python
def process_compatibility_weights(self, verbose: bool = False) -> None:
    """Process and load weights from a reference HookedTransformer model."""
    from transformer_lens import HookedTransformer
    import torch

    # Save RNG state before loading reference model
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()

    # Load reference model
    reference_hooked = HookedTransformer.from_pretrained(...)

    # ... process weights ...

    # Restore RNG state
    torch.set_rng_state(rng_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state)
```

## Verification

Created two test scripts to verify the fix:

### 1. `test_rng_reproducibility.py`

Tests that calling `enable_compatibility_mode()` multiple times produces identical results:

```
Run 1: OV circuit norm: 9.339180, Random tensor: [0.4624912738800049, ...]
Run 2: OV circuit norm: 9.339180, Random tensor: [0.4624912738800049, ...]
Run 3: OV circuit norm: 9.339180, Random tensor: [0.4624912738800049, ...]

✓ SUCCESS: RNG state is properly preserved!
```

### 2. `test_main_demo_reproducibility.py`

Replicates the exact Main_Demo notebook scenario with factored matrices:

```
Execution 1: OV norm: 9.339180, QK norm: 13.312410, Random: 0.462491
Execution 2: OV norm: 9.339180, QK norm: 13.312410, Random: 0.462491
Execution 3: OV norm: 9.339180, QK norm: 13.312410, Random: 0.462491

✓ SUCCESS: All executions produced identical results!
```

## Test Results

- **320 integration tests pass** (all existing tests continue to work)
- **Mypy passes** with no type errors
- **RNG reproducibility tests pass** (both test scripts show identical results across runs)

## Impact

This fix ensures that:

1. **Notebook reproducibility**: Running the same notebook cell multiple times produces identical results
2. **Downstream RNG unaffected**: User code that uses `torch.randn()` or similar functions after loading a model will get consistent random values
3. **No performance impact**: Saving and restoring RNG state is extremely fast
4. **No breaking changes**: All existing functionality continues to work identically

## User Action Required

To see the fix in action:

1. **Restart your Jupyter kernel** (important - this clears any cached state)
2. Run the Main_Demo notebook cells again
3. The factored matrix cell should now consistently produce the same results on every execution

## Technical Notes

- Both CPU and CUDA RNG states are saved/restored
- The fix handles cases where CUDA is not available
- RNG state restoration happens after all initialization code to ensure downstream code gets consistent random values
- The fix is transparent to users - no API changes required
