# Special Cases

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

## Mixture of Experts error rates
Due to the Top-K gating performed in the hidden layer of Mixture of Experts models, small errors can be amplified 
greatly in cases where a different expert is selected, which leads to a higher than normal variance in the error rate
of the final logits. In testing done on Mixtral running in half precision, the standard deviation of the absolute error 
rate of the logits compared to those from the default model was found to be around 2e-3.

There are two main ways to mitigate this:

1. **Skip weight preprocessing.** On the bridge, simply load with `TransformerBridge.boot_transformers(...)` and do not call `enable_compatibility_mode()` - the bridge preserves raw HF weights by default, so no additional flag is needed. On the legacy `HookedTransformer` path, use `HookedTransformer.from_pretrained_no_processing` instead of `HookedTransformer.from_pretrained`.
2. **Increase the precision of the data type used in the model.**

## Qwen3.5 text-only models

Qwen3.5 has no `HookedTransformer.from_pretrained` legacy path. Load it through `TransformerBridge`. Dense text-only checkpoints can be loaded with:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("Qwen/Qwen3.5-0.8B")
```

Qwen3.5 uses a hybrid stack. Full-attention layers expose the usual hooks under
`blocks.N.attn.*`; linear-attention layers expose GatedDeltaNet hooks under
`blocks.N.linear_attn.*`, including `hook_q_pre_conv`, `hook_k_pre_conv`,
`hook_v_pre_conv`, `hook_beta`, `hook_log_decay`, `hook_recurrence_out`, and
`hook_out`. Full multimodal `Qwen3_5ForConditionalGeneration`, image/video
inputs, and Qwen3.5 MoE checkpoints are not supported by this adapter.

## Stateless reparametrization (`torch.func.functional_call`) is unsupported

`torch.func.functional_call` — and `torch.nn.utils.stateless` generally — fails to
restore parameters through a `TransformerBridge` tree. Every replaced component is
registered twice (inside the wrapped HuggingFace model and as a bridge submodule), and
torch's tied-weight handling swaps the single shared parameter slot once per alias: the
second swap records the override as the "original", so restoration silently installs the
override permanently. After such a call the affected weight is a plain tensor, not a
`Parameter`, and later code fails with errors like `"... weight must be a trainable
Parameter"`.

For temporary weight edits, use the supported primitive instead:

```python
from transformer_lens.utilities import temporarily_swap_parameter

weight = model.blocks[11].mlp.out.original_component.weight
with temporarily_swap_parameter(weight, updated_values):
    edited_logits = model(tokens)
# weight is restored here, even if the block raised
```

The swap is in-place, so the `Parameter` object, its `requires_grad` state, `.grad`
buffer, optimizer references, and all aliased registrations stay intact.
