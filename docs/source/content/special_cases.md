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

Qwen3.5 support is available only through `TransformerBridge`, not the legacy
`HookedTransformer.from_pretrained` path. Install a Transformers release that
includes `Qwen3_5ForCausalLM` before loading these models:

```bash
pip install "transformers>=5.2.0"
```

Dense text-only checkpoints can then be loaded with:

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
