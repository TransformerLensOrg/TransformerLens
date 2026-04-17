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
