---
title: Model Tables
---
# Model Tables

```{warning}
`HookedTransformer` is deprecated as of TransformerLens 3.0 and will be removed in the next major version. New code should use [`TransformerBridge`](migrating_to_v3.md) instead. Existing `HookedTransformer` code continues to work through the 3.x branch via a compatibility layer. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

TransformerLens 3.0 provides two model loading paths, each with its own set of supported models.

- **HookedTransformer** -- The original TransformerLens models with full hook-point access and mechanistic interpretability support.
- **TransformerBridge Models** -- Automatic compatibility layer for thousands of HuggingFace models across supported architectures.

```{toctree}
/generated/model_properties_table
/generated/transformer_bridge_models
```
