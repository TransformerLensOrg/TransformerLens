---
title: Model Tables
---
# Model Tables

```{warning}
`HookedTransformer` was removed in TransformerLens 4.0. New code should use [`TransformerBridge`](migrating_to_v3.md), which reproduces HookedTransformer numerics via `enable_compatibility_mode()`. The HookedTransformer model table below is a frozen snapshot kept for users still on the 2.x / 3.x branches. See the [migration guide](migrating_to_v3.md) for conversion recipes.
```

TransformerLens documents two model tables:

- **HookedTransformer** (removed in 4.0) -- the original TransformerLens models; a frozen table kept for 2.x / 3.x users.
- **TransformerBridge Models** -- Automatic compatibility layer for thousands of HuggingFace models across supported architectures.

```{toctree}
hooked_transformer_model_properties
/generated/transformer_bridge_models
```
