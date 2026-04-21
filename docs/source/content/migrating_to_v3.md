# Migrating to TransformerLens 3

TransformerLens 3 introduces **TransformerBridge**, a new way of loading and instrumenting models that replaces `HookedTransformer.from_pretrained` as the recommended path for new code. Existing `HookedTransformer` code continues to run through a compatibility layer, but adopting the bridge unlocks broader architecture support and puts you on the supported path going forward.

This page explains the differences and gives side-by-side migration recipes for the most common patterns.

## Why the change?

`HookedTransformer` was a single unified implementation that every supported architecture had to be mapped into. That was beautiful in theory — interpretability code written once worked everywhere — but in practice it meant that adding a new architecture required reimplementing its forward pass inside TransformerLens, and any divergence from the HuggingFace version was a latent source of bugs.

TransformerBridge flips the arrangement. Instead of reimplementing models, it keeps the native HuggingFace implementation and wraps it behind a consistent interface through an **architecture adapter**. The adapter knows how the HF module graph maps onto a small set of generalized components (embedding, attention, MLP, normalization, blocks) and registers uniform hook points over them. The result is the same familiar TransformerLens experience — hooks, caches, patching — but applied to the real HF model, and extended to 50+ architectures out of the box.

## Loading a model

The loading API changes shape, but the mental model is the same: give it a HuggingFace model id, get back an object with the TransformerLens surface.

```python
# Before — TransformerLens 2.x
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device="cpu")

# After — TransformerLens 3.x
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
```

### Parameters that carried over

`device`, `dtype`, and `tokenizer` all work the same way.

### Parameters that moved

Weight-processing flags (`fold_ln`, `center_writing_weights`, `center_unembed`, `fold_value_biases`, `refactor_factored_attn_matrices`) no longer live on the load call. They moved to `enable_compatibility_mode()` — see the next section.

### Parameters that were removed

`n_devices`, `move_to_device`, `first_n_layers`, and `n_ctx` are not part of `boot_transformers`. If you relied on any of these, file an issue describing your use case — the right pattern for multi-GPU loads under the bridge is still being worked out.

### Parameters that are new

- `load_weights: bool = True` — set to `False` to construct the bridge with just the config (useful for shape-checking without paying the weight-load cost).
- `trust_remote_code: bool = False` — pass through to HuggingFace for models that ship custom modeling code.
- `hf_config_overrides: dict | None = None` — override specific fields of the HF config before the model is constructed.
- `hf_model` / `model_class` — advanced: pass in a pre-loaded HF model or a specific model class.

## Weight processing is now opt-in

This is the biggest behavioral change. `HookedTransformer.from_pretrained` applied `fold_ln`, `center_writing_weights`, and `center_unembed` by default. The bridge does **not** apply any of these on load — the raw HF weights are preserved.

If your existing code depends on folded/centered weights (e.g. for direct logit attribution, or any analysis that reasons about activations in the post-processed coordinate system), call `enable_compatibility_mode` after booting:

```python
# Before
model = HookedTransformer.from_pretrained("gpt2")  # fold_ln=True, center_*=True by default

# After
bridge = TransformerBridge.boot_transformers("gpt2")
bridge.enable_compatibility_mode()  # applies fold_ln, center_writing_weights, center_unembed, fold_value_biases
```

`enable_compatibility_mode` defaults to the same processing HookedTransformer used to do. You can opt out of individual steps, or disable all processing with `no_processing=True`:

```python
bridge.enable_compatibility_mode(
    fold_ln=True,
    center_writing_weights=True,
    center_unembed=True,
    fold_value_biases=True,
    refactor_factored_attn_matrices=False,  # same default as before
)
```

If you want no processing at all — the bridge's native default — you can skip `enable_compatibility_mode` entirely, or call it with `no_processing=True` if you still want the hook/component compatibility layer without the weight transforms.

## Hook names

The canonical hook names on the bridge use a uniform `hook_in` / `hook_out` convention. The old TransformerLens names are preserved through an alias layer, so existing code keeps working without changes:

```python
# Both of these return the same tensor on a bridge
cache["blocks.0.hook_resid_pre"]  # legacy alias — still works
cache["blocks.0.hook_in"]          # canonical name — preferred for new code
```

For the full mapping of legacy → canonical names and the expected tensor shape at each hook point, see the [Model Structure](model_structure.md) page.

## APIs that are unchanged

These work identically on `TransformerBridge` and need no migration:

- `to_tokens`, `to_string`
- `generate`
- `run_with_hooks`
- `run_with_cache`
- `__call__` / `forward`
- `cfg.*` — the bridge exposes a `.cfg` with the same fields (`n_layers`, `n_heads`, `d_model`, `d_vocab`, `n_ctx`, ...)
- `W_Q`, `W_K`, `W_V`, `W_O`, `b_Q`, `b_K`, `b_V`, `b_O` — attention weights are exposed with the same `[n_heads, d_model, d_head]` shape conventions

If your code only touches these APIs, the migration is genuinely just the loading call and (optionally) `enable_compatibility_mode`.

## Model name aliases are deprecated

`HookedTransformer.from_pretrained` accepted a lot of short aliases (`"llama-7b-hf"`, `"gpt-neo-125M"`, etc.) that mapped to specific HuggingFace paths. The bridge accepts the official HuggingFace names directly, and emits a deprecation warning when you pass a legacy alias. The aliases will be removed in the next major version.

```python
# Legacy (deprecated, still works with a warning)
TransformerBridge.boot_transformers("gpt2")

# Preferred
TransformerBridge.boot_transformers("openai-community/gpt2")
```

Check the [TransformerBridge Models](../generated/transformer_bridge_models.md) page for the canonical model ids.

## Full before-and-after example

A typical HookedTransformer notebook setup:

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "gpt2",
    device="cuda",
    dtype=torch.float32,
)

logits, cache = model.run_with_cache("The quick brown fox")
resid_pre = cache["blocks.0.hook_resid_pre"]
pattern = cache["blocks.0.attn.hook_pattern"]
```

The bridge equivalent:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "openai-community/gpt2",
    device="cuda",
    dtype=torch.float32,
)
bridge.enable_compatibility_mode()  # match HookedTransformer's default weight processing

logits, cache = bridge.run_with_cache("The quick brown fox")
resid_pre = cache["blocks.0.hook_in"]           # or "blocks.0.hook_resid_pre" via alias
pattern = cache["blocks.0.attn.hook_pattern"]
```

The cache, hook, and config APIs are the same. The only lines that had to change are the import, the load call, and — if you want the old weight-processing behavior — one extra call to `enable_compatibility_mode`.

## When to stay on HookedTransformer

If your code runs unchanged on TransformerLens 3 via the compatibility layer and you don't need architectures beyond what `HookedTransformer` already supported, there is no hard deadline to migrate. But new architectures, weight-processing controls, and hook refinements are landing on the bridge side — new work should target the bridge, and migrating existing projects is the long-term supported path.
