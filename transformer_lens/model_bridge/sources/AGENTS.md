# Sources — AGENTS.md

Backend loaders that translate external models (HF, native) into a `TransformerBridge`. Read [the root AGENTS.md](../../../AGENTS.md) for project-wide rules.

## File map

| File | Role |
|---|---|
| [`_bridge_builder.py`](_bridge_builder.py) | Loader-agnostic helpers: `build_bridge_config_from_hf`, `build_bridge_from_module`, `detect_tokenizer_bos_eos` |
| [`transformers.py`](transformers.py) | HuggingFace backend: `boot` (entry point used by `TransformerBridge.boot_transformers`), `map_default_transformer_lens_config` (HF→TL config translation), `check_model_support`, `list_supported_models` |
| [`native/model.py`](native/model.py), [`native/init.py`](native/init.py) | TL-native models built from scratch (no HF), used by `TransformerBridge.boot_native` |

## ⚠ The duplicate `_HF_PASSTHROUGH_ATTRS` trap

There are **two** identical `_HF_PASSTHROUGH_ATTRS` lists:

- `transformers.py:481` — inside `map_default_transformer_lens_config`, copies HF-config attrs onto `tl_config`.
- `_bridge_builder.py:18` — module-level, copies HF-config attrs onto `bridge_config` AFTER `TransformerBridgeConfig.from_dict` runs.

Both fire sequentially on the same `hf_config` during `boot_transformers`. **When adding a passthrough attr, add it to BOTH lists** — adding to only one half-fixes the bug. A previous regen-agent run shipped a half-fix that the regression test caught only because we also added an assertion on the canonical bridge config.

## Two-pass config-translation pipeline

```
hf_config
  │
  ├─ map_default_transformer_lens_config()      [transformers.py]
  │    ├─ explicit handlers (d_model, n_heads, head_dim → d_head, etc.)
  │    └─ _HF_PASSTHROUGH_ATTRS copy → tl_config
  │
  ├─ TransformerBridgeConfig.from_dict()        [filters to declared fields]
  │
  └─ _HF_PASSTHROUGH_ATTRS copy → bridge_config [_bridge_builder.py]
```

**Implications:**

- If an attr has an explicit handler in `map_default_transformer_lens_config` (like `head_dim` → `d_head`), it's translated to a declared `TransformerBridgeConfig` field — adding it to PASSTHROUGH is **wrong** (often raises `AttributeError` from a read-only property).
- If an attr is purely runtime / adapter-specific (like Cohere's `logit_scale`), it has no explicit handler and is dropped by `from_dict` — it MUST be in both PASSTHROUGH lists, or the adapter's `getattr(cfg, "<attr>", default)` silently falls back to its default forever.
- See [config/AGENTS.md](../../config/AGENTS.md) for the decision tree.

## `boot_transformers` vs `boot_native`

- `boot_transformers` is the HF path. Goes through `boot()` in `transformers.py` → `build_bridge_from_module` in `_bridge_builder.py` → adapter init.
- `boot_native` builds a TL-native transformer from a `TransformerBridgeConfig` directly (no HF dependency). Uses [`native.py`](../supported_architectures/native.py) as the single adapter; the model class is in [`native/model.py`](native/model.py).

Almost every contributor change goes through the `boot_transformers` path. Touch `native/` only when adding a primitive (new norm type, new positional embedding variant) that the cfg-driven dispatch in `native/model.py` doesn't already cover.
