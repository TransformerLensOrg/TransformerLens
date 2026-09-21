# Sources — AGENTS.md

Execution backends for the bridge. Each source boots an engine (HF, TL-native, vLLM, inspect_ai) and wraps it in a `TransformerBridge` or `RemoteBridge` via a Driver. Read [the root AGENTS.md](../../../AGENTS.md) for project-wide rules.

## File map

| File | Role |
|---|---|
| [`../driver_protocol.py`](../driver_protocol.py) | The `Driver` protocol: `forward(input_ids, capture=, intervene=, max_new_tokens=, return_logits=) → ForwardResult`, `close()`, `supports(feature)`, declared `supported_hook_points` / `non_fireable_hook_points`. Also `to_torch` (torch → DLPack → numpy ladder) and `validate_driver` (run at both bridge constructors). |
| [`_driver_base.py`](_driver_base.py) | `DriverBase` — optional ABC with defaults for the protocol's members (`supports`, `close`, `provides_sequence_logits`). Duck-typing works too. |
| [`_hf_format.py`](_hf_format.py) | HF-*format* (not HF-loading) utilities shared by every source whose backend yields an HF-shaped config/tokenizer: `map_default_transformer_lens_config`, `determine_architecture_from_hf_config`, `setup_tokenizer` |
| [`_bridge_builder.py`](_bridge_builder.py) | Loader-agnostic assembly: `build_bridge_config_from_hf`, `build_bridge_from_module`, `detect_tokenizer_bos_eos`; hosts the single `_HF_PASSTHROUGH_ATTRS` list (lines 22–58) |
| [`transformers_driver.py`](transformers_driver.py) | `TransformersDriver` — wraps an HF `nn.Module`; the reference torch driver (gradients, parameters, state_dict, weight access, intervention callbacks) |
| [`transformers/`](transformers/) | HF backend: `source.py` has `boot` (→ `TransformerBridge.boot_transformers`), `helpers.py` has `check_model_support` / `list_supported_models` / checkpoint-revision resolution; `__init__.py` re-exports the historical `sources.transformers` names |
| [`native/`](native/) | TL-native models built from scratch (no HF): `model.py` / `init.py`, booted via `TransformerBridge.boot_native` |
| [`vllm/`](vllm/) | vLLM backend: `source.py` has `boot_vllm` (→ `RemoteBridge.boot_vllm`); `plugin.py` / `worker_extension.py` install capture hooks in the worker pre-compile; `overlays/` declare per-arch capture specs; `internals.py` pins the validated vLLM version |
| [`inspect/`](inspect/) | inspect_ai backend: `source.py` has `boot_inspect` (→ `RemoteBridge.boot_inspect`); ships two providers (`transformers_provider.py` HF-backed, `vllm_provider.py`) plus `eval.py` helpers for capture inside evals. Package name shadows stdlib `inspect` — never `import inspect` bare inside it |

## The Driver system

Every backend satisfies the `Driver` protocol in [`../driver_protocol.py`](../driver_protocol.py): the bridge calls `driver.forward(...)` and reads a `ForwardResult` (logits, captured activations, raw engine output); hook installation is the driver's problem. Drivers declare which canonical hook names they can fire (`supported_hook_points`) and which they structurally cannot (`non_fireable_hook_points`); `validate_driver` enforces the contract at bridge construction.

Implementations:

- `TransformersDriver` — local HF `nn.Module`; full hooks + gradients. Built for you by `TransformerBridge.boot_transformers` (and by `build_bridge_from_module` / `boot_native`).
- `VLLMDriver` ([`vllm/driver.py`](vllm/driver.py)) — high-throughput capture + declarative affine interventions on a vLLM engine; no gradients, no attention patterns.
- `InspectDriver` ([`inspect/driver.py`](inspect/driver.py)) — capture/intervene through an `inspect_ai` provider, for interp inside evals.

Boot entry points: `TransformerBridge.boot_transformers` / `TransformerBridge.boot_native` (torch bridges), `RemoteBridge.boot_vllm` / `RemoteBridge.boot_inspect` (no local `nn.Module`; lazy imports so the heavy backend is only needed by its callers). The `boot_*` methods are attached to the bridge classes by each source's `__init__.py` via `setattr`.

## `_HF_PASSTHROUGH_ATTRS` — one list, one copy point

There is exactly **one** `_HF_PASSTHROUGH_ATTRS` list, at [`_bridge_builder.py:22-58`](_bridge_builder.py). Do not re-introduce a second copy elsewhere (an older layout had a duplicate in a since-deleted `transformers.py`; it's gone on purpose).

Config-translation pipeline (`build_bridge_config_from_hf`):

```
hf_config
  │
  ├─ map_default_transformer_lens_config()      [_hf_format.py]
  │    └─ explicit handlers (d_model, n_heads, head_dim → d_head, etc.) → tl_config
  │
  ├─ TransformerBridgeConfig.from_dict()        [filters to declared fields]
  │
  └─ _HF_PASSTHROUGH_ATTRS copy → bridge_config [_bridge_builder.py:78]
```

**Implications:**

- If an attr has an explicit handler in `map_default_transformer_lens_config` (like `head_dim` → `d_head`), it's translated to a declared `TransformerBridgeConfig` field — adding it to PASSTHROUGH is **wrong** (often raises `AttributeError` from a read-only property).
- If an attr is purely runtime / adapter-specific (like Cohere's `logit_scale`), it has no explicit handler and is dropped by `from_dict` — it MUST be in the PASSTHROUGH list, or the adapter's `getattr(cfg, "<attr>", default)` silently falls back to its default forever.
- See [config/AGENTS.md](../../config/AGENTS.md) for the decision tree.

## `boot_transformers` vs `boot_native`

- `boot_transformers` is the HF path. Goes through `boot()` in [`transformers/source.py`](transformers/source.py) → `build_bridge_config_from_hf` in `_bridge_builder.py` → adapter init → `TransformersDriver`.
- `boot_native` builds a TL-native transformer from a `TransformerBridgeConfig` directly (no HF dependency). Uses [`native.py`](../supported_architectures/native.py) as the single adapter; the model class is in [`native/model.py`](native/model.py).

Almost every contributor change goes through the `boot_transformers` path. Touch `native/` only when adding a primitive (new norm type, new positional embedding variant) that the cfg-driven dispatch in `native/model.py` doesn't already cover.
