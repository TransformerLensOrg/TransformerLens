# TransformerLens — Codebase Overview

**TransformerLens** is a mechanistic interpretability library for transformer language models. It loads models from HuggingFace `transformers` and exposes every internal activation through a hook system — users can cache, edit, or ablate intermediate state during a forward pass.

---

## Two Systems

| Aspect | **HookedTransformer** (Legacy) | **TransformerBridge** (v3) |
|---|---|---|
| Status | Legacy, maintenance mode, deprecated in 3.0 | Default for new work |
| Location | `transformer_lens/HookedTransformer.py` + `transformer_lens/components/` | `transformer_lens/model_bridge/` |
| Numerics | Folds LayerNorm into weights — does **not** match HF logits | Raw HF weights — logits match HF exactly |
| Architecture | Reimplements every model from scratch with TL-owned components | Wraps HF submodules; delegates forward to the real HF model |
| Registry | `transformer_lens/supported_models.py` (Python dict) | `transformer_lens/tools/model_registry/data/supported_models.json` |
| Hook naming | Uniform canonical names (`blocks.{i}.hook_resid_pre`, etc.) | Architecture-native names; HT aliases registered on top |
| Acceptance tests | Quarantined (`tests/QUARANTINES.md`) | Active |

Also: `HookedEncoder`, `HookedEncoderDecoder`, `HookedAudioEncoder` — legacy variants.

---

## Core Hook System

### HookPoint (`transformer_lens/hook_points.py`)

An `nn.Module` that acts as an **identity function** by default (`forward(x) → x`). Inserted transparently at every activation site. Key API:

- `add_hook(fn, dir="fwd")` — register a function `fn(tensor, *, hook=HookPoint) → Optional[Tensor]`. Returning a tensor replaces the activation (edit/ablation).
- `add_perma_hook()` — survives `reset_hooks()`.
- `remove_hooks()` — clears hooks in a direction.
- `run_with_cache()` — attaches caching hooks to all `HookPoint`s, runs forward, returns `(output, ActivationCache)`.
- `run_with_hooks()` — runs with temporary hooks that auto-cleanup.

### ActivationCache (`transformer_lens/ActivationCache.py`)

Dictionary-like object storing activation tensors keyed by hook name. Provides `.stack()`, `.fold()`, and direct indexing.

### LensHandle

Dataclass storing the `RemovableHandle`, permanence flag, context level, and original user function.

---

## Model Loading

### Bridge path: `TransformerBridge.boot_transformers()`

1. Look up `<hf_repo>` in the model registry → load `AutoConfig` → build `TransformerBridgeConfig`
2. Detect architecture from `config.architectures[0]` (e.g. `"LlamaForCausalLM"`)
3. Look up in `SUPPORTED_ARCHITECTURES` dict (`architecture_adapter_factory.py`) to find the adapter class
4. Load HF model via `AutoModelForCausalLM.from_pretrained()` with `attn_implementation="eager"`
5. Instantiate the `ArchitectureAdapter` subclass, which declares `self.component_mapping` — a tree of `GeneralizedComponent` objects
6. `TransformerBridge.__init__()` wraps the HF model: traverses the component mapping, looks up each HF path via `get_remote_component()`, installs bridge components that wrap real HF submodules
7. Walk all bridge components, collect their `HookPoint` instances into `_hook_registry`

### Legacy path: `HookedTransformer.from_pretrained()`

Reimplements the entire model from TL's own components (`Embed`, `TransformerBlock`, `Attention`, `MLP`, etc.). Applies weight folding at load time — logits do **not** match HF.

### Native path: `TransformerBridge.boot_native()`

Builds a small randomly-initialized model using TL-native components (no HF dependency).

---

## Adapter Architecture Pattern

Every HF architecture family gets one file in `transformer_lens/model_bridge/supported_architectures/`.

### ArchitectureAdapter (base class in `model_bridge/architecture_adapter.py`)

Each adapter sets in `__init__`:
- `cfg.normalization_type`, `cfg.positional_embedding_type`, `cfg.final_rms`, `cfg.gated_mlp`, `cfg.attn_only`
- `weight_processing_conversions` — einops rearrange patterns for HF → TL weight layout conversion
- `component_mapping` — the tree mapping TL paths to HF paths

### GeneralizedComponent (base class in `generalized_components/base.py`)

Bridge-side wrappers that sit between TL and HF. Each wraps a real HF submodule and provides:
- `hook_in` / `hook_out` — standard `HookPoint` instances
- `get_hooks()` — returns all hook points
- `register_aliases()` — sets up HT-compatible alias names
- `original_component` — reference to the actual HF module it wraps

### Subclasses (42+ files in `generalized_components/`)

`EmbeddingBridge`, `PosEmbedBridge`, `UnembeddingBridge`, `NormalizationBridge`, `RMSNormalizationBridge`, `AttentionBridge`, `JointQKVAttentionBridge`, `MLA_AttentionBridge`, `MLPBridge`, `GatedMLPBridge`, `JointGateUpMLPBridge`, `MoEBridge`, `BlockBridge`, `LinearBridge`, `RotaryEmbeddingBridge`, `BloomAttentionBridge`, `MPTAlibiAttentionBridge`, `MambaBlockBridge`, `T5BlockBridge`, `SSMBlockBridge`, etc.

### Example: GPT-2 adapter

```python
self.component_mapping = {
    "embed": EmbeddingBridge(name="transformer.wte"),
    "pos_embed": PosEmbedBridge(name="transformer.wpe"),
    "blocks": BlockBridge(name="transformer.h", submodules={
        "ln1": NormalizationBridge(name="ln_1"),
        "attn": JointQKVAttentionBridge(name="attn", submodules={
            "qkv": LinearBridge(name="c_attn"),
            "o": LinearBridge(name="c_proj"),
        }),
        "ln2": NormalizationBridge(name="ln_2"),
        "mlp": MLPBridge(name="mlp", submodules={
            "in": LinearBridge(name="c_fc"),
            "out": LinearBridge(name="c_proj"),
        }),
    }),
    "ln_final": NormalizationBridge(name="transformer.ln_f"),
    "unembed": UnembeddingBridge(name="lm_head"),
}
```

### Registration (4 places for each new adapter)

1. `supported_architectures/__init__.py` — import + `__all__` entry
2. `factories/architecture_adapter_factory.py` — `SUPPORTED_ARCHITECTURES` dict
3. `tools/model_registry/__init__.py` — `HF_SUPPORTED_ARCHITECTURES` + `CANONICAL_AUTHORS_BY_ARCH`
4. `tools/model_registry/generate_report.py` — `ARCHITECTURE_DESCRIPTIONS`

---

## Directory Layout

```
transformer_lens/
  HookedTransformer.py              # Legacy HT system
  HookedRootModule.py               # Base class for HT with hook infra
  hook_points.py                    # HookPoint, LensHandle, HookIntrospectionMixin
  ActivationCache.py                # Cache object from run_with_cache
  components/                       # HT-side reimplementations
    attention.py, embed.py, mlps/, transformer_block.py
  model_bridge/                     # Bridge system (v3)
    bridge.py                       # TransformerBridge main class
    architecture_adapter.py         # ArchitectureAdapter base
    component_setup.py              # Wiring bridge components to HF submodules
    generalized_components/         # 43+ bridge-side component wrappers
    supported_architectures/        # 60 adapter files, one per HF architecture family
    sources/                        # Model loading backends (transformers, native)
  factories/                        # Factory helpers
    architecture_adapter_factory.py # SUPPORTED_ARCHITECTURES dict
    mlp_factory.py, activation_function_factory.py
  config/                           # HookedTransformerConfig, TransformerBridgeConfig
  utilities/                        # Device, weight processing, HF utilities
  tools/model_registry/             # Bridge registry + verify_models suite
  patching.py, evals.py             # Activation patching, IOI, evaluation utils
```

---

## Development Conventions

- **uv only** — no pip or poetry (`uv sync`)
- **Source `.env`** before any HF-Hub command
- **Base PRs on `dev`**, never `main`
- **Mirror rule**: if you change HT behaviour with a Bridge counterpart, update both
- **No `# type: ignore`** — use `isinstance` / `typing.cast`
- **No pre-commit hook** — run `make format && uv run mypy .` manually before push
- **Never dismiss a failing test as "pre-existing"** — investigate every failure
