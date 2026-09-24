---
title: Migrating to TransformerLens 4.0
---
# Migrating to TransformerLens 4.0

TransformerLens 4.0 **removes the legacy `Hooked*` model stack**.
`TransformerBridge` was introduced in 3.0 as the recommended path, and is now the only path. This guide will detail what was removed and how to recreate those features in the new system. For the deeper conceptual differences between the `HookedTransformer` API and the bridge (hook names, weight processing, per-API recipes), see the [3.0 migration guide](migrating_to_v3.md). The API-mapping table from `HookedTransformer` to `TransformerBridge` is accurate with the final state of all those feature.

Our intention in making this change is to unify future research, as well as future contributions. Having two parallel systems running covering the same ground created much confusion in where features exist, how they should be accessed, and where issues needed to be fixed. Additionally, it created a dual mandate to repair both any time an issue was reported, costing additional man hours.

We will still be accepting issues and resolving any bugs on the existing `HookedTransformer` system, but they will be silo'ed to the 3.x branch. We will not be accepting new features or models.

## The Core Change

```python
# Removed in 4.0
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")

# 4.0
from transformer_lens.model_bridge import TransformerBridge
model = TransformerBridge.boot_transformers("gpt2")
model.enable_compatibility_mode()  # HookedTransformer-equivalent numerics
```

For anyone migrating existing `HookedTransformer` work to the latest system, `enable_compatibility_mode()` reproduces its default weight
processing (LayerNorm folding, `center_writing_weights`, `center_unembed`),
verified against frozen `HookedTransformer` reference activations. Omit it to
work with raw HuggingFace weights (the equivalent of the old
`from_pretrained_no_processing`).

## Removed names and their replacements

| Removed in 4.0 | Replacement |
|---|---|
| `HookedTransformer` | `TransformerBridge.boot_transformers(name)` (+ `enable_compatibility_mode()`) |
| `HookedEncoder` | `TransformerBridge.boot_transformers(name)` on a BERT model |
| `HookedEncoderDecoder` | `TransformerBridge.boot_transformers(name)` on a T5 model |
| `HookedAudioEncoder` | `TransformerBridge.boot_transformers(name)` on a HuBERT/Wav2Vec2 model |
| `BertNextSentencePrediction` | `TransformerBridge.boot_transformers(name, model_class=BertForNextSentencePrediction).predict_next_sentence(a, b)` |
| `HookedTransformerConfig` | `TransformerBridgeConfig` (pass to `TransformerBridge.boot_native(cfg)` for toy/train-from-scratch models) |
| `transformer_lens.train` | `transformer_lens.tools.training` (`train`, `TrainConfig`) |
| `transformer_lens.utils` | `transformer_lens.utilities` — **same names** (e.g. `utils.get_act_name` → `utilities.get_act_name`) |
| `transformer_lens.loading` / `loading_from_pretrained` | Model names/aliases in `transformer_lens.supported_models`; checkpoint labels in `transformer_lens.tools.model_registry.checkpoints`; config derivation is now internal to the bridge's architecture adapters |
| `transformer_lens.components` | `transformer_lens.model_bridge.generalized_components` |
| `transformer_lens.pretrained` (weight converters) | Handled internally by the bridge's adapters; legacy TL-format repos load via `TransformerBridge.boot_tl_legacy(name)` |

Accessing a removed name from the top-level package raises an `ImportError`
naming its replacement, for both `transformer_lens.HookedTransformer` and
`from transformer_lens import HookedTransformer`. (It is an `ImportError` rather
than an `AttributeError` so the message survives the `from ... import ...` form,
which Python's import machinery would otherwise replace with a bare "cannot import
name". A consequence is that `hasattr`/`getattr(..., default)` feature-detection on
these names raises rather than reporting absence.)

## Weight accessors: `W_pos` / `W_E_pos`

4.0 adds direct accessors, so no manual concatenation is needed:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("gpt2")
bridge.enable_compatibility_mode()  # to match HookedTransformer's processed W_pos
bridge.W_pos      # (n_ctx, d_model)
bridge.W_E_pos    # concatenated [W_E; W_pos]
```

Without `enable_compatibility_mode()` these return raw HuggingFace weights;
`HookedTransformer`'s defaults center the writing weights, so match the
processing to match the numbers.

## `train` / `utils`

```python
# Removed
from transformer_lens.train import train, HookedTransformerTrainConfig
from transformer_lens import utils

# 4.0
from transformer_lens.tools.training import train, TrainConfig
from transformer_lens import utilities as utils   # identical names
```

## Traps

- **`HookedEncoder.encoder_output`** does not map to `bridge.encoder_output`.
  The bridge's `encoder_output` is the **audio** frame-entry method
  (HuBERT/Wav2Vec2) and refuses BERT. A name-only migration will pass a `dir()`
  parity check but fail at runtime — run a BERT encoder through
  `bridge(...)` / `run_with_cache` instead.

## Kept

`HookedRootModule` and `HookPoint` are **kept** — they are the supported way to
add TransformerLens-style hooks to an arbitrary `nn.Module`, and are unrelated
to the model-class removal.

## Prerelease note

All three 4.0 prereleases still shipped `HookedTransformer`. If you adopted a
4.0 beta, this final release is a second breaking step for the removed names
above.
