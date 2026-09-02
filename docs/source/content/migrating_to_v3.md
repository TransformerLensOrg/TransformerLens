# Migrating to TransformerLens 3

TransformerLens 3 introduces **TransformerBridge**, a new way of loading and instrumenting models that replaces `HookedTransformer.from_pretrained` as the recommended path for new code. Existing `HookedTransformer` code continues to run through a compatibility layer, but adopting the bridge unlocks broader architecture support and puts you on the supported path going forward.

This page explains the differences and gives side-by-side migration recipes for the most common patterns.

> **Deprecation status.** `HookedTransformer.from_pretrained` — along with the `HookedEncoderDecoder` and `HookedAudioEncoder` load paths — now emits a `DeprecationWarning`. `HookedTransformer` and the other `Hooked*` classes are slated for removal in 4.0; every feature is being migrated to `TransformerBridge` and the driver system (features that aren't a fit for a driver, such as train-from-scratch, are moving to bridge-based homes rather than staying on `HookedTransformer`). New code should use `TransformerBridge.boot_transformers(...)`. Follow the migration progress in the deprecation plan.

## Why the change?

`HookedTransformer` was a single unified implementation that every supported architecture had to be mapped into. That was beautiful in theory — interpretability code written once worked everywhere — but in practice it meant that adding a new architecture required reimplementing its forward pass inside TransformerLens, and any divergence from the HuggingFace version was a latent source of bugs.

TransformerBridge flips the arrangement. Instead of reimplementing models, it keeps the native HuggingFace implementation and wraps it behind a consistent interface through an **architecture adapter**. The adapter knows how the HF module graph maps onto a small set of generalized components (embedding, attention, MLP, normalization, blocks) and registers uniform hook points over them. The result is the same familiar TransformerLens experience — hooks, caches, patching — but applied to the real HF model, and extended to 140+ architectures out of the box.

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

`move_to_device` and `first_n_layers` are not part of `boot_transformers`. If you relied on either, file an issue describing your use case.

> **Multi-GPU update.** The 3.0 release notes listed `n_devices` among the removed parameters. This feature has since been restored alongside `device_map` and `max_memory`. See [Parameters that are new](#parameters-that-are-new) below.

### Parameters that are new

- `load_weights: bool = True` — set to `False` to construct the bridge with just the config (useful for shape-checking without paying the weight-load cost).
- `trust_remote_code: bool = False` — pass through to HuggingFace for models that ship custom modeling code.
- `hf_config_overrides: dict | None = None` — override specific fields of the HF config before the model is constructed.
- `n_ctx: int | None = None` — override the model's context length. The bridge writes to whichever HF config field this architecture uses (`n_positions` / `max_position_embeddings` / etc.) so callers don't need to know the field name. Warns if larger than the model's default.
- `n_devices` / `device_map` / `max_memory` — multi-GPU loading. `device_map` takes a HuggingFace-style map (`"auto"`, `"balanced"`, or an explicit dict) and is passed straight to `from_pretrained`; `n_devices=N` is a convenience that splits the model across `N` visible CUDA devices (translated to a `max_memory` dict internally); `max_memory` sets a per-device budget. `device` and `device_map` are mutually exclusive.
- `revision` / `checkpoint_index` / `checkpoint_value` — load a specific HF revision, or a training checkpoint for checkpointed families (`EleutherAI/pythia*`, `stanford-crfm/*`). `checkpoint_index` / `checkpoint_value` resolve to a revision string, mirroring the old `HookedTransformer.from_pretrained` checkpoint arguments.
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

### Will my numbers match HookedTransformer?

| Computing | Without `enable_compatibility_mode` | With it |
| --- | --- | --- |
| Generated text, CE loss, argmax / top-k | Identical | Identical |
| Raw logits | Differ by per-row constant | Match |
| Logit lens, direct logit attribution | Differ | Match |
| KL divergence vs another model | Differ | Match |
| Residual-stream norms, cached `hook_resid_*` | Differ (grows with depth) | Match |

Bottom-half analyses → call `enable_compatibility_mode()` after booting.

## Dependency changes in 3.0

TransformerLens 3.0 raises its minimum supported `transformers` to **5.4.0** (previously 4.56). This is enforced automatically, fresh installs and `pip install -U transformer_lens` will pull in a compatible release with no action on your part.

If your code calls `transformers` directly alongside TransformerLens (e.g. manual `AutoModel.from_pretrained` calls in notebooks, or a downstream library that imports both), the v4 → v5 jump may surface breaking changes outside TransformerLens's surface area. See HuggingFace's Transformers v5 release notes for what changed there.

A few v5-driven internal adjustments worth knowing about:

- **Gemma embedding scaling.** Transformers v5 changed how Gemma applies embedding scaling; `enable_compatibility_mode()` compensates so legacy `HookedTransformer` numerics are preserved.
- **MPT block unpack arity.** `MptBlock` returns a 2-tuple on v5 vs a 3-tuple on v4; the bridge adapts.
- **Qwen3.5.** Requires a v5 release exposing `Qwen3_5ForCausalLM`; see [Special Cases](special_cases.md).


## Hook names

The canonical hook names on the bridge use a uniform `hook_in` / `hook_out` convention. The old TransformerLens names are preserved through an alias layer, so existing code keeps working without changes:

```python
# Both of these return the same tensor on a bridge
cache["blocks.0.hook_resid_pre"]  # legacy alias — still works
cache["blocks.0.hook_in"]          # canonical name — preferred for new code
```

For the full mapping of legacy → canonical names and the expected tensor shape at each hook point, see the [Model Structure](model_structure.md) page.

### Hook semantic notes

Two semantic differences inside `enable_compatibility_mode()` worth knowing if you are porting activation-patching, DLA, or attribution-patching code:

- **`blocks.{i}.hook_mlp_in` fires pre-ln2** (matching legacy `HookedTransformer`). Enable it with `bridge.set_use_hook_mlp_in(True)` or `bridge.cfg.use_hook_mlp_in = True`; direct config assignment routes through the same validation and propagation path as the setter. The pre-ln2 placement means cached values from one run can be patched into another and re-flow through `ln2 → mlp` consistently across the bridge and `HookedTransformer`.
- **`hook_q_input` / `hook_k_input` / `hook_v_input` / `hook_attn_in`** also fire pre-ln1 in compat mode. On the per-head LN application that follows, the bridge routes through the raw HF norm rather than the `NormalizationBridge` wrapper, so `ln1`'s sub-hooks (`hook_in`, `hook_normalized`, `hook_scale`) do **not** fire once per head the way legacy `LayerNormPre` would. Q/K/V projections downstream still match legacy numerically; only the intermediate LN sub-hook firing is suppressed.

Post-norm architectures (OLMo 2, BERT-style encoders) and MLA blocks (DeepSeek V2/V3/R1) do not participate in the pre-ln1 capture — `MLABlockBridge` does not expose those aliases, and post-norm models would read the post-attention residual instead of the block input.

Additionally, **HookedRootModule** has been moved to its own module. Prefer `from transformer_lens import HookedRootModule`. The legacy `from transformer_lens.hook_points import HookedRootModule` still works in 3.x, but emits a `DeprecationWarning`. This import path will be removed in 4.0.

`HookedRootModule` itself (together with `HookPoint`) is **kept** — it is permanent infrastructure, not part of the 4.0 removal of the legacy model classes, and remains the supported way to add TransformerLens-style hooks to your own `nn.Module`s.

## APIs that are unchanged

These work identically on `TransformerBridge` and need no migration:

- `to_tokens`, `to_string`
- `generate`
- `run_with_hooks`, `run_with_cache` — including batched-list inputs (parity fixed in 3.x)
- `__call__` / `forward` — accepts both 1D `[seq]` and 2D `[batch, seq]` token tensors
- `cfg.*` — the bridge exposes a `.cfg` with the same fields (`n_layers`, `n_heads`, `d_model`, `d_vocab`, `n_ctx`, ...)
- `W_Q`, `W_K`, `W_V`, `W_O`, `b_Q`, `b_K`, `b_V`, `b_O` — attention weights are exposed with the same `[n_heads, d_model, d_head]` shape conventions

> **Gradients require the transformers driver.** `run_with_cache(incl_bwd=True)`,
> backward hooks, attribution patching, and other gradient-based analyses need
> local autograd, so load the model with `TransformerBridge.boot_transformers(...)`.
> Serving and remote drivers do not expose gradients; see
> {ref}`Fundamental limits <drivers-fundamental-limits>`.

If your code only touches these APIs, the migration is genuinely just the loading call and (optionally) `enable_compatibility_mode`.

### BERT Next Sentence Prediction

NSP runs on the bridge today — load the NSP head via `model_class` and use the
sentence-pair helpers, which own the `token_type_ids` plumbing:

```python
from transformers import BertForNextSentencePrediction
from transformer_lens.model_bridge import TransformerBridge

nsp = TransformerBridge.boot_transformers(
    "google-bert/bert-base-cased",
    model_class=BertForNextSentencePrediction,
)

nsp.predict_next_sentence("A man walked into a grocery store.", "He bought an apple.")
# 'The sentences are sequential'

# Lower level: pair tokenization alone (input_ids + token_type_ids + mask)
tokens = nsp.to_sentence_pair_tokens("A man walked into a grocery store.", "He bought an apple.")
nsp(tokens["input_ids"], token_type_ids=tokens["token_type_ids"], return_type="logits")
```

**Pass `token_type_ids`** when tokenizing by hand. They are what tells BERT where
the first sentence ends and the second begins; without them the NSP head scores a
single undifferentiated span and can return the wrong verdict (on the pair above,
dropping them collapses the logits from [4.36, -4.39] to [1.10, -0.06], and a
genuinely non-sequential pair flips to "sequential").

**Skip `enable_compatibility_mode()` for NSP.** NSP needs no weight processing:
without it the bridge reproduces an independently loaded HuggingFace model's NSP
logits exactly (bit-for-bit against an eager-attention load; ~2e-6 against HF's
default sdpa kernel). Compatibility mode applies unembed centering, which
subtracts the per-input mean of the two logits — ~1.15e-2 on this pair, varying
by input. Both logits shift together, so verdicts are unchanged; exact logit
values are not.

The legacy `BertNextSentencePrediction` wrapper is deprecated and cannot wrap a
`TransformerBridge` — it reaches for `HookedEncoder`-only internals
(`encoder_output`, `pooler`, `nsp_head`). Use the recipe above instead.

### New in 3.x: streaming generation

Both `HookedTransformer` and `TransformerBridge` now expose `generate_stream`, which yields tokens progressively instead of returning the full completion at once:

```python
for chunk in bridge.generate_stream("The quick brown fox", max_new_tokens=50):
    print(chunk, end="", flush=True)
```

Same sampling kwargs as `generate` (`temperature`, `top_k`, `top_p`, `do_sample`, etc.).

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

## Migrating specific `HookedTransformer` APIs

`HookedTransformer` is deprecated and will be removed in 4.0. The compatibility layer keeps existing code running in the meantime, but new work should target `TransformerBridge`, and migrating existing projects is the long-term supported path.

Most `HookedTransformer` methods and properties exist on `TransformerBridge` under the same name — see [APIs that are unchanged](#apis-that-are-unchanged). The table below covers the cases where the name or access path differs.

> If you hit a `HookedTransformer` API whose bridge equivalent isn't obvious and isn't listed here, [open an issue](https://github.com/TransformerLensOrg/TransformerLens/issues); when you (or we) work out the equivalent, add a row below.

Weight-matrix rows return **raw** HuggingFace weights by default. `HookedTransformer.from_pretrained` applies weight processing (LayerNorm folding, `center_writing_weights`, `center_unembed`) at load, so those properties differ numerically from the bridge's unless the bridge is in compatibility mode — see [Will my numbers match HookedTransformer?](#will-my-numbers-match-hookedtransformer).

| `HookedTransformer` | `TransformerBridge` equivalent | Notes |
|---|---|---|
| `model.W_pos` | `bridge.pos_embed.W_pos` | Raw weight (also `bridge.pos_embed.weight`). `center_writing_weights` centers `W_pos` in default HT loads, so it matches HT's only under matching processing (`enable_compatibility_mode()`, or HT loaded with no processing). |
| `model.W_E_pos` | `torch.cat([bridge.W_E, bridge.pos_embed.W_pos], dim=0)` | No single accessor — concatenate the token + positional matrices. Same weight-processing caveat as `W_pos` (both `W_E` and `W_pos` are centered writing-weights). |
| `HookedTransformer.from_pretrained_no_processing(name)` | `TransformerBridge.boot_transformers(name, no_processing=True)` | Both load raw weights, so these match. |
| `model.input_to_embed(...)`; `model(..., start_at_layer=k)` | `bridge.input_to_embed(...)`; `bridge(..., start_at_layer=k)` | The bridge accepts the residual entering block `k`. Embedding-stage hooks are excluded, but blocks `0..k-1` still execute on a discarded path before block `k` swaps in that residual. |
| `model.get_caching_hooks(...)`; `model.add_caching_hooks(...)` | Same methods on `bridge` | Prefer these methods or `run_with_cache` over `cache_all` and `cache_some`, which now emit `DeprecationWarning`. |
| `model.run_with_cache(..., pos_slice=..., incl_bwd=...)` | Same call on `bridge` | `pos_slice` limits cached positions. `incl_bwd=True` requires the gradients-capable transformers driver and a scalar output such as `return_type="loss"`. |
| `model(..., past_kv_cache=cache)` | `bridge(..., past_key_values=cache, return_type="logits_and_cache")` | The returned cache is the HuggingFace-native cache object, not `TransformerLensKeyValueCache`. |
| `model.mod_dict` | `bridge.mod_dict` | Includes the named-module tree plus canonical and HookedTransformer-style hook aliases. See [Hook names](#hook-names). |
| `model.reset_hooks(...)` | `bridge.reset_hooks(clear_contexts=..., direction=..., including_permanent=..., level=...)` | The bridge supports the same scoped hook cleanup controls. |
| A helper typed as `HookedTransformer` | Type it as `TransformerLensModel` or `TransformerLensModelWithWeights` from `transformer_lens.model_protocol` | Use the narrower structural protocol unless the helper reads weights or advanced `ActivationCache` helpers. |
| `HookedTransformerConfig(...)` | `TransformerBridgeConfig(...)` | Construct the bridge config with the same core fields, then pass it to `TransformerBridge.boot_native(config)` for train-from-scratch or toy models. Do not pass a legacy config object to `boot_native`. |
| `cfg.init_weights` | `bridge.init_weights()` | `boot_native()` honors the config flag during construction; call `init_weights()` to reinitialize a TL-native bridge in place. |
| `model.all_head_labels()` | `bridge.all_head_labels` | This is a property on the bridge, so omit the call parentheses. |
| `model.set_tokenizer(tokenizer)` | `TransformerBridge.boot_transformers(name, tokenizer=tokenizer)` | A bridge's tokenizer is fixed when it boots. Reboot to change it; assigning `bridge.tokenizer` directly bypasses tokenizer/config wiring. |
| `from transformer_lens.train import train, HookedTransformerTrainConfig` | `from transformer_lens.tools.training import train, TrainConfig` | The training loop moved to `tools.training` and `HookedTransformerTrainConfig` renamed to `TrainConfig`. The old imports still work but emit `DeprecationWarning`. |

The following example demonstrates the `W_pos` and `W_E_pos` equivalents under matching weight processing:

```python
import torch

from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge

model = HookedTransformer.from_pretrained(
    "gpt2", device="cpu", dtype=torch.float32
)
bridge = TransformerBridge.boot_transformers(
    "openai-community/gpt2", device="cpu", dtype=torch.float32
)
bridge.enable_compatibility_mode()

W_pos = bridge.pos_embed.W_pos
W_E_pos = torch.cat([bridge.W_E, W_pos], dim=0)

assert W_pos.shape == (bridge.cfg.n_ctx, bridge.cfg.d_model)
assert W_E_pos.shape == (
    bridge.cfg.d_vocab + bridge.cfg.n_ctx,
    bridge.cfg.d_model,
)
torch.testing.assert_close(W_pos, model.W_pos)
torch.testing.assert_close(W_E_pos, model.W_E_pos)
```

The equality checks use matching weight processing: `enable_compatibility_mode()` centers the bridge's writing weights in the same way as a default `HookedTransformer` load. A raw bridge load instead matches `HookedTransformer.from_pretrained_no_processing`; see [Will my numbers match HookedTransformer?](#will-my-numbers-match-hookedtransformer) for the broader rule.

The loading, tokenization, generation, and basic hook calls listed under
[APIs that are unchanged](#apis-that-are-unchanged) do not need wrappers. The
recipes below cover the capabilities whose calling convention or runtime
behavior deserves extra attention.

They use a local transformers-backed bridge:

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers(
    "openai-community/gpt2",
    device="cpu",
)
```

### Resume from an intermediate residual stream

Capture the residual entering block `k`, then supply it as the next call's
input:

```python
tokens = bridge.to_tokens("The quick brown fox")
_, cache = bridge.run_with_cache(tokens)

k = 3
residual = cache[f"blocks.{k}.hook_in"]
resumed_logits = bridge(residual, start_at_layer=k)
```

`start_at_layer` excludes embedding-stage hooks and omits blocks below `k` from
the returned cache. It does not skip their computation: the HuggingFace model
still runs blocks `0..k-1` on a path whose result is discarded when block `k`
swaps in `residual`.

### Cache selected positions and gradients

`run_with_cache` is the shortest path for one run. It accepts the same hook-name
filters described under [Hook names](#hook-names):

```python
loss, cache = bridge.run_with_cache(
    "The quick brown fox",
    names_filter=lambda name: name.endswith("hook_out"),
    pos_slice=-1,
    incl_bwd=True,
    return_type="loss",
)
last_position_grad = cache["blocks.0.hook_out_grad"]
```

`incl_bwd=True` calls backward internally, so the output must be scalar; use
`return_type="loss"`. Backward caching is available only on the local
gradients-capable transformers driver. For hooks that should persist across
multiple calls, use `get_caching_hooks` or `add_caching_hooks`, then remove them
with `reset_hooks`. `cache_all` and `cache_some` remain compatibility shims but
emit `DeprecationWarning`.

### Reuse the HuggingFace KV cache

Request `logits_and_cache`, then feed the returned cache into the next call:

```python
tokens = bridge.to_tokens("The quick brown fox")
past_key_values = None

for position in range(tokens.shape[1]):
    logits, past_key_values = bridge(
        tokens[:, position : position + 1],
        past_key_values=past_key_values,
        return_type="logits_and_cache",
    )
```

`past_key_values` is the native cache object produced by the wrapped
HuggingFace model. It is not a `TransformerLensKeyValueCache`, and code should
not depend on the latter's layout or methods.

### Load a legacy TL-format checkpoint

Historical training-run checkpoints (OthelloGPT, grokking demos, ARENA
content) were saved via `HookedTransformer.state_dict()` before the bridge
existed, using property-style keys (`blocks.0.attn.W_Q`, `embed.W_E`, ...)
and per-head tensor shapes that `bridge.load_state_dict` doesn't recognize
natively. `convert_tl_checkpoint` is a one-time converter for exactly this:
convert once, load, then re-save in bridge format — `load_state_dict` itself
stays native-only rather than carrying a second, permanent key convention.

```python
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities.tl_checkpoint_conversion import convert_tl_checkpoint

cfg = TransformerBridgeConfig(...)  # same hyperparameters the checkpoint was trained under
legacy_state_dict = torch.load("othello_gpt.pth")

bridge = TransformerBridge.boot_native(cfg)
bridge.load_state_dict(convert_tl_checkpoint(legacy_state_dict, cfg), strict=True)

torch.save(bridge.state_dict(), "othello_gpt_bridge_format.pth")  # re-save once, done
```

### Type helpers for both model classes

Use the structural protocol when a helper should accept either a
`HookedTransformer` or a `TransformerBridge`:

```python
from transformer_lens.model_protocol import TransformerLensModel


def cache_activations(model: TransformerLensModel, text: str):
    _, cache = model.run_with_cache(text)
    return cache
```

Use `TransformerLensModelWithWeights` instead when the helper also needs the
weight-processing surface used by advanced `ActivationCache` operations.


### Build a TL-native model from scratch

For toy models and interpretability-research training loops, the bridge
replaces `HookedTransformerConfig + HookedTransformer(cfg)` with
`TransformerBridgeConfig + TransformerBridge.boot_native(cfg)`:

```python
# Before
from transformer_lens import HookedTransformer, HookedTransformerConfig

cfg = HookedTransformerConfig(
    d_model=64,
    d_head=8,
    n_heads=8,
    n_layers=2,
    n_ctx=256,
    d_vocab=50257,
    act_fn="gelu_new",
    seed=42,
)
model = HookedTransformer(cfg)

# After
from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge

cfg = TransformerBridgeConfig(
    d_model=64,
    d_head=8,
    n_heads=8,
    n_layers=2,
    n_ctx=256,
    d_vocab=50257,
    act_fn="gelu_new",
    seed=42,               # optional: makes initialisation reproducible
)
bridge = TransformerBridge.boot_native(cfg, device="cpu")
```

`boot_native` makes no HuggingFace Hub call and requires no `transformers`
import. `cfg.seed` seeds the weight initialiser; omitting it lets the
global RNG advance normally. Passing a `HookedTransformerConfig` (or any
other legacy config object) to `boot_native` raises `TypeError` — construct
a `TransformerBridgeConfig` directly. To reinitialize weights in place, call `bridge.init_weights()` — if `cfg.seed` is set,
`init_weights()` rebuilds from that same seed and produces identical weights; clear or
change `cfg.seed` first for a genuinely fresh draw. Unlike `HookedTransformer.init_weights()`,
which calls `torch.manual_seed(cfg.seed)` globally, `boot_native` forks the RNG — training
loops ported verbatim that rely on global seed state for data shuffling will silently
lose reproducibility.
