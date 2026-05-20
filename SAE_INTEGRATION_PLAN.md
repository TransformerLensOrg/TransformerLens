# SAE Hook Integration — TransformerBridge

**Status:** Design proposal (not yet greenlit)
**Context:** Surfaced as a candidate feature for the 2026 community survey. This plan
captures the recommended shape so the implementation isn't over-promised.

## Core thesis

**SAEs are just specialized hooks.** Don't build a parallel API surface — extend the
existing one. A single class composes with every mechanism the bridge already exposes
(`run_with_cache`, `run_with_hooks`, `add_perma_hook`, `list_hooks()`, `generate()`).

## Primitive

```python
class SAEHook:
    """A hook that runs an SAE at a hookpoint with configurable modes."""

    MODES = ("observe", "reconstruct", "intervene")

    def __init__(self, sae, mode="observe", intervene_fn=None):
        # sae: anything with .encode(act) → latents and .decode(latents) → act
        # mode='observe':    cache latents, pass original activation through (zero perturbation)
        # mode='reconstruct': replace activation with sae.decode(sae.encode(act))
        # mode='intervene':  pass latents through intervene_fn before decode
        ...

    def __call__(self, activation, hook):
        latents = self.sae.encode(activation)
        hook.ctx["sae_latents"] = latents
        if self.mode == "observe":
            return activation
        if self.mode == "intervene":
            latents = self.intervene_fn(latents, hook)
        return self.sae.decode(latents)
```

Stashing latents in `hook.ctx` means the existing cache machinery picks them up for free.

## Bridge convenience layer

```python
# Sugar over add_hook + SAEHook construction
bridge.attach_sae("blocks.6.hook_resid_pre", sae, mode="reconstruct")

# Scoped attach (cleans up on exit; mirrors run_with_hooks contract)
with bridge.saes({
    "blocks.6.hook_resid_pre":  (sae6,  "observe"),
    "blocks.10.hook_resid_post": (sae10, "intervene", ablate_feature_42),
}):
    logits, cache = bridge.run_with_cache(tokens)
# cache["blocks.6.hook_resid_pre"]              → activations (as today)
# cache["blocks.6.hook_resid_pre.sae_latents"]  → SAE latents (new, via hook.ctx)
```

Two new methods (`attach_sae`, `saes()`), not four.

## Why this shape fits the bridge

1. **Reuses the hook lifecycle.** No new lifecycle to debug — existing `LensHandle` /
   `add_hook` / `remove_hooks` flow handles attach/detach/scoping.
2. **Generation-aware for free.** `bridge.generate()` already invokes hooks per-token;
   SAE hooks fire correctly without extra plumbing.
3. **Composes with `list_hooks()` / `HookPoint.__repr__`** (landed for #297) — researchers
   can introspect which SAEs are currently attached without a separate API.

## The hard part: KV cache + SAE

If an SAE replaces residual-stream activations mid-generation, the cached K/V from
earlier positions is now inconsistent. Two options:

- **Strict** *(v1 default)*: invalidate KV cache when any SAE in `reconstruct` or
  `intervene` mode is attached to a pre-attention hookpoint. Slower (recompute full
  prefix per generation step) but correct.
- **Lenient** *(v2 opt-in)*: leave KV cache alone, document that SAE-modified prefixes
  are not retroactively re-propagated. Faster, but generation post-attach reflects
  partial state.

Surface as `bridge.attach_sae(..., kv_cache="strict"|"lenient")` once researchers
complain about strict-mode speed.

## Compatibility constraints

- **SAE Lens interop.** The `sae` parameter is duck-typed on `.encode(act)` and
  `.decode(latents)` — SAE Lens's `SAE` class satisfies this. Don't require a new SAE
  base class.
- **Hookpoint name flexibility.** Accept both bridge-native names and legacy HT names
  (`blocks.6.hook_resid_pre`). Bridge's compatibility-mode aliases already handle this.
- **Detach contract.** SAE parameters must be detached from the computation graph
  during inference by default; opt-in `requires_grad=True` for joint training scenarios.

## Implementation footprint

| Piece | Approx. LoC |
|---|---|
| `SAEHook` class | ~80 |
| `bridge.attach_sae` / `saes()` context manager | ~60 |
| KV-cache invalidation handling | ~50 |
| Tests (toy SAE on Pythia-70m + GPT-2) | ~250 |
| Demo notebook | (separate) |
| **Total** | **~440 + tests** |

## v1 acceptance checklist

- [ ] `SAEHook` class with three modes
- [ ] `bridge.attach_sae(hookpoint, sae, mode=...)` → returns `LensHandle`
- [ ] `bridge.saes({...})` context manager
- [ ] SAE latents flow into `run_with_cache` output (no separate `run_with_sae_cache`)
- [ ] Strict KV-cache invalidation under `reconstruct` / `intervene`
- [ ] Compatible with SAE Lens's `SAE.encode` / `SAE.decode` interface (no adapter required)
- [ ] Tests: toy SAE on Pythia-70m exercising all three modes + generation + cache
- [ ] Demo notebook showing observe / reconstruct / feature-ablation flows

## Out of scope for v1

- SAE *training* through the bridge — supported via existing backward hooks, but no
  dedicated training API.
- Activation steering as a separate primitive — overlaps with `intervene` mode; ship
  as its own feature only if usage patterns diverge.
- Disk-cached activation datasets for SAE training — better as a standalone tool that
  consumes `run_with_cache` output.
- Cross-layer SAE composition (transcoders) — same primitive works, but the
  bookkeeping (which SAE feeds which) deserves a separate design pass.

## Survey framing

If "native SAE hook integration" appears as a survey option, this is the actual
scope behind it: ~440 LoC of bridge plumbing that turns SAEs into composable hook
primitives. Not a giant new subsystem. The honest uncertainty to disclose is the
KV-cache mode choice; everything else is well-bounded.
