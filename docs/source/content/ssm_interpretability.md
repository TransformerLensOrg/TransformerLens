# SSM / Recurrent-Model Interpretability

TransformerBridge exposes a family-agnostic interpretability surface for state-space
and linear-attention models — Mamba-1, Mamba-2, gated-delta-net (Qwen3.5 / Qwen3-Next),
and the SSM hybrids (NemotronH, GraniteMoeHybrid). The same three `ActivationCache`
methods work across every family, dispatching to each block's SSM mixer regardless of
which slot it occupies (`.mixer` for Mamba, `.linear_attn` for gated-delta-net) and
skipping a hybrid's passthrough (attention / MLP / MoE) layers automatically.

> These features are **bridge-only** — there is no `HookedTransformer` counterpart for
> SSM models. Load with `TransformerBridge.boot_transformers(...)`.

---

## Discovery

```python
bridge = TransformerBridge.boot_transformers("state-spaces/mamba-130m-hf")
_, cache = bridge.run_with_cache(tokens, use_cache=False)

cache.ssm_layers()          # -> [0, 1, 2, ...]  block indices whose mixer is recurrent
```

`ssm_layers()` is purely structural (no dependence on `cfg.layers_block_type`): a
hybrid's attention/MLP layers are excluded, so on NemotronH you get only the Mamba
layers.

## Read-only analysis (no forward re-run)

Both methods reconstruct their quantities post-hoc from cached hooks. They return a
single tensor when *every* block is an SSM layer, else a `{layer_idx: tensor}` dict
over the SSM layers.

```python
# Effective ("hidden") attention  M = L ⊙ (C Bᵀ)   [batch, heads, seq, seq]
M = cache.compute_ssm_effective_attention()          # all SSM layers
M0 = cache.compute_ssm_effective_attention(layer=0)  # one layer

# Recurrent state trajectory S_t
S = cache.compute_ssm_state()                        # all SSM layers
S5 = cache.compute_ssm_state(layer=5, time_step=-1)  # one layer, final step (memory-bounded)
```

State shape is family-specific (Mamba-1 is per-channel; Mamba-2 and gated-delta-net
are per-head), so `compute_ssm_state()` returns a dict except when one mixer type
covers every block.

> **`use_cache=False` is required for gated-delta-net.** Its interior hooks
> (`hook_q/k/v`, `hook_beta`, `hook_log_decay`) fire only on the hooked prefill path;
> the default cached path exposes only `hook_in`/`hook_out`, and the reconstruction
> methods then raise. Mamba reconstructs from `in_proj`/`conv1d` hooks either way, but
> passing `use_cache=False` uniformly is safe.

## Canonical hook vocabulary

Every family exposes the same state-mutation names (defined once, in
`SSMStateHookMixin`), so interp tooling can address the same quantity across families:

| Hook | Meaning | Kind |
|---|---|---|
| `hook_ssm_state` | post-scan recurrent-state trajectory `S_t` | real HookPoint (fires on the eager-scan path) |
| `hook_ssm_write` | per-step write influence | **real** for Mamba-1/2 (`dt·(x⊗B)`); **alias → `hook_beta`** for gated-delta-net (state-dependent write) |

Additional canonical aliases resolve per family where the quantity exists:
`hook_ssm_out`, `hook_ssm_B`, `hook_ssm_C`, `hook_ssm_decay`, `hook_ssm_dt`.

---

## Causal intervention: the `eager_scan` opt-in

By default the mixer runs HF's fused recurrence kernel, which is opaque — the state
trajectory is never materialized, so it cannot be patched. Setting `eager_scan = True`
on a realized SSM mixer swaps the kernel for a readable Python scan that fires
`hook_ssm_state` (and, for the input-linear families, `hook_ssm_write`), so you can
read and edit the state mid-recurrence.

```python
from transformer_lens.model_bridge.generalized_components.ssm_protocol import find_ssm_mixer

# Enable on every realized SSM mixer (skips hybrid passthrough slots).
mixers = [m for b in bridge.blocks if (m := find_ssm_mixer(b)) is not None]
for m in mixers:
    m.eager_scan = True

# hook path uses the mixer's slot name: `.mixer` (Mamba) or `.linear_attn` (gated-delta-net)
STATE = "blocks.0.mixer.hook_ssm_state"

def zero_state(state, hook):
    return torch.zeros_like(state)   # ablate the whole recurrent state

logits = bridge.run_with_hooks(tokens, use_cache=False, fwd_hooks=[(STATE, zero_state)])

for m in mixers:
    m.eager_scan = False             # restore the default (bit-identical) fused path
```

**Two intervention semantics** (mirroring Mamba-Knockout and state-patching):

- Patching **`hook_ssm_state`** changes only the *same-position* readout
  (`y_t = C_t·S_t` or `S_t^T q_t`); it does not alter the forward recurrence.
- Patching **`hook_ssm_write`** re-runs the recurrence, so the edit **propagates** to
  every later state. For gated-delta-net this is the write-strength gate
  (`hook_beta`), so zeroing it suppresses all writes.

### Caveats

| | |
|---|---|
| **Prefill only** | The eager scan runs when `cache_params is None` (i.e. `use_cache=False`, no generation step). Autoregressive decode falls back to the fused kernel. |
| **Numerical** | The Python scan matches the fused kernel only to floating-point tolerance (≈1e-6 fp32), never bit-for-bit. |
| **Cost** | O(seq) Python with an O(batch·seq·…) state tensor — orders of magnitude slower/heavier than the kernel. Use short sequences. |
| **Default untouched** | With `eager_scan = False` (the default), `run_with_cache` is bit-identical to HF and `hook_ssm_state` does not fire. |

> See [`ssm2_mixer.py`](../../../transformer_lens/model_bridge/generalized_components/ssm2_mixer.py),
> [`ssm_mixer.py`](../../../transformer_lens/model_bridge/generalized_components/ssm_mixer.py), and
> [`gated_delta_net.py`](../../../transformer_lens/model_bridge/generalized_components/gated_delta_net.py)
> for the per-family recurrences and reconstruction identities.
