# The Hook System

Hooks are the primary value proposition of TransformerLens. They let you intercept, cache, edit, or ablate intermediate activations as a model runs — without modifying the model code. This page covers the user-facing API; for the implementation, see [`transformer_lens/hook_points.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/hook_points.py).

---

## What a hook is

A `HookPoint` is an `nn.Identity` module placed at a named position inside a model (e.g., `blocks.0.attn.hook_q`). At runtime it passes its input through unchanged — but a registered hook function can read the tensor (caching), modify it (intervention), or replace it (ablation, patching).

The hook system is purely PyTorch's `register_forward_hook` / `register_full_backward_hook` underneath, but with two additions that matter:

1. **Named positions.** Every hook point has a string name like `blocks.{i}.hook_resid_post`, so you can address it without traversing the module tree.
2. **Context-scoped lifecycle.** Hooks added inside a `with model.hooks(...)` context auto-remove when the block exits — no manual cleanup required.

---

## Three ways to use hooks

### 1. `run_with_cache` — read everything

The simplest workflow: run a forward pass and get back both the logits and a dict of every cached activation, keyed by hook name.

```python
logits, cache = model.run_with_cache("Hello, world")
cache["blocks.0.attn.hook_q"]           # Q tensor at layer 0
cache["blocks.5.hook_resid_post"]       # residual stream after block 5
cache["ln_final.hook_normalized"]       # post-final-norm activations
```

`cache` is an `ActivationCache` — a dict-like with conveniences (`cache.decompose_resid()`, `cache.apply_ln_to_stack(...)`, etc.). See [`transformer_lens/ActivationCache.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/ActivationCache.py).

For TransformerBridge:

```python
from transformer_lens.model_bridge import TransformerBridge
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
logits, cache = bridge.run_with_cache("Hello, world")
```

Pass `names_filter=` to cache only a subset (saves memory):

```python
logits, cache = model.run_with_cache(
    "Hello, world",
    names_filter=lambda name: name.endswith("hook_resid_post"),
)
```

### 2. `run_with_hooks` — intervene during the forward pass

Attach temporary hooks for one forward pass, then auto-remove them.

```python
def zero_out_head_3(activation, hook):
    activation[:, :, 3, :] = 0   # ablate attention head 3
    return activation

logits = model.run_with_hooks(
    "Hello, world",
    fwd_hooks=[("blocks.5.attn.hook_z", zero_out_head_3)],
)
```

Each hook is a `(hook_name, hook_fn)` tuple. The hook function signature is **always**:

```python
def hook_fn(tensor: torch.Tensor, *, hook: HookPoint) -> Optional[torch.Tensor]:
    # Read or modify `tensor`. Return None to leave the activation unchanged,
    # or return a tensor of the same shape to replace it.
    ...
```

`hook` (the `HookPoint` instance) exposes `hook.name` so a single function can dispatch on which hook called it.

For backward hooks (gradient interventions), use `bwd_hooks=[...]` with the same tuple shape.

### 3. `add_hook` + `remove_all_hook_fns` — manual lifecycle

When you need hooks that persist across multiple forward passes (e.g., during training), drop down to the underlying API:

```python
hook_point = model.blocks[5].attn.hook_z
hook_point.add_hook(my_hook_fn, dir="fwd")           # temporary
hook_point.add_hook(my_hook_fn, dir="fwd", is_permanent=True)  # survives reset

# later
model.remove_all_hook_fns()                          # removes temporary hooks
model.remove_all_hook_fns(including_permanent=True)  # also removes permanent
```

`add_hook` returns nothing useful; lifecycle is owned by the `HookPoint`. The `is_permanent` flag is the only way to survive a `remove_all_hook_fns()` call.

---

## Hook naming

Stable strings; differ between HookedTransformer and TransformerBridge:

| System | Style | Example |
|---|---|---|
| `HookedTransformer` (legacy) | Uniform across architectures | `blocks.5.attn.hook_q`, `blocks.5.hook_resid_post`, `hook_embed` |
| `TransformerBridge` (default) | Architecture-native | `blocks.5.attn.q.hook_out`, `blocks.5.hook_out`, `embed.hook_out` |
| `TransformerBridge` + compatibility mode | Bridge-native AND HT-style aliases | Above + `blocks.5.attn.hook_q` etc. |

Full catalogue: [Main Demo](../generated/demos/Main_Demo), [Exploratory Analysis Demo](../generated/demos/Exploratory_Analysis_Demo). Architecture diagram: [TransformerLens_Diagram.svg](../_static/TransformerLens_Diagram.svg).

Porting HT code to Bridge: `bridge.enable_compatibility_mode()` (see [Compatibility Mode](compatibility_mode.md)) registers HT aliases so existing names resolve.

---

## Common patterns

### Cache one activation, run a single forward pass

```python
logits, cache = model.run_with_cache("text", names_filter="blocks.5.hook_resid_post")
resid_5 = cache["blocks.5.hook_resid_post"]
```

### Zero-ablate a head

```python
def ablate(z, hook):
    z[:, :, head_idx, :] = 0
    return z

model.run_with_hooks("text", fwd_hooks=[(f"blocks.{layer}.attn.hook_z", ablate)])
```

### Activation patching (swap an activation from one prompt into another)

```python
_, clean_cache = model.run_with_cache(clean_prompt)
target = clean_cache["blocks.5.hook_resid_post"]

def patch(resid, hook):
    return target   # replace corrupted's activation with clean's

logits = model.run_with_hooks(
    corrupted_prompt,
    fwd_hooks=[("blocks.5.hook_resid_post", patch)],
)
```

[`transformer_lens/patching.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/patching.py) wraps this pattern for systematic sweeps across layers / positions.

### Gradient intervention via backward hook

```python
def scale_grad(grad, hook):
    return grad * 0.1

model.run_with_hooks(
    "text",
    bwd_hooks=[("blocks.5.hook_resid_post", scale_grad)],
)
```

---

## Lifecycle gotchas

- **Temporary hooks added outside `run_with_hooks` / `model.hooks(...)` do NOT auto-clean.** Call `model.remove_all_hook_fns()` or you'll leak hooks across runs.
- **Permanent hooks (`is_permanent=True`) survive `remove_all_hook_fns()`** — use `including_permanent=True` to clear them.
- **Hook functions that return a tensor replace the activation in-flight.** Returning `None` leaves it unchanged. In-place modification (`tensor[…] = …`) + `return tensor` is the common pattern.
- **Backward hooks see `(grad,)` tuples** at the PyTorch level — the wrapper in `hook_points.py` unwraps to a bare tensor for you. Your hook function still receives a bare tensor.
- **Hooks fire in registration order**, with `prepend=True` to register at the front.

---

## See also

- [Compatibility Mode](compatibility_mode.md) — when to enable HT-style hook aliases on a Bridge model.
- [Migrating to TransformerLens 3](migrating_to_v3.md) — porting HookedTransformer hook patterns to TransformerBridge.
- [Main Demo](../generated/demos/Main_Demo) — end-to-end walkthrough using the hook system.
- [`transformer_lens/hook_points.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/hook_points.py), [`transformer_lens/ActivationCache.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/ActivationCache.py), [`transformer_lens/patching.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/patching.py) — source.
