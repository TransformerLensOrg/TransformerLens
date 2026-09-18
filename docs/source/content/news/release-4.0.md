# TransformerLens 4.0
**[release date TBD]**

TransformerLens 4.0 is here. Where 3.0 changed how models are loaded via the
TransformerBridge and its architecture adapters, 4.0 expands options for **what runs the
forward pass underneath them**. The headline of this release is the new
**execution backends** (we call them Drivers, fed by model **Sources**): the
same bridge you already use with the same hook names, the same cache, the same
intervention surface can now run on **vLLM** for high-throughput capture and
steering, or inside an **`inspect_ai`** evaluation harness.

The other significant change in this release is the removal of the legacy
`Hooked*` classes, completing the transition to TransformerBridge that began in 3.0. Continuing to support two separate implementations of the same tool led to confusion about where features lived, where changes should be implemented, and what a change needed to support. With a new, unified system, there is now a singular home for all future features.

If you have been following the dev-4.x branch, none of this will be a surprise.
If you have been on 3.x, your bridge code carries forward unchanged.

## What changed: the Driver system

In 3.0, the bridge wrapped a local HuggingFace `nn.Module` and layered hook
points over it. That is still the default and the most capable path, but it
limited study to only HuggingFace model executions via `transformers`. If you wanted the throughput
of vLLM, or wanted to capture activations inside an `inspect_ai` eval, you need to write bespoke engine-specific code.

4.0 separates the bridge system from the source of the model. A **Driver** is anything that can run a forward pass and
fire hooks, now the bridge talks to all of them through one protocol. Every backend
declares which hook points it can serve, and the *same canonical hook names*
(`blocks.0.hook_out`, `attn.hook_out`, …) work across all of them. An
analysis written against one backend transfers to another. What changes between
backends is which hooks are fireable and whether gradients exist. This comes down to the tradeoffs of each Driver.

Three backends ship in 4.0:

### transformers — full hooks + gradients (the reference path)

The existing behavior, maintained as it was designed in 3.x. `TransformerBridge.boot_transformers("gpt2")` wraps
a local HF model, the full HookPoint tree fires, backward hooks and gradients
work, and weight access is available. This is the path for circuit analysis,
attribution patching, and anything that needs a real autograd surface. If you
are doing what you did in 3.x, nothing changes for you.

### vLLM — high-throughput capture and steering

`RemoteBridge.boot_vllm(...)` runs the forward pass on a vLLM engine
(PagedAttention, `torch.compile`, CUDA graphs) with capture hooks installed
inside the worker before compilation. Activations come back and replay through
the bridge's HookPoint tree, so `run_with_cache` works exactly as it does
locally. Unlike observation-only tooling, each hook also applies an affine
transform, so declarative interventions (`suppress` / `scale` / `add` / `set`)
propagate to downstream layers.

```python
import torch
from transformer_lens.model_bridge import RemoteBridge

bridge = RemoteBridge.boot_vllm("meta-llama/Llama-3.2-1B", dtype=torch.float16)
logits, cache = bridge.run_with_cache("Hello, world")

# Declarative intervention: zero the embedding output for this forward only.
logits2, cache2 = bridge.run_with_cache(
    "Hello, world",
    intervene={"embed.hook_out": {"op": "suppress"}},
)
```

The vLLM backend requires a CUDA GPU and is installed with the `vllm` extra
(`uv sync --extra vllm`, or `pip install "transformer-lens[vllm]"`). It supports
single-node tensor and pipeline parallelism, both GPU-validated for capture /
intervention / logit parity against the single-rank path.

### Inspect — interpretability inside `inspect_ai` evals

`RemoteBridge.boot_inspect(...)` wraps an
[`inspect_ai`](https://inspect.aisi.org.uk/) model provider in a bridge, so
activation capture and interventions run inside the same harness as your
behavioral evals. The default `tl_bridge` provider is HF-backed and numerically
faithful to `boot_transformers` (residual / attention / MLP capture, full affine
interventions, full-sequence logits); a vLLM-backed sibling is also available.
Install with the `inspect` extra.

```python
from transformer_lens.model_bridge import RemoteBridge

bridge = RemoteBridge.boot_inspect("HuggingFaceTB/SmolLM2-135M")
logits, cache = bridge.run_with_cache("Hello, world")
```

For capture *during* an eval, add the `capture_activations([...])` solver to a
Task's solver chain: full activations land in per-sample artifacts and a compact
summary goes to the sample store for analysis.

### What the backends can and can't do

The capability tiers come from the engines themselves, and the new
[Execution Backends](../drivers.md) page documents them in full. In short:
circuit-finding and anything gradient-based run on `transformers`; capture and
steering scale out on vLLM; both speak the same hook names. Serving engines are
not autograd engines, so the remote backends have no gradients, no attention
patterns or scores (the QKᵀ→softmax path is fused into the kernel), and
interventions there are declarative.

## The Major Deprecation News: `HookedTransformer` has been removed

3.0 introduced the bridge and kept `HookedTransformer` running to cover features that were not yet ported to the bridge system, with the stated long-term intent to remove it in the next
major version. The five legacy model classes —
`HookedTransformer`, `HookedEncoder`, `HookedEncoderDecoder`,
`HookedAudioEncoder`, and `BertNextSentencePrediction` — along with the
supporting stack (`loading_from_pretrained`, `HookedTransformerConfig`, the
`components` tree, the per-architecture weight converters, and the `train` /
`utils` shims) have been deleted.

`TransformerBridge` is now the single model system, and it supports **15,000+
models across 140+ architecture families**. For the vast majority of users the
migration is simple:

```python
# Before (removed in 4.0)
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2")

# After
from transformer_lens.model_bridge import TransformerBridge
model = TransformerBridge.boot_transformers("gpt2")
model.enable_compatibility_mode()   # HookedTransformer-equivalent numerics
```

`enable_compatibility_mode()` reproduces `HookedTransformer`'s default weight
processing (LayerNorm folding, weight centering), verified against frozen
`HookedTransformer` reference activations captured before the removal. For mappings of `Hooked*` features to their new bridge variant, there is a replacement table in
[Migrating to TransformerLens 4.0](../migrating_to_v4.md).

**`HookedRootModule` and `HookPoint` are kept**. They are the supported way to
add TransformerLens-style hooks to an arbitrary `nn.Module` and are maintained for that purpose.

## Breaking changes and deprecations

This major version removes a significant amount of past public API. The specifics:

### The `Hooked*` classes and their stack

Every removed name now raises an `AttributeError` naming its replacement when
accessed from the top-level package (e.g. `from transformer_lens import
HookedTransformer`). See the [4.0 migration guide](../migrating_to_v4.md).

### Known capability reductions

Two narrow capabilities lived only in the deleted `HookedTransformer` attention
path and have no bridge equivalent: Qwen-1's `use_logn_attn` / runtime-adaptive
Dynamic-NTK long-context scaling, and `ungroup_grouped_query_attention`
(per-query-head K/V hook shapes on GQA models).

### Prereleases

All three 4.0 prereleases still shipped `HookedTransformer`. If you adopted a 4.0
beta, this final release is a second breaking step for the names above.

## Roadmap

As with the last two major version announcements, I've broken this into three timeframes. The
mid- and long-term items are a draft and priorities can shift with user feedback.

### Immediate - within the next month

Smoothing rough edges on the new backends. Expect rapid 4.x patches as issues
are reported. I would especially like to hear from anyone running the vLLM
backend on their own hardware — the single-GPU path and the tensor/pipeline
parallel paths are GPU-validated, but real workloads find things a suite of tests doesn't.

### Mid-term - within the next 3 months

- **Broader driver coverage.** The driver protocol is deliberately small so that
  new execution backends can be added without touching the bridge. vLLM and
  Inspect are the first two; other serving engines are candidates. If you have sources you'd like us to add, please file an issue!
- **Documentation and recipes for the remote backends** — Usage examples & suggestions for best fit solutions.

### Long-term - within the next year

- **Deeper multi-node support** for the vLLM backend (Ray-based multi-node is
  currently unsupported).
- **Continued adapter coverage and authoring tooling**, carrying forward the
  3.0 roadmap. The Automated adapter builder via Claude now exists under `devtools/adapter_builder`. We'd like to take this further and try to automate the creation of new adapters as new frontier architectures are released on HuggingFace. Stay tuned for any exciting new developments!

## Contributors

This section is only relevant to contributors; if you use TransformerLens only
as a tool, you can skip it.

### Branch changes

During the 4.x cycle we maintained a `dev-4.x` branch for the driver work and
the `Hooked*` removal alongside the regular `dev` branch. With 4.0 shipping,
that work has landed and `dev` is again the single active development branch.
New pull requests should target `dev`.

### The driver contract

New execution backends implement the `Driver` protocol in
[`driver_protocol.py`](../../../../transformer_lens/model_bridge/driver_protocol.py):
a `forward` that returns a `ForwardResult` (logits + a captured hook-name →
activation map), a `close`, a `supports` capability check, and the two declared
hook-name sets (`supported_hook_points` / `non_fireable_hook_points`).
`validate_driver` checks the contract when a bridge is constructed. Two parity
scripts (`scripts/vllm_parity_report.py` and the Inspect equivalent) diff each
remote backend against `boot_transformers` on real models.

### Dependency changes

`better-abc`, a runtime dependency of the deleted component tree, was dropped.
The `vllm` and `inspect` extras are optional and documented on the
[Execution Backends](../drivers.md) page; `vllm` is Linux-only and cannot
co-install with the `lit` extra.

## Conclusion

Thank you for keeping up with TransformerLens! 3.0 decoupled TransformerLens from any single model
*implementation*; 4.0 decouples it from any single *execution engine*. The hope
is the same as it was a year ago: that the interpretability code you write keeps
working as the field moves, whether you're tracing a circuit on a local model,
collecting a dataset on a vLLM cluster, or instrumenting a model inside an eval.
With the legacy classes now retired, the
library is smaller, more consistent, and hopefully easier to build on.

If you hit a bug or a rough edge while migrating, please open an issue or reach out to me via Slack. I am always happy to help anyone working with TransformerLens create the best possible solution for the project they are working on.
