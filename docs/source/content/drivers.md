# Execution Backends (the Driver System)

TransformerLens v4 separates *what you study* (the bridge's hook names, cache, and intervention surface) from *what runs the forward pass* (a **Driver**). Every backend — local HuggingFace `transformers`, vLLM, or an `inspect_ai` provider — satisfies the same protocol, so the same hook names work everywhere; what changes is which hooks each backend can fire and whether gradients exist at all.

This page covers the user-facing surface. The contract lives in [`transformer_lens/model_bridge/driver_protocol.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/dev-4.x/transformer_lens/model_bridge/driver_protocol.py).

---

## The Driver protocol

A driver is anything that implements:

```python
def forward(input_ids, *, capture=(), intervene=None,
            max_new_tokens=1, return_logits=True, **kwargs) -> ForwardResult
def close() -> None
def supports(feature: str) -> bool
```

plus two declared hook-name sets:

- `supported_hook_points` — canonical bridge hook names this backend can fire (e.g. `blocks.0.hook_out`).
- `non_fireable_hook_points` — names the backend structurally cannot serve (fused kernels, sampler shortcuts).

`ForwardResult` carries `logits`, a `captured` mapping of hook name → activation, and the engine's `raw_output`. Tensors are native to the driver's framework and converted at the bridge boundary. `validate_driver` checks the contract when a bridge is constructed.

Interventions come in two dialects:

- **Callables** (`InterventionFn`) — arbitrary Python hook functions, for drivers that can dispatch Python at the engine boundary (the transformers backend).
- **Declarative specs** (`InterventionSpec`) — plain mappings like `{"op": "suppress"}`, for drivers that can't run Python mid-forward (vLLM under `torch.compile`, remote providers).

Capability tiers follow from this: **circuit-finding and anything gradient-based runs on the transformers backend; capture and steering scale out on vLLM; both use the same hook names**, so analyses transfer between them.

---

## The three backends

### transformers — full hooks + gradients

The reference backend. Wraps a local HF `nn.Module`; the full HookPoint tree fires, backward hooks and gradients work, and `parameters()` / `state_dict()` / weight access are all available.

```python
from transformer_lens.model_bridge import TransformerBridge

bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")
logits, cache = bridge.run_with_cache("Hello, world")
```

There is also `TransformerBridge.boot_native(config)`, which builds a small randomly-initialized TL-native model on the same driver — no HuggingFace Hub call — useful for tests and toy-model experiments.

### vLLM — high-throughput capture + declarative interventions

`RemoteBridge.boot_vllm` constructs a vLLM engine (PagedAttention, `torch.compile`, CUDA graphs) and installs capture hooks inside the worker *before* compilation. Activations come back over `collective_rpc` and replay through the bridge's HookPoint tree. This is the throughput path for SAE/probe data collection; unlike observation-only tools (vllm-lens), each hook also applies an affine transform `output = output * scale + bias`, so declarative interventions (`suppress` / `scale` / `add` / `set`) propagate to downstream layers.

```python
import torch
from transformer_lens.model_bridge import RemoteBridge

bridge = RemoteBridge.boot_vllm(
    "meta-llama/Llama-3.2-1B",
    dtype=torch.float16,
    max_model_len=2048,          # cap the KV-cache reservation
)
logits, cache = bridge.run_with_cache("Hello, world")

# Declarative intervention: zero the embedding output for this forward only.
logits2, cache2 = bridge.run_with_cache(
    "Hello, world",
    intervene={"embed.hook_out": {"op": "suppress"}},
)
```

Notes grounded in the source docstrings ([`sources/vllm/source.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/dev-4.x/transformer_lens/model_bridge/sources/vllm/source.py)):

- **Fireable hooks** (decoder-only overlay): `embed.hook_out`, `blocks.{i}.hook_out` / `attn.hook_out` / `mlp.hook_out`, `ln_final.hook_normalized`.
- **Returned logits are reconstructed full-sequence logits**: vLLM's sampler bypasses `lm_head`, so the driver rebuilds them host-side as `ln_final @ lm_head.weight.T` (+ bias, + Gemma soft-cap) — valid at every position, so loss works. If the unembedding weight is unreachable it falls back to final-position log-probs and the bridge rejects `return_type="loss"`.
- **Convention alignment**: vLLM materializes `ln_final` *post-weight*, but the driver un-folds the exposed capture (÷ weight, or ÷ (1 + weight) for Gemma) so `ln_final.hook_normalized` matches the pre-weight value `boot_transformers` serves. If the norm weight is unreachable it warns and serves the raw post-weight value. See [`sources/vllm/overlays/decoder_only.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/dev-4.x/transformer_lens/model_bridge/sources/vllm/overlays/decoder_only.py) for which hooks diverge from HF conventions.
- `enable_batching=True` switches to the eager batched path (`batch_size > 1`, chunked prefill) for data collection; `enable_position_interventions=True` lets a spec carry a `pos` field (int or list) to scope an edit to specific sequence positions.
- `tensor_parallel_size=2` enables single-node tensor parallelism (GPU-validated: capture/intervention/logit parity vs TP=1 within the standard band): captures read from rank 0 — every served hook point is post-all-reduce and replicated — with a first-forward cross-rank check that fails loud if that ever stops holding; vocab-sharded unembeddings are gathered for logit reconstruction. Incompatible with `enable_batching`; pipeline parallelism and multi-node (Ray) remain unsupported.
- Requires a CUDA GPU. Install with `pip install "transformer-lens[vllm]"` (or `uv sync --extra vllm`). The extra is Linux-only (vLLM ships no macOS/Windows wheels), pins the validated `vllm 0.20.x` band — which in turn pins its matching `torch` — and cannot co-install with the `[lit]` extra (numpy version conflict). See [`sources/vllm/internals.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/dev-4.x/transformer_lens/model_bridge/sources/vllm/internals.py) before bumping the band.

### Inspect — interp inside `inspect_ai` evals

`RemoteBridge.boot_inspect` wraps an `inspect_ai` model provider in a bridge, so activation capture and interventions run inside the same harness as behavioral evals. Install with the `inspect` extra (`uv sync --extra inspect`, or `pip install "transformer-lens[inspect]"`).

```python
from transformer_lens.model_bridge import RemoteBridge

bridge = RemoteBridge.boot_inspect("HuggingFaceTB/SmolLM2-135M")  # provider="tl_bridge"
logits, cache = bridge.run_with_cache("Hello, world")
```

From the [`boot_inspect` docstring](https://github.com/TransformerLensOrg/TransformerLens/blob/dev-4.x/transformer_lens/model_bridge/sources/inspect/source.py):

- The default `tl_bridge` provider is HF-backed: residual/attn/mlp capture, full affine interventions, and full-sequence logits. `tl_bridge_vllm` is the vLLM-backed sibling; `provider="vllm-lens"` targets a running vllm-lens provider (residual-only, additive-steering-only).
- **Fireable hooks** (`tl_bridge`): `blocks.{i}.hook_in` (resid_pre), `ln2.hook_in` (resid_mid), `hook_out` (resid_post), `attn.hook_out`, `mlp.hook_out`, plus head-split `attn.hook_q/k/v` / `attn.hook_z` / `attn.hook_pattern` where a per-model structural self-check finds them. `embed`, `ln_final`, and `attn.hook_attn_scores` are always non-fireable — use `boot_transformers()` for those.
- For parity with `boot_transformers`, the provider loads with the same dtype (fp32 by default) and eager attention.
- For capture *during an eval*, add the `capture_activations([...])` solver from [`sources/inspect/eval.py`](https://github.com/TransformerLensOrg/TransformerLens/blob/dev-4.x/transformer_lens/model_bridge/sources/inspect/eval.py) to a Task's solver chain: full activations go to per-sample `.npz` artifacts, and a compact summary lands in the sample store for `samples_df` analysis.

---

## Fundamental limits

Serving engines are not autograd engines. On the vLLM backend (and any future serving backend):

- **No gradients.** PagedAttention and the compiled graph have no autograd surface; backward hooks, attribution patching, and anything gradient-based need `boot_transformers`.
- **No attention patterns or scores.** The QKᵀ → softmax path is fused into the attention kernel (`attn.hook_pattern` / `attn.hook_attn_scores` are declared non-fireable, along with pre/post-RoPE Q/K).
- **Interventions are declarative only.** Arbitrary Python hook functions cannot run inside the compiled worker; the spec vocabulary (`suppress` / `scale` / `add` / `set`, optional `pos`) is the intervention surface.

The Inspect `tl_bridge` provider is HF-backed, so its captures are numerically faithful to `boot_transformers` — but the driver surface is capture/intervene over a wire format, not a local module: no gradients or weight mutation through the bridge.

---

## Verifying parity

Two scripts diff each remote backend against `boot_transformers` on real models, comparing every hook the driver claims to serve:

- `uv run python scripts/vllm_parity_report.py` — GPU-only; boots both backends in fp32 and diffs all fireable capture points (with a refold diagnostic that distinguishes an `ln_final` un-fold regression from a mapping error), plus an argmax agreement check.
- `uv run python scripts/inspect_parity_report.py` — validates the provider's structural self-check: every boundary it offers must match `boot_transformers`.

`demos/vLLM_Bridge_Integration_Test.ipynb` is the end-to-end GPU validation of the vLLM capture *and mutation* path (a manual Colab run, not CI).
