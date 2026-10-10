# Backward Lens

Backward Lens projects factors of MLP weight gradients into a language model's
vocabulary space. It is useful for inspecting which token directions are associated
with the forward inputs and backward signals that compose a gradient. A readable
vocabulary projection is a diagnostic, not by itself evidence that a token or neuron
causes a model behavior.

TransformerLens supports decoder-only implementations through
`TransformerBridge` whose MLP projections are dense (for example GPT-2 and
Pythia/GPT-NeoX) or gated (for example Qwen2), with a direct MLP contribution
to the residual stream in each requested layer. It follows the method
introduced by [Katz et al. (2024)](https://aclanthology.org/2024.emnlp-main.142/).

## Gradient factorization

For one linear projection and one prompt, let $x_i \in \mathbb{R}^{d_{in}}$ be
the input at token position $i$, and let
$\delta_i = \partial L / \partial y_i \in \mathbb{R}^{d_{out}}$ be the loss
gradient at its output. A dense MLP projection stores its weight either as
`[in, out]` (`Conv1D`, e.g. GPT-2) or as `[out, in]` (`torch.nn.Linear`, e.g.
Pythia/GPT-NeoX). `BackwardLens` reads which layout applies from the Bridge
projection component itself, not from the model class, then computes

$$
\nabla_W L = \sum_i x_i \delta_i^\mathsf{T} = X^\mathsf{T}\Delta,
$$

reporting the result in that projection's own storage layout: directly for
`[in, out]` storage, transposed for `[out, in]` storage.

`BackwardLens` captures both factors and independently computes the weight gradient.
Each matrix result includes the reconstructed gradient and maximum absolute and
scale-aware relative reconstruction errors.

### Dense MLP matrices

The two projections expose different residual-width factors. The weight shapes
below use `Conv1D` (`[in, out]`) storage, as GPT-2 uses; `torch.nn.Linear` storage
(e.g. Pythia/GPT-NeoX) reports each weight transposed.

| Result | Weight shape | Projected factor | Shape before vocabulary projection |
|---|---:|---|---:|
| `input_projection` (FF1 / `c_fc`) | `[d_model, d_mlp]` | Forward input $x_i$ | `[position, d_model]` |
| `output_projection` (FF2 / `c_proj`) | `[d_mlp, d_model]` | Backward signal $\delta_i$ | `[position, d_model]` |

The FF1 readout therefore describes the layer-normalized residual state entering the
MLP (post-`ln_2`, including its gain and bias).
The FF2 readout describes raw loss gradients at the MLP output on supported blocks,
where that output contributes directly to the residual stream. These are different
quantities and should not be interpreted interchangeably.

### Gated MLP matrices

A gated MLP computes $y = \operatorname{down}(\operatorname{act}(\operatorname{gate}(x)) \odot \operatorname{up}(x))$,
so it has three linear projections instead of two. Each projection still
factorizes its own weight gradient as a sum of per-position outer products, and
`BackwardLens` returns a third matrix result, `gate_projection`, alongside
`input_projection` (the up projection) and `output_projection` (the down
projection). The weight shapes below use `torch.nn.Linear` (`[out, in]`) storage,
as Qwen2 uses.

| Result | Weight shape | Projected factor | Shape before vocabulary projection |
|---|---:|---|---:|
| `gate_projection` (`gate_proj`) | `[d_mlp, d_model]` | Forward input $x_i$ | `[position, d_model]` |
| `input_projection` (`up_proj`) | `[d_mlp, d_model]` | Forward input $x_i$ | `[position, d_model]` |
| `output_projection` (`down_proj`) | `[d_model, d_mlp]` | Backward signal $\delta_i$ | `[position, d_model]` |

The gate and up projections both consume the same residual input $x$, so their
forward-input factors and vocabulary logits coincide exactly; the two results are
retained separately because their weight gradients and output gradients differ.
On supported direct-output blocks, the down projection carries the distinct shift
direction, matching the dense FF2 treatment. For dense MLPs, `gate_projection` is
`None`.

## Vocabulary projection

For each residual-width row $v$, the lens computes a fresh readout

$$
P(v) = \operatorname{Unembed}(\operatorname{LN}_{final}(v)).
$$

Final-normalization statistics are recomputed independently for every factor. The
implementation does not reuse normalization scales cached during the model's forward
pass. The retained rankings contain signed, pre-softmax values; they do not contain
probabilities. By default, each matrix keeps only the 10 largest and 10 smallest
values and token ids per position. Set `top_k` to change that bound or
`return_full_logits=True` to also retain the full vocabulary tensors.

When `normalized=True`, the analysis also computes the Normalized Logit Lens:

$$
P_{norm}(v) = P\left(\frac{v}{\lVert v\rVert_2}\right).
$$

Exact-zero rows remain zero before projection and are identified by `zero_norm_mask`.
The original float32 norms are retained in `factor_norms`. Normalization is most useful
when comparing directions whose norms differ greatly, especially very small backward
signals. Because final LayerNorm has an epsilon and may have a bias, raw and normalized
projections need not be identical.

## Sign convention

All backward signals and weight gradients preserve the raw `d(loss) / d(tensor)`
sign. Gradient descent subtracts them:

$$
W_{new} = W - \eta \nabla_W L.
$$

For FF2 backward signals, `bottom(...)` and
`gradient_descent_target_ranks(...)` inspect the smallest raw-gradient logits, which
are often the most relevant ordering for the subtracted update. Do not simply negate
projected logits: final LayerNorm bias and epsilon mean that projection is not exactly
sign-symmetric.

## Minimal example

```python
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import BackwardLens

model = TransformerBridge.boot_transformers(
    "openai-community/gpt2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32,
)

result = BackwardLens(model).analyze(
    prompt="The capital of France is",
    target_token=" Paris",
    layers=[0, 6, 11],
    normalized=True,
  top_k=10,
)

last_layer = result.layer(11)
ff1 = last_layer.input_projection
ff2 = last_layer.output_projection

ff1_tokens = ff1.top_tokens(model.tokenizer, k=5)
ff2_update_tokens = ff2.bottom_tokens(model.tokenizer, k=5)
target_ranks = ff2.gradient_descent_target_ranks(
    result.target_token_id,
    normalized=True,
)
```

`target_token` must encode to exactly one token without a beginning-of-sequence token.
For GPT-2 tokenization, a leading space is often significant. The prompt is tokenized
according to the model and tokenizer configuration, so it includes a prepended BOS
only when that configuration requests one. Treat `result.prompt_token_ids` as the
source of truth for aligning all position-indexed factors and readouts.

The same call works unchanged against a Pythia/GPT-NeoX Bridge, whose `torch.nn.Linear`
MLP projections resolve to the `"out_in"` weight layout instead of GPT-2's `"in_out"`:

```python
pythia = TransformerBridge.boot_transformers(
    "EleutherAI/pythia-70m", device="cpu", dtype=torch.float32
)

pythia_result = BackwardLens(pythia).analyze(
    prompt="The capital of France is",
    target_token=" Paris",
    layers=[0],
)

pythia_result.layer(0).input_projection.factors.weight_layout  # "out_in"
```

The same call also works against a gated Bridge whose MLP exposes separate gate,
up, and down projections and contributes directly to the residual stream. The
result gains a `gate_projection` matrix per layer:

```python
qwen = TransformerBridge.boot_transformers(
    "Qwen/Qwen2-0.5B", device="cpu", dtype=torch.float32
)

qwen_result = BackwardLens(qwen).analyze(
    prompt="The capital of France is",
    target_token=" Paris",
    layers=[0],
)

qwen_result.layer(0).gate_projection is not None  # True
qwen_result.layer(0).gate_projection.factors.weight_layout  # "out_in"
```

## Result structure

`BackwardLens.analyze(...)` returns a detached `BackwardLensResult`:

- `prompt`, `prompt_token_ids`, `target_token`, and `target_token_id` record the inputs.
- `loss` is final-position cross-entropy against the one-token target.
- `layers` preserves the requested layer order; `result.layer(index)` retrieves one.
- Each layer has `input_projection` and `output_projection` matrix results, plus
  `gate_projection` for gated MLPs (`None` for dense MLPs).
- Each matrix exposes `factors`, `factor_norms`, `zero_norm_mask`, `vocabulary_size`,
  and retained raw `top_ranking` and `bottom_ranking` values and token ids.
- `top(...)` and `bottom(...)` return prefixes of the retained signed rankings, so
  their `k` cannot exceed the `top_k` passed to `analyze(...)`.
- `top_tokens(...)` and `bottom_tokens(...)` decode ids with a caller-provided
  tokenizer. Results deliberately retain no model or tokenizer reference.
- The analyzed target's exact ranks are always retained as zero-based competition
  ranks, so tied logits receive the same rank. Ranking another token requires full
  logits.
- `vocabulary_logits` is present only with `return_full_logits=True`.
  `normalized_top_ranking`, `normalized_bottom_ranking`, and normalized target ranks
  are present with `normalized=True`; full `normalized_vocabulary_logits` requires
  both options.
- `includes_normalized_logits` and `includes_full_logits` record the requested modes.
- Maximum reconstruction errors summarize every MLP matrix over all requested
  layers, including the gate projection when present.

Returned tensors are detached, owned CPU copies. Factors, reconstructed gradients,
norms, retained values, and optional vocabulary logits use float32; token ids and
ranks use int64. The bounded default avoids retaining a
`[layer, matrix, position, d_vocab]` collection of full tensors.

## Requirements and non-goals

The current implementation requires:

- A freshly booted, raw `TransformerBridge` whose MLP projections have a
  Bridge-orientable weight layout (`Conv1D`, e.g. GPT-2, or `torch.nn.Linear`,
  e.g. Pythia/GPT-NeoX and Qwen2).
- Original, trainable weights and a dense or gated MLP whose projections are
  distinct linear layers.
- A direct MLP contribution to the residual stream in each requested layer, with
  the adapter mapping `hook_mlp_out` to `mlp.hook_out` and no post-MLP normalization.
- Compatibility mode and weight processing to remain disabled.
- One non-empty prompt, one single-token target, and unique valid layer indices.

Supported model families:

| Family | MLP | Layout |
|---|---|---|
| GPT-2 | dense | `in_out` |
| Pythia / GPT-NeoX | dense | `out_in` |
| Qwen2 | gated | `out_in` |
| Phi-3 | gated, fused gate/up split at boot | `out_in` |

Families whose MLP stores gate and up in one fused matrix are supported when the
Bridge splits that matrix into distinct gate and up projections at boot, as the
Phi-3 adapter does. A gated MLP that exposes no distinct gate projection is
rejected with a clear error.

### Output-boundary restriction

Post-MLP-normalized layouts, such as Gemma 2/3, OLMo 2, EXAONE-4, and GLM-4,
are rejected. Gemma and GLM adapters expose `ln2_post`; OLMo 2 and EXAONE-4
instead map the residual contribution to `ln2.hook_out`. An ordinary pre-MLP
`ln2` does not prevent support. Discovery checks only requested layers, before
tokenization, model execution, or temporary-hook installation, and rejects the
whole analysis if any requested layer has an unsupported output boundary.

For a raw down-projection output $y$ followed by normalization $z=N(y)$,
$\partial L/\partial y = J_N(y)^\mathsf{T}\,\partial L/\partial z$.
The raw output gradient can still factorize the down-weight gradient exactly,
but it is not the gradient at the residual-add boundary. Residual width alone
does not justify projecting it as the FF2 shift. Applying the norm to that
gradient, or enabling `normalized=True`, does not correct this mismatch.

Support is structural, not a family-name allowlist: variants that omit post
norms are evaluated according to their actual Bridge layout. Missing, ambiguous,
or redirected `hook_mlp_out` declarations are also rejected. The check relies on
the adapter's boundary declaration; it does not infer arbitrary transformations
hidden inside custom forward implementations.

It does not currently support batched prompts, multi-token target losses,
mixture-of-experts routing, architecture families whose MLP projections have an
unknown weight layout, post-MLP-normalized output readouts, compatibility-mode
weights, model editing, or causal
claims about the displayed vocabulary rankings.

## Model-state safety

An analysis uses one gradient-enabled forward pass and one `torch.autograd.grad` call.
It does not call `backward()` or modify parameter `.grad` buffers. It preserves model
weights, `requires_grad` flags, train/eval state, existing hooks, and CPU/CUDA/MPS RNG
state, and it removes only its own temporary hooks on success or failure. Existing
activation-editing hooks still affect the analyzed computation.

## Troubleshooting

| Symptom | Cause and resolution |
|---|---|
| Raw-Bridge or processed-weight error | Reboot with `TransformerBridge.boot_transformers(...)`; do not enable compatibility mode or process weights. |
| Target encodes to zero or multiple tokens | Choose text that maps to one token under the model's tokenizer without BOS; check leading whitespace. |
| Duplicate or out-of-range layer error | Pass a non-empty sequence of unique indices in `[0, model.cfg.n_layers)`. |
| Normalized logits were not requested | Call `analyze(..., normalized=True)` before using `logits(normalized=True)` or normalized ranks. |
| Full logits were not retained | Call `analyze(..., return_full_logits=True)` before using `logits(...)` or ranking a token other than the analyzed target. |
| Requested `k` exceeds retained `top_k` | Increase `top_k` in `analyze(...)`; accessor methods cannot recover discarded rankings. |
| FF2 ranking appears sign-reversed | Remember that results are raw loss gradients and gradient descent subtracts them; inspect bottom tokens or ascending target ranks. |
| Results change when custom hooks are installed | Existing hooks are intentionally respected; remove them to analyze the unmodified model computation. |
| Gated MLP exposes no distinct gate projection | The Bridge did not split a fused gate/up matrix into separate projections; only models whose adapter performs that split are supported. |
| Unsupported MLP output boundary | The selected layer applies post-MLP normalization or lacks an unambiguous direct-output declaration. Select a supported direct-output layer/model; changing `normalized` does not fix the boundary mismatch. |

## References

- [TransformerLens Backward Lens demonstration](https://github.com/TransformerLensOrg/TransformerLens/blob/dev/demos/Backward_Lens_Demo.ipynb).
- Shahar Katz, Yonatan Belinkov, Mor Geva, and Lior Wolf. 2024.
  [Backward Lens: Projecting Language Model Gradients into the Vocabulary Space](https://aclanthology.org/2024.emnlp-main.142/).
  *Proceedings of EMNLP 2024*, pages 2390–2422.
- [Authors' research demonstration](https://github.com/shacharKZ/BackwardLens).
  TransformerLens does not import, vendor, or depend on that repository's code.