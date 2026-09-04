# Backward Lens

Backward Lens projects factors of MLP weight gradients into a language model's
vocabulary space. It is useful for inspecting which token directions are associated
with the forward inputs and backward signals that compose a gradient. A readable
vocabulary projection is a diagnostic, not by itself evidence that a token or neuron
causes a model behavior.

TransformerLens currently provides a focused GPT-2 implementation through
`TransformerBridge`. It follows the method introduced by
[Katz et al. (2024)](https://aclanthology.org/2024.emnlp-main.142/).

## Gradient factorization

For one linear projection and one prompt, let $x_i \in \mathbb{R}^{d_{in}}$ be
the input at token position $i$, and let
$\delta_i = \partial L / \partial y_i \in \mathbb{R}^{d_{out}}$ be the loss
gradient at its output. GPT-2 stores `Conv1D` weights in `[in, out]` order, so

$$
\nabla_W L = \sum_i x_i \delta_i^\mathsf{T} = X^\mathsf{T}\Delta.
$$

`BackwardLens` captures both factors and independently computes the weight gradient.
Each matrix result includes the reconstructed gradient and maximum absolute and
scale-aware relative reconstruction errors.

### The two GPT-2 MLP matrices

The two projections expose different residual-width factors:

| Result | Weight shape | Projected factor | Shape before vocabulary projection |
|---|---:|---|---:|
| `input_projection` (FF1 / `c_fc`) | `[d_model, d_mlp]` | Forward input $x_i$ | `[position, d_model]` |
| `output_projection` (FF2 / `c_proj`) | `[d_mlp, d_model]` | Backward signal $\delta_i$ | `[position, d_model]` |

The FF1 readout therefore describes the layer-normalized residual state entering the
MLP (post-`ln_2`, including its gain and bias).
The FF2 readout describes raw loss gradients at the MLP output. These are different
quantities and should not be interpreted interchangeably.

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

## Result structure

`BackwardLens.analyze(...)` returns a detached `BackwardLensResult`:

- `prompt`, `prompt_token_ids`, `target_token`, and `target_token_id` record the inputs.
- `loss` is final-position cross-entropy against the one-token target.
- `layers` preserves the requested layer order; `result.layer(index)` retrieves one.
- Each layer has `input_projection` and `output_projection` matrix results.
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
- Maximum reconstruction errors summarize both MLP matrices over all requested layers.

Returned tensors are detached, owned CPU copies. Factors, reconstructed gradients,
norms, retained values, and optional vocabulary logits use float32; token ids and
ranks use int64. The bounded default avoids retaining a
`[layer, matrix, position, d_vocab]` collection of full tensors.

## Requirements and non-goals

The current implementation requires:

- A freshly booted, raw `TransformerBridge` using `GPT2ArchitectureAdapter`.
- Original, trainable GPT-2 `Conv1D` weights and a dense, non-gated MLP.
- Compatibility mode and weight processing to remain disabled.
- One non-empty prompt, one single-token target, and unique valid layer indices.

It does not currently support batched prompts, multi-token target losses, gated MLPs,
other architecture families, compatibility-mode weights, model editing, or causal
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
| Target encodes to zero or multiple tokens | Choose text that maps to one GPT-2 token without BOS; check leading whitespace. |
| Duplicate or out-of-range layer error | Pass a non-empty sequence of unique indices in `[0, model.cfg.n_layers)`. |
| Normalized logits were not requested | Call `analyze(..., normalized=True)` before using `logits(normalized=True)` or normalized ranks. |
| Full logits were not retained | Call `analyze(..., return_full_logits=True)` before using `logits(...)` or ranking a token other than the analyzed target. |
| Requested `k` exceeds retained `top_k` | Increase `top_k` in `analyze(...)`; accessor methods cannot recover discarded rankings. |
| FF2 ranking appears sign-reversed | Remember that results are raw loss gradients and gradient descent subtracts them; inspect bottom tokens or ascending target ranks. |
| Results change when custom hooks are installed | Existing hooks are intentionally respected; remove them to analyze the unmodified model computation. |

## References

- [TransformerLens Backward Lens demonstration](https://github.com/TransformerLensOrg/TransformerLens/blob/dev/demos/Backward_Lens_Demo.ipynb).
- Shahar Katz, Yonatan Belinkov, Mor Geva, and Lior Wolf. 2024.
  [Backward Lens: Projecting Language Model Gradients into the Vocabulary Space](https://aclanthology.org/2024.emnlp-main.142/).
  *Proceedings of EMNLP 2024*, pages 2390–2422.
- [Authors' research demonstration](https://github.com/shacharKZ/BackwardLens).
  TransformerLens does not import, vendor, or depend on that repository's code.