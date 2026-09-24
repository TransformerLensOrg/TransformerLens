# Choosing an analysis tool

Start with the question you want to answer, then choose the measurement. A vocabulary
readout, a geometric similarity score, and an intervention answer different questions,
even when they highlight the same attention head or token.

This guide covers the high-level tools in `transformer_lens.tools.analysis`. For
capturing activations or installing your own interventions, start with the
[hook system](hook_system.md). New experiments should use `TransformerBridge`; see
the [4.0 migration guide](migrating_to_v4.md) for existing `HookedTransformer` code.

## Choose by research question

| I want to know… | Start with | Inputs and outputs | What the result establishes |
|---|---|---|---|
| Which components contribute to an answer's logit? | **Direct Logit Attribution (DLA)** | A prompt or activation cache, answer token, and optional comparison token → labeled logit contributions. | A decomposition of the residual-stream readout for that run. |
| Which activations or paths should I investigate with patching? | **Attribution Patching** | Aligned clean/corrupt token pairs and a differentiable scalar metric → signed node or edge scores. | A first-order estimate of the effect of replacing corrupt activations with clean ones. |
| Does a particular head-to-head route affect my metric? | **Direct Path Patching** | Clean/corrupt caches, a source head, and a metric → destination-head patch scores. | The measured effect of the implemented path intervention, subject to its LayerNorm approximation. |
| What vocabulary directions appear in an MLP's gradient factors? | **Backward Lens** | A prompt, one target token, and selected layers → gradient factors and vocabulary rankings. | A diagnostic of the forward inputs and backward signals composing a weight gradient. |
| How can I read or edit residuals through a fitted transport map? | **Jacobian Lens** | A matching lens artifact and model, plus prompts or activations → vocabulary readouts, decompositions, or interventions. | Readouts under the fitted map; causal effects require running and measuring an intervention. |
| How much do two subspaces overlap? | **Projection Kernel** | Two subspace bases, or Bridge attention-head weight spaces → overlap scores and principal-angle information. | Shared geometric support, independent of the choice of basis within each subspace. |
| What are a head's QK/OV singular directions? | **SVD Circuits: head decomposition** | Model weights and a layer/head index → singular values, vectors, and degeneracy reports. | The linear structure of the head's weight maps. |

These are starting points, not a required pipeline. For example, a weight-space
comparison can motivate a patching experiment, but a large overlap score alone does
not identify the head's function or prove a causal connection.

## Check model requirements before combining tools

A model appearing in the [model tables](model_tables.md) does not imply that every
analysis below supports its architecture. Tools also depend on weight processing,
available hooks, and the meaning of the selected tensor axes.

| Tool | Requirements to check | Main compute or memory cost |
|---|---|---|
| DLA | Targets TransformerBridge in 4.0. Enable compatibility mode and use the standard attention/MLP residual decomposition; Mamba/SSM/Mixer/LinearAttention hybrid layouts are rejected. | One cached forward pass, or reuse of a suitable cache. Head decomposition can require additional per-head results. |
| Attribution Patching | Targets TransformerBridge. Clean/corrupt inputs must be `[batch, seq]` token tensors with matching shapes and aligned positions. The required embedding, head, and MLP hook aliases must exist. Edge granularity additionally requires attention bridges that support per-head results and a pre-Q/K/V residual fork, with `n_key_value_heads == n_heads`; GQA/MQA is not yet supported. Edge sweeps temporarily enable the required writer and reader hooks, then restore the caller's hook flags. Run the model and all nested stochastic modules in evaluation mode. The metric must return a differentiable scalar, and the current backward-cache path requires an active autograd graph; fully frozen models are not yet supported. | Two forwards and one backward **per prompt pair**; activation and gradient caches. For a uniform-head decoder, the number of edge scores grows as `O(seq_len × (n_layers × n_heads)²)`, while the cache also gains per-head `[batch, seq, n_heads, d_model]` tensors. Pair scores are averaged across the batch. |
| Direct Path Patching | Requires a TransformerBridge exposing the expected attention weights, Q/K/V hook aliases, source `hook_z`, and destination LayerNorm scales. Folded LayerNorm parameters improve the approximation. On GQA/MQA models, direct Q-path patching remains available, but `component="k"` and `"v"` currently require `n_key_value_heads == n_heads`; per-query-head K/V semantics are not yet defined. | Clean/corrupt caching, then repeated forwards over destination heads. Sweeping all source heads adds another sweep dimension. |
| Backward Lens | Requires a raw decoder-only TransformerBridge with dense, non-gated MLPs and without compatibility mode, such as GPT-2 or Pythia/GPT-NeoX. Use one target token and check the restrictions in the [tool guide](backward_lens.md). | Gradient computation plus vocabulary projections for selected layers and positions; retaining full logits increases memory. |
| Jacobian Lens | Requires a fresh, causal decoder-only Bridge with raw HF weights, without compatibility mode or weight processing. Validate the lens against the model. Fitting additionally requires all modules in evaluation mode. | Loading an existing artifact avoids fitting. The ordinary fitting estimator uses one forward and `ceil(d_model / dim_batch)` backwards per prompt; larger batches increase memory. |
| Projection Kernel | The numerical API accepts finite, real floating-point matrices via orthonormal bases. The attention-head wrapper requires Bridge weights with compatible dimensions and ranks. | Basis extraction uses SVD. All-head comparisons allocate basis stacks and a pairwise score grid; they can be large despite requiring no forward pass. |
| SVD head decomposition | Uses Bridge per-block weight accessors; requires accessible, compatible `W_Q`, `W_K`, `W_V`, and `W_O` for the selected head. No compatibility mode or activation cache is needed. | Factored QK/OV SVD, with rank bounded by `d_head`; the dense `d_model × d_model` product is not materialized. |

**Compatibility mode is a method-specific choice.** DLA needs it on Bridge, whereas
Jacobian Lens requires raw weights. Use separately loaded models when comparing these
workflows, and collect each cache from the model configuration that will consume it.
Do not reuse a cache after changing weights or processing mode. See
[compatibility mode](compatibility_mode.md) for the numerical conventions.

Evaluation mode and gradient recording are separate choices: `model.eval()` disables
training behavior such as dropout; it does not disable autograd. Do not wrap a
gradient-based analysis in `torch.no_grad()` or `torch.inference_mode()`.

## Read the output with the right interpretation

### DLA: contributions, not ablation effects

Use `direct_logit_attribution(..., unit="component")` for additive component
contributions, or `unit="head"` for heads plus a remainder. With `unit="layer"`,
the entries are **cumulative** residual readouts after sublayers, not independent
contributions to sum together.

An answer-minus-comparison-token direction often makes the question more specific
than a single answer logit. String answers must encode to one token; a leading space
can change tokenization. A complete component decomposition accounts for the
residual contribution, so the unembedding bias `b_U` is excluded. Removing a component
can change downstream computation, so its DLA score is not its ablation effect.

API: {func}`~transformer_lens.tools.analysis.direct_logit_attribution.direct_logit_attribution`.

### Attribution Patching: screen candidates, then measure interventions

At node granularity, the score uses
`(clean_activation - corrupt_activation) · corrupt_gradient`. At edge granularity,
`EdgeAttributionConfig(granularity="edge")` scores each writer-to-reader path
using the writer's activation difference and the gradient at that reader's input;
`edge_scores` and `top_edges()` expose those paths, while `node_scores` aggregates
the outgoing edge scores for each writer. A positive score predicts an increase
in the chosen metric when moving the corrupt activation toward the clean one.
Reversing the metric reverses this interpretation. Large activation changes and
nonlinear downstream behavior can make the estimate inaccurate; compare promising
candidates with actual activation replacements using the [hook system](hook_system.md).

Both node and edge granularity use plain attribution with `ig_steps=1`.
`ig_steps>1` still raises `NotImplementedError`; EAP-IG is not available yet.

API: {func}`~transformer_lens.tools.analysis.attribution_patching.attribution_patch`.

### Direct Path Patching: state the path and its approximation

`get_act_patch_direct_path` fixes a source head and sweeps later destination heads,
patching into their Q, K, or V inputs. It reconstructs the source head's output from
cached `hook_z` and `W_O`, so caching `hook_result` is unnecessary. The calculation
projects the source-output change using cached LayerNorm scaling; it does not
recompute the full nonlinear normalization response to that change. Folded
parameters do not remove this fixed-scale assumption.

Interpret a score relative to the unpatched corrupt metric. Entries at or before the
source layer are zero placeholders, not measured interventions. State the source head,
destination input, prompt pair, and normalization convention when reporting a path.

API: {func}`~transformer_lens.tools.analysis.direct_path_patching.get_act_patch_direct_path`.

### Backward Lens: preserve the gradient sign

Vocabulary rankings describe factors of `d(loss) / d(weight)`. Gradient descent
**subtracts** this gradient. The highest raw-gradient token is therefore not
automatically the token favored by an update. Use the documented bottom rankings
and gradient-descent target ranks where appropriate. Align positions using
`result.prompt_token_ids`, which records the actual tokenization.

See [Backward Lens](backward_lens.md) for the example, sign conventions, and result
structure.

### Jacobian Lens: distinguish readout, reconstruction, and intervention

Start with a matching published artifact when possible and call
`lens.validate_model(model)`. Pin model and artifact revisions. A vocabulary readout
uses the fitted transport map; sparse decomposition describes an activation in its
dictionary. A small reconstruction error does not establish that editing a
coordinate will cause the predicted answer.

For a causal question, install the relevant hooks and measure model outputs against
an unedited baseline and suitable controls. Check the selected intervention's layer
and position semantics: a multi-layer edit is not generally equivalent to independent
single-layer edits.

See [fitting and provenance](jacobian_lens_fitting.md) and the
[decomposition demo](../generated/demos/Jacobian_Lens_Decomposition_Demo.ipynb).

### Weight-space tools: use geometry to form hypotheses

Projection Kernel compares subspaces, without measuring weight magnitude or model
behavior on a prompt. In attention-head results, use `valid_mask` when interpreting
scores: invalid pairs are stored as zeros. See [Projection Kernel](projection_kernel.md)
for rank selection, normalization, and all-head memory costs.

For SVD head decomposition, inspect the rank report before assigning meaning to an
individual direction. Near-equal singular values define a subspace whose basis can
rotate; numerically null directions are also unsuitable for individual attribution.
The weight decomposition alone is not a causal validation of a proposed subfunction.

API: {func}`~transformer_lens.tools.analysis.svd_circuits.decompose_head`.

## Try a geometry question without downloading a model

The following subspaces share exactly one axis. Their raw Projection Kernel score
is 1, and the normalized score is 0.5 because both have rank 2.

```python
import torch

from transformer_lens.tools.analysis import orthonormal_subspace, projection_kernel

axes = torch.eye(3)
first = orthonormal_subspace(axes[:, [0, 1]])
second = orthonormal_subspace(axes[:, [0, 2]])
result = projection_kernel(first, second)

print(result.score.item())       # 1.0
print(result.normalized.item())  # 0.5
```

Replacing these matrices with head-weight bases gives a geometric comparison of
heads. Establishing what those heads do still requires prompts and behavioral
measurements.

## Plan a focused experiment

For a question such as “which components help the model prefer Paris to London?”:

1. Define the prompt set, answer token ids, final position, and signed logit-difference
   metric. Check the unmodified model's answers before interpreting an intervention.
2. Use DLA for a readout decomposition, or aligned clean/corrupt pairs with
   Attribution Patching to shortlist intervention candidates. These scores need not
   agree because they measure different quantities.
3. Replace selected activations and record the observed metric change. Add a no-op
   replacement to check the hook setup and controls matched to the experimental
   claim, such as alternative sites or perturbations of comparable magnitude.
4. Inspect per-prompt results before aggregating. Report failures and exclusions,
   and evaluate the hypothesis on held-out prompts when making a general claim.
5. Record model/tokenizer revisions, dtype, weight-processing mode, token ids,
   hook names, positions, metric definition, and any lens artifact or random seed.

The goal is to connect an interpretable measurement to a clearly specified
experiment, with enough information for another researcher to repeat it.
