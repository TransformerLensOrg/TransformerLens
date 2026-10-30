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
| Which activation, if restored from the clean run, recovers the answer, and how sure am I per cell? | **Activation Patching** | Corrupted tokens, a clean cache and a metric → a grid of patched-metric values; with `reduce="none"` the per-example values plus mean, standard error, bootstrap interval and permutation p-value per cell. | The measured effect of one causal intervention per cell, with its uncertainty across the prompt batch. |
| Which activations or paths should I investigate with patching? | **Attribution Patching** | Aligned clean/corrupt token pairs and a differentiable scalar metric → signed node or edge scores. | A first-order estimate of the effect of replacing corrupt activations with clean ones. |
| Does a particular head-to-head route affect my metric? | **Direct Path Patching** | Clean/corrupt caches, a source head, and a metric → destination-head patch scores. | The measured effect of patching that one path, with every other reader of the source head left on its corrupted output. |
| What does each layer "predict", and where does a token's rank change? | **Logit Lens** | A prompt or activation cache, optional target tokens, top-k or a vocabulary subset → per-layer logits, log-probabilities, ranks and entropy. | What the residual stream reads as through the real final norm and unembedding; the final entry reproduces the model's own logits. |
| How can I read or edit residuals through a fitted transport map? | **Jacobian Lens** | A matching lens artifact and model, plus prompts or activations → vocabulary readouts, decompositions, or interventions. | Readouts under the fitted map; causal effects require running and measuring an intervention. |
| Does a subspace at one hook point carry a variable the output depends on? | **Interchange Interventions** | A hook point, base and source inputs, a basis (fixed, a set of MLP neurons, or learned by DAS from pairs labelled by a high-level model), and a task readout → task logits with the base's subspace coordinates set to the source's. | The effect of that intervention on the chosen pairs; compare a learned subspace with its trained controls (arbitrary labels, an untrained model), a norm-matched placebo and random bases before reading it as use. |
| How much do two subspaces overlap? | **Projection Kernel** | Two subspace bases, or Bridge attention-head weight spaces → overlap scores and principal-angle information. | Shared geometric support, independent of the choice of basis within each subspace. |

These are starting points, not a required pipeline. For example, a weight-space
comparison can motivate a patching experiment, but a large overlap score alone does
not identify the head's function or prove a causal connection.

## Check model requirements before combining tools

A model appearing in the [model tables](model_tables.md) does not imply that every
analysis below supports its architecture. Tools also depend on weight processing,
available hooks, and the meaning of the selected tensor axes.

| Tool | Requirements to check | Main compute or memory cost |
|---|---|---|
| DLA | Targets TransformerBridge in 4.0. Enable compatibility mode and use the standard attention/MLP residual decomposition; Mamba/SSM/Mixer/LinearAttention hybrid layouts are rejected, as are bridges whose readout is not `ln_final -> unembed` on the last block (multi-stream AltUp blocks, a head transform such as BERT's `mlm_head`). When DLA runs the prompt, the model and all submodules must be in evaluation mode, or it raises `ValueError` before the forward; a supplied cache is accepted as is. | One cached forward pass, or reuse of a suitable cache. Head decomposition can require additional per-head results. |
| Activation Patching | Any bridge whose activation names resolve (`resid_pre`, `attn_out`, `mlp_out`, `z`, `q`/`k`/`v`, `pattern`). The metric returns a scalar by default and a `[batch]` tensor under `reduce="mean"` or `"none"`; several metrics can be passed as a mapping and share one sweep. Run the model in evaluation mode. | One forward of the corrupted batch per swept cell (plus one for `baseline=True`); per-example storage is `cells × batch` floats per metric. Statistics are computed afterwards from the stored values and add no forwards. |
| Attribution Patching | Targets TransformerBridge. Clean/corrupt inputs must be `[batch, seq]` token tensors with matching shapes and aligned positions. The required embedding, head, and MLP hook aliases must exist. Edge granularity additionally requires attention bridges that support per-head results and a pre-Q/K/V residual fork. On GQA/MQA models, where any attention layer's K/V head count (`per_layer_num_key_value_heads` where set, else `n_key_value_heads`) differs from `n_heads`, edge granularity and faithfulness raise `ValueError` before any forward; node granularity remains available. Edge granularity and faithfulness are unsupported on post-norm blocks (such as Gemma 2/3, OLMo 2/3 and GLM-4), scaled-residual blocks (such as Granite) and parallel attention/SSM blocks (such as Falcon-H1), where per-head outputs are normalized, scaled or joined by an unmodeled mixer before the residual add; both raise there, and node granularity remains available. Edge sweeps temporarily enable the required writer and reader hooks, then restore the caller's hook flags. Run the model and all nested stochastic modules in evaluation mode. The metric must return a differentiable scalar, and the current backward-cache path requires an active autograd graph; fully frozen models are not yet supported. | Two forwards and one backward **per prompt pair**; activation and gradient caches. For a uniform-head decoder, the number of edge scores grows as `O(seq_len × (n_layers × n_heads)²)`, while the cache also gains per-head `[batch, seq, n_heads, d_model]` tensors. Pair scores are averaged across the batch. |
| Direct Path Patching | Requires a TransformerBridge whose attention layers have the pre-Q/K/V residual fork that `use_split_qkv_input` enables (the joint-QKV and position-embedding attention bridges, e.g. GPT-2, Pythia, Llama, Mistral, Qwen2); attention without it, such as GPT-J, GPT-Neo, OPT, BERT or a bridge built with `TransformerBridge.boot_native`, raises `NotImplementedError`. The supplied caches must hold the source layer's `hook_z`. The function turns the fork flag on for the sweep and restores the caller's `use_split_qkv_input` and `use_attn_in`; LayerNorm need not be folded. Q destinations use query heads; K/V destinations use native KV heads, shared by groups of query heads on GQA/MQA. K/V sweeps require a uniform KV count across destination layers (`per_layer_num_key_value_heads` where set, else `n_key_value_heads` or `n_heads`), and raise `ValueError` before any forward if those counts differ. Every block from the source layer on must have an `attn` component; SSM, mixer, linear-attention and AltUp blocks raise `ValueError` before any forward. Source blocks must add the head outputs to the residual unchanged: post-norm blocks (such as Gemma 2/3, OLMo 2/3 and GLM-4), scaled-residual blocks (such as Granite) and parallel attention/SSM blocks (such as Falcon-H1) raise `ValueError` before any forward, since a head's residual write there is not `hook_z @ W_O`. Run the model and all submodules in evaluation mode; both entry points raise `ValueError` before any forward otherwise. | Clean/corrupt caching, then one forward of the corrupted batch per later destination query head or KV group. While the fork is on, every attention layer normalizes `n_heads` Q copies and `n_kv_heads` copies each for K and V, about `n_heads + 2 × n_kv_heads` times the usual attention-input activation memory. Sweeping all source heads adds another sweep dimension. |
| Logit Lens | Targets TransformerBridge, raw or compatibility mode. Needs a decoder with an `ln_final` component (or no final norm) and `W_U`; encoder-decoder and encoder-only bridges are rejected, as are quantized unembeddings and bridges whose readout is not `ln_final -> unembed` on the last block (multi-stream AltUp blocks, a head transform such as BERT's `mlm_head`; a prompt run with `apply_ln=True` is also checked against the model's own output, a supplied cache only structurally). Hybrid SSM stacks work with `incl_mid=False`. Run the model in evaluation mode. | One cached forward pass over the residual hooks, or reuse of a cache. Logits are produced in chunks of `chunk_size` rows; full-vocabulary output is materialized only when no `targets`, `top_k` or `vocab` selector is given. |
| Jacobian Lens | Requires a fresh, causal decoder-only Bridge with raw HF weights, without compatibility mode or weight processing. Validate the lens against the model. Run the model and all submodules in evaluation mode; fitting and every call that runs the model raise `ValueError` otherwise. | Loading an existing artifact avoids fitting. The ordinary fitting estimator uses one forward and `ceil(d_model / dim_batch)` backwards per prompt; larger batches increase memory. |
| Interchange Interventions | Any model with the hook API (TransformerBridge or `HookedRootModule`). Activations at the hook must be `[batch, d_model]`, or `[batch, pos, d_model]` with one position index per example; base and source rows are paired. | One forward pass per `batch_size` sources to collect activations, then one per `batch_size` bases per intervention with `intervened_logits`; both run without gradients, so memory is bounded by `batch_size`. `run_with_intervention` is a single differentiable pass over all the bases it is given. DAS adds a forward and backward pass per batch per epoch; `micro_batch_size` bounds its memory. |
| Projection Kernel | The numerical API accepts finite, real floating-point matrices via orthonormal bases. The attention-head wrapper requires Bridge weights with compatible dimensions and ranks. | Basis extraction uses SVD. All-head comparisons allocate basis stacks and a pairwise score grid; they can be large despite requiring no forward pass. |

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

### Activation Patching: per-example cells, then intervals

Every `get_act_patch_*` sweep runs the whole corrupted batch for each patched
cell, so the per-example values are free. Pass a metric that returns a `[batch]`
tensor and `reduce="none"` to keep them: the result is a `PatchingResult` whose
`values` end in a batch axis, with `mean()`, `stderr()`, `bootstrap_ci()` and, when
the sweep was run with `baseline=True`, `effect()` (patched minus unpatched
corrupted run) and `permutation_pvalue()`. `reduce="mean"` averages a `[batch]`
metric back to the plain tensor the sweeps have always returned, and a mapping of
metrics is evaluated on each patched forward so two metrics cost one sweep.

Read the statistics for what they are. The bootstrap interval describes how the
cell's mean would move across resamples of *these* prompts; with four to eight
prompts it is wide, and a constant column collapses it to a point. The sign-flip
permutation test asks whether the per-example effects are symmetric about zero;
it is exact for twelve or fewer prompts and a Monte Carlo estimate above that.
Neither says anything about prompts outside the batch, and neither changes what a
single cell establishes: the measured effect of one intervention.

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

#### Faithfulness: check that a circuit actually explains the behavior

A ranked edge list is a hypothesis, not a result. `faithfulness()` tests one aligned
clean/corrupt pair at a time by ablating every edge *outside* the candidate circuit
and reporting how much of the clean-to-corrupt metric gap the circuit recovers:

```python
from transformer_lens.tools.analysis import (
    EdgeAttributionConfig,
    attribution_patch,
    faithfulness,
)

ranked = attribution_patch(
    model, clean, corrupt, metric_fn, config=EdgeAttributionConfig(granularity="edge")
)
report = faithfulness(model, clean, corrupt, metric_fn, ranked.top_edges(k=50))
print(report.recovered, report.circuit_size, report.total_edges)
```

`recovered` is a fraction, not a percentage: `1.0` means the circuit reproduces
the clean metric, `0.0` means it reproduces the corrupt metric. Values outside
`[0, 1]` are possible and meaningful, since a circuit can overshoot.

The residual stream is a running sum, so each reader's input is rebuilt exactly
by subtracting the excluded writers' live contributions and adding their
replacements. This is a forward-only measurement: it takes no gradient, so it is
independent of the linearization that ranked the edges in the first place.

`FaithfulnessConfig(ablation=...)` selects the replacement. The default,
`"corrupt"`, substitutes the corrupt run's own contribution, which is the
evaluation the external EAP-IG reference reports. `"mean"` requires an explicit,
held-out calibration batch through the keyword-only `mean_tokens` argument:

```python
from transformer_lens.tools.analysis import FaithfulnessConfig

calibration_tokens = model.to_tokens([
    "The capital of Spain is",
    "The capital of Italy is",
]).to(clean.device)
mean_report = faithfulness(
    model, clean, corrupt, metric_fn, ranked.top_edges(k=50),
    config=FaithfulnessConfig(ablation="mean"),
    mean_tokens=calibration_tokens,
)
```

Calibration must be a nonempty `[n_calibration, seq]` batch of `torch.int32` or
`torch.int64` token ids, with the evaluated sequence length and the same device
as the clean/corrupt token tensors. Align the sequence positions before averaging;
the helper averages examples only, preserving each position and attention head.
It captures calibration activations with the same model, hook configuration, and
weight-processing mode used for evaluation.

Mean-mode calls without calibration now raise `ValueError`; there is no fallback
to the evaluated pair. Exact calibration rows equal to either endpoint are also
rejected. This check cannot establish dataset independence: callers must choose
held-out examples and avoid semantically equivalent leakage. Passing `mean_tokens`
in corrupt mode is rejected rather than silently ignored.

Recovery is still `(circuit_metric - corrupt_metric) / (clean_metric - corrupt_metric)`.
An empty mean-ablated circuit can have nonzero recovery because its calibration
baseline need not reproduce the corrupt run. A full circuit still reproduces the
clean metric. Compare against a random edge set of the same size, using the same
calibration batch, before claiming a circuit is meaningful.

`report.edge_class_recovered` is circuit-relative leave-one-out recovery. For a
candidate `C` and graph class `E_t`, each entry evaluates `C` with its retained
edges in `E_t` removed, not the full graph with the class removed. Edges originally
outside `C` stay ablated. Every diagnostic uses the aggregate's replacements,
clean/corrupt endpoints, and normalization:

```text
edge_class_recovered[t] = recovered(C minus E_t)
```

An absent class leaves the candidate unchanged, so its entry equals
`report.recovered`. For an empty circuit every class entry equals aggregate
empty-circuit recovery, including with mean replacements. Duplicate circuit
entries count once; `report.circuit_size` is the number of unique retained edges.
`report.edge_class_counts` remains the full-graph count per class, not the number
retained in `C`.

The difference `report.recovered - report.edge_class_recovered[t]` measures
conditional dependence on that class's retained edges. Interactions and inhibitory
edges can increase recovery after removal; contributions are not additive or
necessarily positive. This is not a measurement of attribution ranking error.
Q/K inputs feed attention scores, but that alone does not establish which class
is ranked most accurately. LayerNorm, MLPs, and downstream attention can introduce
nonlinearity for the other classes too. The breakout costs one extra forward per
class present in the candidate; absent classes reuse aggregate recovery.

API: {func}`~transformer_lens.tools.analysis.attribution_patching.faithfulness`.

### Direct Path Patching: patch one edge exactly

`get_act_patch_direct_path` (in `transformer_lens.patching`, also importable from
`transformer_lens.tools.analysis`) fixes a source query/output head and sweeps later Q
heads or K/V groups, patching only the selected input branch. The source
head's residual write on each run is its cached `hook_z` times `W_O`. For each
destination, the corrupted prompt is rerun with the clean-minus-corrupted write added,
at every position, to the destination head's own copy of the residual, before its
LayerNorm. The norm and projection are recomputed on that copy and every other reader
still sees the corrupted output, so on pre-norm blocks the score is the exact edge
intervention, with LayerNorm folded or not. On MHA it equals ablating the same per-position
edges with the edge-ablation hook behind `faithfulness()`; that tool's separate GQA
restriction is unchanged. A source block that
normalizes, scales or joins its attention output before the residual add (post-norm,
scaled-residual or parallel-mixer blocks) is refused, since the head's write there is
not `hook_z @ W_O`.

That per-head copy is the `hook_q_input`/`hook_k_input`/`hook_v_input` fork, which
fires only while `use_split_qkv_input` is on. The function turns the flag on for the
sweep (and `use_attn_in` off, since the two are exclusive) and restores both on
return or error, so you set neither. While it is on, each attention layer normalizes a
separate residual copy per Q head and per KV group; budget activation memory for
long prompts or many heads accordingly.

Interpret a score relative to the unpatched corrupt metric. Entries at or before the
source layer are zero placeholders, not measured interventions. State the source head,
destination input, prompt pair, and normalization convention when reporting a path.

For Q, the result is `[n_layers, n_heads]`. For K/V it is
`[n_layers, n_kv_heads]`: with four query heads and two KV heads, destination group 0
feeds query heads 0 and 1, and group 1 feeds heads 2 and 3. Patching one group is one
shared-branch intervention, not two independent query-head effects. On MHA each group
has one head. All-sources results keep query/output-head source indices and have shape
`[n_layers, n_heads, n_layers, n_kv_heads]` for K/V. Only the destination layers must
agree on KV count; earlier source layers do not set that axis.

For example, this raw-weight SmolLM2 sweep uses the default pre-norm fork. It needs
the checkpoint/tokenizer downloaded (source `.env` first). The revision is pinned so
the clean/corrupt baselines can be reproduced.

```python
import torch
from transformer_lens import TransformerBridge
from transformer_lens.patching import get_act_patch_direct_path

model = TransformerBridge.boot_transformers(
    "HuggingFaceTB/SmolLM2-135M",
    revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    device="cpu",
).eval()
clean = model.to_tokens("The capital of France is")
corrupt = model.to_tokens("The capital of Germany is")
assert clean.shape == corrupt.shape
paris = model.to_single_token(" Paris")
berlin = model.to_single_token(" Berlin")

def metric(logits):
    return (logits[:, -1, paris] - logits[:, -1, berlin]).mean()

with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean)
    corrupt_logits, corrupt_cache = model.run_with_cache(corrupt)
    scores = get_act_patch_direct_path(
        model, corrupt, clean_cache, corrupt_cache, metric,
        src_layer=5, src_head=0, component="k", verbose=False,
    )
print("clean/corrupt:", metric(clean_logits).item(), metric(corrupt_logits).item())
print(scores.shape)  # [30, 3]: destination layers and KV groups
print(scores[12])    # paths from L5H0 into layer 12's K groups 0, 1 and 2
```

Each SmolLM2 KV group feeds three of its nine query heads. Compare the scores with
the corrupt baseline and repeat over prompt pairs to evaluate a hypothesis about
that path's task role.

API: {func}`~transformer_lens.patching.get_act_patch_direct_path`.

### Logit Lens: read the stream through the real norm

`logit_lens(model, prompt)` (or `cache.logit_lens()`) applies the model's own
`ln_final` with recomputed statistics, then `W_U`, `b_U` and the adapter's output
transform (Gemma-style softcap, Cohere logit scale) to every entry of the
accumulated residual stream. Its final entry equals the model's logits on a raw
bridge; in compatibility mode `center_unembed` shifts every logit by a per-position
constant, so compare `return_type="log_probs"` or ranks across processing modes, not
raw logits. `apply_ln=False` reads the unnormalized stream and does not reproduce the
model.

`targets=` reports each token's logit and its full-vocabulary rank at every layer
(`result.rank_trajectory(" Paris")`); a `[batch, pos]` tensor of next-token ids
instead gives the rank of the *actual* next token at every position and layer,
without materializing the full vocabulary; `top_k=` keeps the leading tokens with
their ids; `vocab=` keeps a subset. Ranks, entropy and log-probabilities are always
computed over the full vocabulary, whatever was selected. `logit_readout` is the
same kernel for any `d_model` vector: a head output, a steering direction, a
sparse-coder decoder column. The
[SVD Circuits demo](../generated/demos/SVD_Circuits_Demo.ipynb) reads attention-head
singular directions through it, and the
[Backward Lens demo](../generated/demos/Backward_Lens_Demo.ipynb) reads MLP
weight-gradient factors.

Intermediate-layer readouts are observations, not interventions. A token ranked
first at layer 6 is what the unembedding would say *if the model stopped there*; it
does not establish that later layers are inert, and the recomputed norm can magnify a
small early residual. Pair a lens trajectory with patching or DLA before making a
causal claim. For a fitted per-layer map instead of the identity, see
[Jacobian Lens](#jacobian-lens-distinguish-readout-reconstruction-and-intervention).

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
