# Representation Geometry

Representation geometry compares concept directions under a metric derived from
centered unembedding covariance. TransformerLens provides a tensor-level API and a
non-mutating adapter for raw `TransformerBridge` readouts.

The metric is motivated by Park, Choe and Veitch,
[arXiv:2311.03658v2](https://arxiv.org/abs/2311.03658v2), Section 3.2, Equation (3.3).
It is one choice of causal inner product under the paper's assumptions, not a
universal guarantee of separability. Whitening identities establish arithmetic
correctness, not causal model use. Counterfactual datasets and independent
interventions require empirical validation.

The [download-free walkthrough](../generated/demos/RepresentationGeometry_Demo.ipynb)
uses a known synthetic covariance and a tiny randomly initialized TransformerBridge.
Its constructed orthogonality and simplex examples are not paper reproduction or
evidence about trained language models.

API: {class}`~transformer_lens.tools.analysis.representation_geometry.RepresentationGeometry`.

## Start with a readout tensor

```python
import torch
from transformer_lens.tools.analysis import RepresentationGeometry

readout = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)
tensor_geometry = RepresentationGeometry(readout)
concept = tensor_geometry.concept_direction([(0, 1), (1, 2)], label="step")

assert concept.raw_measurement.tolist() == [1.0]
assert concept.n_pairs == 2
assert concept.raw_dispersion.item() == 0.0
```

The input layout is `[d_model, d_vocab]`, as in `bridge.W_U`. Tokens have uniform
weight in the declared covariance population. `token_ids=` selects a non-empty
set of unique vocabulary IDs; it does not restrict which tokens may appear in
contrast pairs. No special tokens are excluded implicitly.

The object holds a detached snapshot. Source-weight edits do not change its fit
or contrast directions. Matrix properties return copies; result dataclasses have
frozen metadata but mutable tensor contents, which should be treated as read-only.

## Covariance, dual spaces and ridge

Let $g_v$ be unembedding column $v$ and $V$ the selected token count:

$$
\bar g = \frac{1}{V}\sum_v g_v,
\qquad
\Sigma = \frac{1}{V}\sum_v (g_v-\bar g)(g_v-\bar g)^\top.
$$

This is centered **population** covariance, not sample covariance divided by
$V-1$. For an exact fit, $M=\Sigma$. An explicit positive `ridge=epsilon` instead
sets $M=\Sigma+\epsilon I$.

For column-vector measurement $g$ and intervention/context vector $h$:

$$
W=M^{-1/2},\qquad S=M^{1/2},\qquad
 g'=Wg,\qquad h'=Sh.
$$

The dual transforms preserve the pairing:

$$
(h')^\top g' = h^\top g.
$$

Do not apply measurement whitening to intervention vectors. Runtime tensors use
trailing feature dimensions and row multiplication:

- `whiten_measurement(g)` computes `g @ W.T`.
- `unwhiten_measurement(g_prime)` computes `g_prime @ S.T`.
- `whiten_intervention(h)` computes `h @ S.T`.
- `unwhiten_intervention(h_prime)` computes `h_prime @ W.T`.
- `measurement_inner_product(a, b)` computes $a^\top M^{-1}b$.
- `intervention_inner_product(a, b)` computes $a^\top Mb$.
- `measurement_cosine` and `intervention_cosine` normalize the corresponding metrics.

Binary operations broadcast leading batch dimensions and compare corresponding
rows, not an implicit all-pairs grid. Cosines reject zero directions; zero-vector
inner products are valid. Operations preserve vector gradients, but the fit does
not backpropagate into the readout weights. Bare tensors do not identify their
semantic basis, so callers must choose the appropriate space and must not
whiten twice.

Exact whitening satisfies $W\Sigma W^\top=I$. With ridge, the identity is
$W(\Sigma+\epsilon I)W^\top=I$; the original whitened covariance is not generally
identity. Ridge changes the metric. Exact geometry is invariant under consistent
invertible dual basis changes; an isotropic ridge is invariant under orthogonal
changes, not arbitrary changes unless the regularizer transforms too.

A singular or near-singular exact fit raises. There is no silent eigenvalue
clamping or pseudo-inverse. An explicit ridge must be large enough for a stable
positive-definite fit under the declared rank threshold.

## TransformerBridge basis contract

For an already constructed raw Bridge:

```python
geometry = RepresentationGeometry.from_bridge(bridge, compute_dtype=torch.float64)
concept = geometry.concept_direction(
    [(" king", " queen"), (" man", " woman")], label="gender"
)
print(geometry.basis)
print(concept.pairs)
```

Use the model's exact tokenizer spelling, including leading spaces. The local
notebook demonstrates construction without downloading weights or a tokenizer.
A checkpoint-loaded Bridge must follow the usual [environment and loading
instructions](getting_started.md).

The adapter uses **the input to the actual linear unembedding after final LN or
RMS normalization**, including the norm's learned gain and bias. It does not fold
norm parameters into `W_U`, divide a weight-only direction by an activation's
cached scale, enable compatibility mode, move the model, change training flags,
or execute a forward pass. The unembedding bias is not part of the covariance
metric; logit differences additionally contain the corresponding bias difference.

The supported contract requires:

- A causal decoder-only TransformerBridge with readable final LN/RMS and a plain
  `torch.nn.Linear` unembedding whose weights match the public `W_U` accessor.
- Raw, unprocessed weights, without compatibility mode or normalization folding.
- No final output projection or custom post-readout transform, including active
  soft caps. Parametrized, custom-forward and offloaded meta readouts are rejected.
- Readout dimensions matching `cfg.d_model` and `cfg.d_vocab_out`.

This is a bounded support contract, not certification of every architecture.
`GeometryBasis` records source, location, normalization type, architecture and
model name. It is not a unique model identity or permission to mix snapshots.
Pre-final-normalization residuals and intermediate-layer activations are not
in this basis merely because they have the same width.

The optional Hugging Face tokenizer is deep-copied. String endpoints must encode
to exactly one known token with `add_special_tokens=False`; empty, multi-token
and unknown-token substitutions raise. IDs need no tokenizer. Mixed string/ID
pairs are accepted; inferred labels preserve supplied strings and decode IDs.
Explicit `pair_labels=` overrides those labels. Subsequent source-tokenizer
changes do not affect the snapshot.

## Contrast direction and dispersion

Pair `(lo, hi)` contributes $\delta_i=g_{hi}-g_{lo}$. The measurement is the
**unnormalized, equally weighted mean** $\mu=N^{-1}\sum_i\delta_i$. Centering
cancels in differences and is not applied again. Reversing every pair reverses
the direction's sign without changing its dispersion.

`ConceptDirection` contains raw/whitened measurement vectors, raw/whitened
**metric-derived** intervention vectors, per-pair differences, frozen IDs/labels,
pair count, geometry diagnostics and basis provenance.

`derived_intervention(g)` computes $M^{-1}g$. The equality between its whitened
coordinates and the measurement's whitened coordinates follows by construction.
It is not an independently estimated contextual intervention or evidence that
input-embedding differences align causally. No `W_E` intervention estimator is
implied.

The reported population dispersion is

$$
D_{raw}=\frac{1}{N}\sum_i\lVert\delta_i-\mu\rVert^2,
\qquad
D_{white}=\frac{1}{N}\sum_i\lVert W(\delta_i-\mu)\rVert^2.
$$

One pair has dispersion zero, not a confidence guarantee. Near-canceling nonzero
means can have high dispersion. Empty/self/duplicate/reversed-duplicate pairs,
invalid IDs, zero contrasts, zero aggregate directions and non-finite results
raise instead of being silently filtered. The API does not infer causal validity
from pair count or low dispersion.

## Categorical diagnostics, not assumed simplices

Supply explicit concept vertices in a declared raw space:

```python
vertices = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
report = tensor_geometry.categorical_geometry(vertices, space="measurement")
print(report.affine_rank)
print(report.is_simplex())
print(report.is_regular_simplex())
```

This example uses the one-dimensional tensor geometry from the first section,
not the Bridge geometry from the separate example. The input has shape
`[n_categories, d_model]` with at least two rows. Optional labels must be unique,
non-empty strings in vertex order.

Vertices are centered at their centroid and transformed with the declared
measurement or intervention map. The report exposes centered/raw/whitened
vertices, singular values, affine rank, Gram matrix, pairwise distances, cosines,
angles in radians, and a validity mask.

- `is_simplex()` checks affine independence: rank equals `n_categories - 1`.
- `is_regular_simplex()` additionally checks relative pair-distance spread
  `(max - min) / mean` over off-diagonal distances. `rtol=` overrides this
  closeness policy, not the rank threshold.
- Duplicate, coincident and dependent vertices remain in the report. Undefined
  centroid-relative angles/cosines are NaN where `angle_valid_mask` is false;
  they are not fabricated zeros.
- Default rank tolerance is `max(vertices.shape) * compute_eps`. Norm validity
  uses that relative tolerance against the largest row norm. Default regularity
  tolerance is eight times the default rank tolerance.

These are numerical properties of supplied vertices. Arbitrary category-token
rows are not automatically the category representations constructed in
[arXiv:2406.01506](https://arxiv.org/abs/2406.01506), and no token-to-category
estimator, hierarchy or paper reproduction is implied.

## Precision and resource costs

Float16/bfloat16 readouts promote to float32; float32 stays float32 and float64
stays float64 unless `compute_dtype=` explicitly selects float32 or float64.
Operations return fit precision on the fit device. CPU and CUDA are supported;
there is no implicit device transfer. Model-scale GPU accuracy is not established
by the download-free CPU examples.

The covariance rank threshold defaults to `d_model * compute_eps` times the
largest eigenvalue. Diagnostics record population, storage/compute dtype,
regularization, rank, thresholds and condition numbers. Explicit tolerance
choices must be reported with research results.

Weight-only fitting is not cheap on large models. Dense covariance formation
costs approximately `O(V * d_model**2)` and eigendecomposition `O(d_model**3)`.
The object retains a full readout copy in its storage dtype and approximately
five `d_model**2` matrices in compute precision. Fitting also needs centered
readout, conversion and eigensolver temporaries. With `d_model=4096` and
`V=131072`, one float32 readout-sized array is **2 GiB** and one covariance-sized
array **64 MiB**; these are shape-based estimates, not measured benchmarks.
`token_ids=` reduces covariance work but does not remove the retained full
readout snapshot. Categorical reports additionally allocate `O(n_categories**2)`
matrices. Start with small models and small category sets.

## Steering, probes and research evidence

A derived intervention belongs in the declared post-normalization readout basis,
not automatically at an earlier residual hook. [Hooks](hook_system.md) and
[Jacobian Lens](jacobian_lens_fitting.md) provide complementary intervention
mechanisms; installing an edit still requires matched controls and behavioral
measurements.

For [sparse probes](sparse_probing.md), coefficients are measurement covectors.
Reconstruct their full coordinate support and undo preprocessing before any
geometry comparison, preserving intercept/prediction equivalence. Dense
whitening changes coordinate sparsity, and probing on unrelated feature spaces
requires an explicit compatible mapping. No probe/steering adapter is supplied
by the geometry object itself.

Raw cosine is a baseline, not universally invalid. Report model/tokenizer
revisions, basis, dtype, processing state, contrast construction, ridge and rank
policy. Independent model experiments are needed to establish separability,
intervention effects or categorical/hierarchical structure. The synthetic and
tiny-model checks do not validate these claims on trained models.
