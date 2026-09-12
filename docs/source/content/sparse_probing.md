# Sparse Probing

Sparse probing measures how strongly binary-label information is concentrated in a small
set of activation coordinates. TransformerLens provides a model-independent fit and sweep API;
callers decide how to collect and aggregate activations.

Probe performance establishes held-out decodability under the chosen split, preprocessing,
regularization, and sparsity. It does not establish causal model use, neuron monosemanticity,
or superposition.

## Fit one probe

```python
import torch

from transformer_lens.tools.analysis import fit_sparse_probe

features = torch.randn(200, 768)
labels = torch.arange(200) % 2
result = fit_sparse_probe(
    features,
    labels,
    k=8,
    positive_label=1,
    test_fraction=0.3,
    preprocess="none",
    class_weight="balanced",
    l2_strength=1e-2,
    seed=0,
)

print(result.selected_features)
print(result.metrics.f1)
```

`features` must have shape `[example, feature]` and dtype float16, bfloat16, float32, or float64.
Labels must be a one-dimensional Boolean or integer tensor containing exactly two values, one equal
to `positive_label`. Each class needs at least two examples.

## Selection and split contract

The function creates a deterministic stratified split before computing any learned statistic.
For each class $c$, its realized test count is

$$
n_{\mathrm{test},c} =
\operatorname{clamp}(\lceil f n_c \rceil, 1, n_c - 1),
$$

where $f$ is `test_fraction`. Realized train and test counts are returned because small classes
can differ materially from the requested aggregate fraction.

On the training split only, feature $j$ receives the signed score

$$
s_j = \mathbb{E}[X_j \mid y=\mathrm{positive}]
      - \mathbb{E}[X_j \mid y=\mathrm{negative}].
$$

The selected support contains the $k$ largest $|s_j|$. Equal scores are resolved by increasing
feature index. `preprocess="none"` fits the selected raw coordinates. With
`preprocess="standardize"`, selected columns are centered and scaled using training statistics;
zero-variance columns receive scale one. The same transform is then applied to held-out values.

Because L2 regularization is scale-sensitive, preprocessing can change the fitted probe and
the resulting k-curve. A sweep therefore fixes preprocessing and L2 strength across every k.

## Logistic objective

For selected training features $X$, labels $y \in \{0,1\}$, coefficients $w$, and intercept $b$,
the optimizer minimizes

$$
\frac{1}{n}\sum_i \alpha_{y_i}
\operatorname{BCEWithLogits}(X_i w + b, y_i)
+ \frac{\lambda}{2}\lVert w\rVert_2^2,
\qquad
\alpha_c = \frac{n}{2n_c}.
$$

The displayed weights apply to the default `class_weight="balanced"`; pass `None` to use
$\alpha_c=1$. The intercept is not regularized. Positive predictions have nonnegative logits.
Accuracy, precision, recall, F1, and all four confusion counts are returned; precision or F1 is
zero when its denominator is zero. F1 is the primary sparse-probing metric.

Feature-score reductions use float64 for float64 inputs and float32 otherwise. Selected matrices
move to CPU float64 for deterministic LBFGS fitting. All result tensors are detached CPU tensors.
The fit raises when output is non-finite or the final objective-gradient infinity norm exceeds
`gradient_tolerance`, which must lie in `(0, 1]`.
Results retain the requested `k`, `max_iter`, and `gradient_tolerance` alongside the realized
objective, gradient norm, iteration count, and convergence flag.

## Sweep and controls

```python
from transformer_lens.tools.analysis import sweep_sparse_probe

sweep = sweep_sparse_probe(
    features,
    labels,
    ks=[1, 2, 4, 8, 16],
    n_random_subsets=20,
    n_label_shuffles=20,
    seed=0,
)

for k, probe, random_control in zip(
    sweep.ks,
    sweep.results,
    sweep.random_coordinate_controls,
    strict=True,
):
    print(k, probe.metrics.f1, random_control.f1.median())
```

Every k uses the same split, preprocessing mode, and L2 strength. `ks` must be strictly
increasing and unique.

Random-coordinate controls sample k distinct coordinates and fit the same classifier.
Label-shuffle controls permute training labels, repeat selection and fitting, and evaluate against
the untouched held-out labels. The API returns raw control supports and metric distributions; it
does not convert them into p-values or representation labels. A repeat count of zero disables that
control.

Controls can be expensive: the sweep performs one main fit plus both requested control counts for
every k. Start with small grids and repeat counts.

## Composing with cached activations

Use `run_with_cache` to construct the feature matrix separately so token, position, batching, and
aggregation choices remain explicit:

```python
tokens = model.to_tokens(prompts)
_, cache = model.run_with_cache(tokens, names_filter=[hook_name])
features = cache[hook_name][:, -1, :]
result = fit_sparse_probe(features, labels, k=8)
```

The example selects the final sequence position, which is not appropriate for every dataset.
Choose the hook and position policy before interpreting selected coordinates. The API cannot detect
leakage already introduced into caller-provided `features`.

## Reference

The raw mean-difference selector and sparse-probing framing follow Wes Gurnee et al.,
“Finding Neurons in a Haystack: Case Studies with Sparse Probing,”
[TMLR 2023](https://openreview.net/forum?id=JYs1R9IMJr),
with [reference code](https://github.com/wesg52/sparse-probing-paper).

TransformerLens intentionally adds stratification, stable tie-breaking, explicit objective and
convergence diagnostics, and deterministic controls. Its optional centered standardization and
Torch LBFGS solver are not exact reproductions of the reference implementation.