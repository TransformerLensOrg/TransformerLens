# Sparse Probing

Leakage-safe $k$-sparse linear probing over activation tensors.

This guide covers the **model-free** core in `transformer_lens.tools.analysis.sparse_probing`.
Activation extraction, position aggregation, plotting, and multiclass support are follow-ups.

## Why this exists

A central question in mechanistic interpretability is how strongly a labelled
feature is concentrated in individual activation coordinates. Gurnee et al.
(2023) study this with $k$-sparse probes: select at most $k$ neurons, retrain
a classifier on those neurons, and evaluate held-out performance as $k$ varies.

TransformerLens already exposes activations through `run_with_cache`, but had
no maintained probing primitive. Rebuilding splitting, preprocessing,
selection, fitting, and controls per experiment is where leakage is most easily
introduced.

> The probe reports **decodability**, not causal use. A high-accuracy probe
> does not establish that the model uses the decoded feature or that a
> selected neuron is monosemantic. Interpret $k$-curves with care.

Reference: Gurnee et al., *Finding Neurons in a Haystack*, TMLR 2023.
Reference code: `wesg52/sparse-probing-paper` at `a610e102`.

## Installation

No extra dependencies. The probe uses only PyTorch (already required by
TransformerLens) and a deterministic CPU float64 LBFGS solver. It does not
depend on scikit-learn, Gurobi, plotting, or datasets.

## Quick start

```python
from transformer_lens import HookedTransformer
from transformer_lens.tools.analysis import fit_sparse_probe, sweep_sparse_probe
import torch

# 1. Collect activations however you prefer (run_with_cache, Bridge, etc.)
# X has shape [n_examples, d_features], y has shape [n_examples]
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
# ... build prompts, run model.run_with_cache, aggregate across positions ...
# X = ...  # your activation matrix
# y = ...  # Boolean or integer labels (exactly two values)

# 2. Fit a single k-sparse probe (leakage-safe)
probe = fit_sparse_probe(
    X, y,
    k=4,
    test_fraction=0.3,
    positive_label=1,
    preprocess="none",      # or "standardize"
    l2_strength=1e-2,
    seed=0,
)
print(probe.metrics.f1, probe.selected_indices)

# 3. Sweep k with controls on one fixed split
sweep = sweep_sparse_probe(
    X, y,
    ks=[1, 2, 4, 8, 16],
    test_fraction=0.3,
    positive_label=1,
    preprocess="none",
    l2_strength=1e-2,
    n_random_subsets=20,
    n_label_shuffles=20,
    seed=0,
)
for p in sweep.probes:
    print(f"k={p.k} F1={p.metrics.f1:.3f} sel={p.selected_indices}")
```

### Composing with run_with_cache

Callers obtain `X` through any appropriate `run_with_cache` workflow and
choose their own position-aggregation policy. Example:

```python
_, cache = model.run_with_cache(prompts)
# e.g. residual stream at layer 8, last position
acts = cache["blocks.8.hook_resid_post"][:, -1, :]  # [n, d]
X = acts
```

The probe is agnostic to model, layer, and position; it operates on the
supplied matrix.

## Algorithm

For $X \in \mathbb{R}^{n \times d}$ and binary labels $y$:

1. Validate finite floating $X$, binary Boolean/integer $y$, $1 \le k \le d$,
   $0 < \text{test\_fraction} < 1$, positive $l2\_strength$, and at least two
   examples per class.
2. **Split first.** One deterministic stratified split before any learned
   statistic. For class $c$, $n_{\text{test},c} = \text{clamp}(\lceil
   \text{test\_fraction}\cdot n_c\rceil, 1, n_c-1)$. Realized counts are
   returned.
3. On $X_{\text{train}}$ only, compute the raw mean-difference score
   $\text{mean}(X_{\text{train}}[y=\text{positive}]) -
   \text{mean}(X_{\text{train}}[y=\text{negative}])$; rank by absolute value
   with deterministic tie-breaking (lower index wins).
4. Select the top $k$ coordinates. `preprocess="none"` matches the reference
   default. `preprocess="standardize"` uses selected-column train mean and
   population standard deviation, maps zero std to scale one, and applies the
   same transform to test data. Because L2 is scale-sensitive, this mode
   changes the objective and is recorded.
5. Fit logits $z = Xw + b$ by minimizing mean class-balanced binary
   cross-entropy plus $l2\_strength \cdot \|w\|^2/2$; weights $n/(2n_c)$,
   bias $b$ unregularised. Optimise selected matrices on CPU float64 with
   Torch LBFGS and strong-Wolfe line search. Require finite output and
   `grad_threshold`; otherwise raise.
6. Predict positive at $z \ge 0$. Report held-out confusion counts, accuracy,
   precision, recall, and F1 (zero when denominator is zero). F1 is primary.
7. Sweeps reuse the exact split across every strictly increasing unique $k$.
   Random-coordinate controls sample supports without replacement; label-shuffle
   controls permute training labels, repeat selection/fitting, and evaluate
   against untouched test labels. Controls use a local `torch.Generator`.

Every $k$-sweep uses one fixed preprocessing mode and L2 strength. Curve
shape is conditional on those choices; no hyperparameter search is performed.

Solver details: score reductions run in at least float32 on the input device;
only the selected $[n, k]$ matrices are transferred to CPU float64 for LBFGS.
The implementation never mutates global RNG state.

## API

### fit_sparse_probe

```python
fit_sparse_probe(
    X, y, k,
    test_fraction=0.3,
    positive_label=1,
    preprocess="none",
    l2_strength=1e-2,
    seed=0,
    grad_threshold=1e-5,
    max_iter=100,
) -> SparseProbeResult
```

Returns class labels/counts, selected indices/scores, coefficients/intercept,
preprocessing metadata, split indices, held-out metrics, and solver metadata
(objective, grad norm, iterations, convergence).

### sweep_sparse_probe

```python
sweep_sparse_probe(
    X, y, ks,
    test_fraction=0.3,
    positive_label=1,
    preprocess="none",
    l2_strength=1e-2,
    n_random_subsets=20,
    n_label_shuffles=20,
    seed=0,
    grad_threshold=1e-5,
    max_iter=100,
) -> SparseSweepResult
```

`ks` must be strictly increasing unique values. Control repeat counts are
non-negative; zero disables that control.

## Interpreting results

- **F1 is primary.** Accuracy is reported but unsafe under imbalance.
- **Controls are raw distributions.** No automatic significance label is
  assigned. Compare the planted $k$-curve against random-coordinate and
  label-shuffle baselines.
- **Document leakage.** The function cannot detect leakage already present in
  caller-provided $X$ (e.g. dataset-level normalisation). Compute $X$ before
  calling and pass raw tensors.

## Known divergences from the reference

- Stratified split (reference: random split).
- Explicit centred train-only standardisation (reference: optional scaling by
  clamped train std $\times 10$ without centring, disabled by default).
- Defined Torch loss/L2 convention with balanced weights and CPU float64 LBFGS
  plus convergence metadata (reference: sklearn SAGA, `max_iter=200`).

## Limitations

Activation extraction, plotting, notebooks, multiclass labels, SAE-latent
probing, MIP selection, and causal validation are deferred to follow-ups.
