# Projection Kernel

Projection Kernel (PK) measures overlap between two linear subspaces without depending on
the choice of basis within either subspace. TransformerLens provides a model-independent
numerical API and a TransformerBridge wrapper for comparing attention-head weight spaces.

## Definition

For subspaces $S,T \subseteq \mathbb{R}^d$ with orthonormal bases
$U \in \mathbb{R}^{d \times r}$ and $V \in \mathbb{R}^{d \times s}$,

$$
\operatorname{PK}(S,T) = \lVert U^\top V \rVert_F^2
                         = \sum_i \cos^2(\theta_i),
$$

where $\theta_i$ are the principal angles. Raw PK lies in $[0,\min(r,s)]$.
TransformerLens also returns

$$
\frac{\operatorname{PK}(S,T)}{\sqrt{rs}},
$$

the cosine between the two projection matrices. For equal rank $m$, this is
PK divided by $m$.

PK measures shared geometric support. It does not measure weight magnitude, prove that one
head composes with another, identify head function, or establish a causal pathway.

## Model-independent API

Extract rank explicitly before scoring:

```python
import torch

from transformer_lens.tools.analysis import orthonormal_subspace, projection_kernel

matrix_a = torch.randn(32, 4)
matrix_b = torch.randn(32, 6)
basis_a = orthonormal_subspace(matrix_a)
basis_b = orthonormal_subspace(matrix_b)
result = projection_kernel(basis_a, basis_b)

print(result.score)
print(result.normalized)
print(result.cosines)
print(result.angles)
```

`orthonormal_subspace` uses a reduced SVD. Its default relative rank tolerance is
`max(max(matrix.shape) * compute_eps, storage_eps)`. This accounts for both SVD roundoff
and input quantization without letting large low-precision matrices produce tolerances above
one. Supplying `rank` selects the leading singular subspace, but the requested rank cannot
exceed the measured numerical rank.

Float64 inputs remain float64. Float32 inputs remain float32. Float16 and bfloat16 inputs are
promoted to float32 before SVD, and outputs remain float32. Inputs must be finite,
two-dimensional, real floating-point tensors.

## Attention-head affinity

The TransformerBridge wrapper computes OQ, OK, or OV affinity for every selected head pair:

```python
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import attention_head_subspace_affinity

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
result = attention_head_subspace_affinity(
    model,
    source_role="O",
    target_role="Q",
    layer_order="forward",
)

print(result.scores.shape)
print(result.normalized.shape)
print(result.top_pairs(20, normalized=True))
```

The axes are
`[source_attention_layer, source_head, target_attention_layer, target_head]`.
`source_layer_indices` and `target_layer_indices` map tensor positions to original block
numbers. `valid_mask` has the same shape as the score tensors. Invalid entries are zero.

With `layer_order="forward"`, only strict earlier-to-later pairs are valid. Use
`layer_order="all"` to include every source-target layer pair.

## Weight orientation

TransformerLens exposes each basis-generating matrix in residual-stream coordinates:

| Role | Matrix used as a basis | Per-head shape |
|---|---|---|
| Q | `W_Q` | `[d_model, d_head]` |
| K | `W_K` | `[d_model, d_head]` |
| V | `W_V` | `[d_model, d_head]` |
| O | `W_O.T` | `[d_model, d_head]` |

The O transpose is required: `W_O` itself has shape `[d_head, d_model]`.

## MHA, GQA, and hybrid models

- Multi-head attention produces query-head axes for O and Q and K/V role axes of the same
  size.
- Grouped-query attention preserves native K/V heads. OK and OV are therefore rectangular:
  query heads by KV heads. K/V weights are not repeated to query-head count.
- Hybrid models include only blocks exposing bridged attention and retain original block
  numbers in their layer metadata.
- Architectures without readable standard Q/K/V/O projections, such as MLA or opaque
  native-forward attention, fail with a role- and layer-specific error.

By default, every head must be full column rank. An explicit `rank` applies the same
truncation to both roles and must not exceed any participating head's measured rank.
`source_ranks` and `target_ranks` report each head's measured numerical rank before
truncation. Scalar `source_rank` and `target_rank` are the retained basis widths used for
their respective roles.
When `rtol` is omitted, the wrapper uses the least-precise participating storage dtype to
derive one shared tolerance for both roles.

## Memory and scaling

The Bridge wrapper materializes dense orthonormal basis stacks before pairwise scoring.
Approximate basis memory per role is

`layers * heads * d_model * rank * element_size`.

For 32 layers, 32 heads, `d_model=4096`, rank 128, and float32, this is about **2.15 GB
per role** before allocating the other role, score tensors, masks, temporary pairwise
products, or ranking objects. Use an explicit lower rank or CPU execution when appropriate,
and start with smaller models.

`HeadAffinityResult.top_pairs()` currently creates and sorts one Python
`HeadAffinityPair` for every valid pair. On large head grids this can add substantial
memory and runtime even when only a small `k` is requested: pair enumeration scales as
`O((layers * heads)^2)`. Streaming or tiled basis extraction and tensor-only top-k ranking
are possible future optimizations; they are not implemented by the current API.

## Random-subspace reference

For independent Haar-distributed rank-$m$ planes in $\mathbb{R}^d$,
`random_projection_kernel_moments(d, m)` returns

$$
\mathbb{E}[\operatorname{PK}] = \frac{m^2}{d}, \qquad
\operatorname{Var}(\operatorname{PK}) =
\frac{2m^2(d-m)^2}{d^2(d-1)(d+2)}.
$$

These moments are descriptive. Trained heads are dependent and anisotropic, so the helper
does not return a p-value or claim a calibrated significance test.

## Relationship to Composition Score

PK discards singular-value magnitude and asks whether two read/write spaces overlap.
Composition Score retains the scale of the full linear maps and asks how strongly they
compose under its assumptions. They are complementary metrics and can rank pairs
differently. High values from either metric should be treated as candidate relationships
for activation-level or causal follow-up.

The method follows Hiroaki Yamagiwa, Yusuke Takase, and Hidetoshi Shimodaira,
“Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel,”
[arXiv:2601.10266](https://arxiv.org/abs/2601.10266).