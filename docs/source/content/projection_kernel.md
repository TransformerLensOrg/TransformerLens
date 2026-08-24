# Projection Kernel

Projection Kernel (PK) measures overlap between two linear subspaces without depending on
the choice of basis within either subspace. TransformerLens provides a model-independent
numerical API for comparing linear subspaces.

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
`max(matrix.shape) * eps` in the computation dtype. Supplying `rank` selects the leading
singular subspace, but the requested rank cannot exceed the measured numerical rank.

Float64 inputs remain float64. Float32 inputs remain float32. Float16 and bfloat16 inputs are
promoted to float32 before SVD, and outputs remain float32. Inputs must be finite,
two-dimensional, real floating-point tensors.

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