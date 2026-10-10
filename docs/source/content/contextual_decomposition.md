# Contextual decomposition: affine and GELU allocation

The contextual-decomposition operators are private, model-free inference kernels.
They do not expose a model analysis API. Each activation is represented by two
constituents, $x = \beta + \gamma$. Inputs must be finite, nonempty strided
float32 or float64 tensors with matching shapes, devices, and dtypes. Operations
return detached tensors without mutating inputs or creating backward graphs.

## Affine bias allocation

For TL row-vector weights $W$ and bias $b$, first compute

$$
u = \beta W, \qquad v = \gamma W.
$$

Allocate bias elementwise by projected magnitude:

$$
s = \begin{cases}
|u| / (|u| + |v|), & |u| + |v| > 0, \\
0, & u = v = 0,
\end{cases}
\qquad
\beta' = u + s b, \qquad \gamma' = v + (1-s)b.
$$

The nonzero-denominator rule follows the affine allocation in equations 4-5 of
[Hsu et al., *Efficient Automated Circuit Discovery in Transformers using
Contextual Decomposition*](https://arxiv.org/abs/2407.00886).
Assigning all bias to gamma when both projections vanish is an explicit completion
of the undefined ratio, not a claim that the paper uniquely specifies that case.
It preserves zero relevant input without splitting a purely constant output.

The implementation divides both magnitudes by their maximum before computing
$s$. Both bias fractions are evaluated directly, rather than computing the
remaining fraction by subtracting a rounded value from one. This avoids overflow
in the magnitude sum, preserves very small remaining shares, and does not add an
epsilon that would lose bias mass for small projections. Floating-point rounding can still
hide small bias additions beside very large constituents; finite outputs do not
guarantee a well-conditioned reconstruction.

## Ordered GELU allocation

For activation $f$, use the ordered contextual difference

$$
\gamma' = f(\gamma), \qquad
\beta' = f(\beta+\gamma) - f(\gamma).
$$

This allocation is asymmetric, not a Shapley average or an LRP backward rule.
It preserves the complete remaining-context activation and assigns its difference
from the total activation to beta. Cancelling or negative constituents are not
silently redistributed.

Supported formulas are exact GELU, the explicit GPT-2 `gelu_new` formula

$$
f(x) = \frac{x}{2}\left[1+\tanh\left(\sqrt{2/\pi}
\left(x+0.044715x^3\right)\right)\right],
$$

and PyTorch's fused tanh approximation (`gelu_pytorch_tanh`). The two tanh paths
have the same mathematical formula but retain their distinct floating-point
operation order. Exact GELU instead uses the Gaussian CDF. Unsupported activation
names are rejected.

The fixed private policy identity is `magnitude-affine-ordered-gelu-v1`. It names
only these implemented allocations; it does not claim complete paper/reference
compatibility or permit undocumented mixtures of rules.

## What the tests establish

Tests compare both constituents against independently evaluated scalar equations,
including nonzero bias, zero denominators, mixed signs, cancellation, and saturated
activations. Separate checks compare the reconstructed total with the ordinary
affine or GELU computation. A deliberately incorrect beta paired with a compensating
gamma demonstrates why reconstruction alone cannot verify attribution.

These checks establish the stated allocation rules. They do not establish causal
faithfulness of a source ranking or a circuit, nor applicability to an entire
TransformerBridge model.
