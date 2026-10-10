# Contextual decomposition: normalization and attention allocation

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
the implemented allocations, with the normalization and attention extensions
specified below. It does not claim complete paper/reference compatibility or
permit undocumented mixtures of rules. The identity is private; its unchanged
affine and GELU behavior is retained.

## Full-input LayerNorm

Center beta and gamma over the feature axis separately, but divide both by
$\sqrt{\mathrm{Var}(\beta+\gamma)+\epsilon}$. Gain is applied to both centered
constituents. Normalization bias is allocated by the **uncentered input**
magnitudes, with all bias assigned to gamma at jointly zero coordinates. This
is an explicitly declared normalization allocation, not the affine rule applied
to independently normalized inputs.

A supplied native denominator must be finite, positive, shape `[..., 1]`, and
consistent with the combined input and epsilon within a relative tolerance of
32 machine epsilons. Larger discrepancies are rejected rather than silently
using stale cache statistics. Native model parity is separate from this kernel
contract; no native forward substitution is performed by these operators.

## Nonlinear masked multi-head attention

Q/K/V are affine-decomposed and reshaped to `[batch, heads, sequence, head_width]`.
For each head, allocate:

$$
S_\beta = Q_\beta K_\beta^T / \sqrt{d_h}, \qquad
S_\gamma = (Q_\beta+Q_\gamma)(K_\beta+K_\gamma)^T / \sqrt{d_h} - S_\beta.
$$

On the same structural mask $M$:

$$
P_\beta = \mathrm{softmax}_M(S_\beta), \qquad
P_\gamma = \mathrm{softmax}_M(S_\beta+S_\gamma)-P_\beta.
$$

Value mixing assigns:

$$
Z_\beta = P_\beta V_\beta, \qquad
Z_\gamma = (P_\beta+P_\gamma)(V_\beta+V_\gamma)-Z_\beta.
$$

This follows the printed nonlinear attention allocation in CD-T, rather than
replacing both probability constituents with a fixed native pattern. Cross terms
are allocated to gamma. Probability beta need not be zero for zero beta scores;
gamma probabilities can be negative and need not sum to one.

Causal and key-padding masks use boolean allowed-key entries. Masking is applied
inside softmax; infinity is never subtracted between constituents. Fully masked
query rows are rejected. A key-padding mask does not suppress padded queries;
callers must exclude unwanted query outputs explicitly. Left-padding plus causal
attention can produce fully masked rows and is therefore not implicitly supported.

Project each head's beta/gamma context through its own output-weight slice.
These head writes have shape `[batch, sequence, heads, width]` and **exclude**
the output projection's shared bias. Sum writes first, then allocate that bias by
the summed projected magnitudes. Applying the bias once per head would duplicate
it. Returned diagnostics retain projected Q/K/V, scores, probabilities, context,
head writes, and the final attention output.

Only ordinary full-sequence multi-head attention with the specified square-root
scaling is covered. GQA, cached decoding, dropout, positional score biases, and
alternate score scaling are not implied. Residual addition requires identical
shapes/devices/dtypes and does not broadcast; slicing returns independent tensor
storage and rejects empty/scalar constituent outputs.

## What the tests establish

Tests compare both constituents against independently evaluated scalar equations,
including nonzero bias, zero denominators, mixed signs, cancellation, and saturated
activations. Separate checks compare the reconstructed total with the ordinary
affine or GELU computation. A deliberately incorrect beta paired with a compensating
gamma demonstrates why reconstruction alone cannot verify attribution.

These checks establish the stated allocation rules. They do not establish causal
faithfulness of a source ranking or a circuit, nor applicability to an entire
TransformerBridge model.
