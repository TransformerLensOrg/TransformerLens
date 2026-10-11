# Single-source contextual decomposition on GPT-2

`ContextualDecomposition` provides nonlinear source-to-target constituent
allocation on a raw fp32, evaluation-mode GPT-2 `TransformerBridge` with eager
attention. It supports one head/head-position or residual source, zero or supplied
references, and downstream internal or logit targets. It does not discover
circuits, fit references, score datasets, or establish causal faithfulness.

## Runnable model-free-download example

This example initializes a small HF GPT-2 locally. No Hub download, tokenizer,
or external artifact is needed:

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.tools.analysis import (
    CDBaseline, CDSource, CDTarget, ContextualDecomposition,
)

config = GPT2Config(vocab_size=16, n_positions=8, n_embd=8, n_layer=2, n_head=2)
config._attn_implementation = "eager"
with torch.random.fork_rng():
    torch.manual_seed(5)
    model = build_bridge_from_module(
        GPT2LMHeadModel(config).eval(), "GPT2LMHeadModel", hf_config=config,
        dtype=torch.float32, device="cpu",
    ).eval()

analysis = ContextualDecomposition(model)
result = analysis.decompose(
    torch.tensor([[1, 2, 3], [3, 2, 1]]),
    source=CDSource.attention_head(0, 1, positions=(0,)),
    target=CDTarget.logit_difference(
        positive_token_id=(2, 3), negative_token_id=(1, 2), position=(1, 2),
    ),
    baseline=CDBaseline.zero(),
    retain_intermediates=("blocks.1.ln1.hook_out",),
)
torch.testing.assert_close(
    result.target_beta + result.target_gamma, result.target_total,
    rtol=2e-5, atol=2e-6,
)
print(result.target_beta, result.baseline_description)
```

Use `CDSource.residual(layer, location="pre" | "mid" | "post")` for residual
anchors, `CDTarget.internal(hook_name, position=..., head=...)` for supported
internal tensors, and `CDTarget.logit(token_id=..., position=...)` for one logit.
The default policy is `magnitude-affine-ordered-gelu-v1`; unknown policies are
rejected, not silently replaced with fixed-pattern attribution.

`CDBaseline.from_tensor(reference, description=...)` snapshots a finite fp32
`[batch, sequence, width]` reference. There is no implicit dtype/device conversion.
Results own detached tensors and read-only metadata mappings. The native target
is retained independently as `target_total`, not reconstructed to hide drift.
`diagnostics` reports per-hook numerical errors; `coverage` lists checked native
locations, not proof that the allocation is causally faithful. By default no
intermediate tensors are retained in the result; request supported downstream
hook names explicitly. This controls returned retention, not peak executor memory.
The executor still captures and validates the complete native suffix.

## Constituent rules

The underlying operators are private inference kernels. Each activation is represented by two
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

## Private GPT-2 residual-suffix execution

The private executor captures a raw fp32, evaluation-mode local GPT-2 Bridge
with eager attention. It reads native HookPoints without editing tensors and
checks each reconstructed suffix intermediate and final logit tensor against
that capture. A block-input residual anchor accepts beta in residual space;
gamma is the native residual minus beta. The unchanged prefix supplies that
anchor, while token and position embedding totals are independently checked.

Q/K/V projections use the actual split native linear weights; GPT-2 Conv1D MLP
and output projections retain their row-vector layout. The final unembedding
uses its actual weight orientation and bias, including biases introduced by
Bridge conversion. No residual correction is added to hide a mismatch.

Native parity checks use relative tolerance `2e-5` and absolute tolerance `2e-6`;
results record maximum absolute error and native reference magnitude for each
checked intermediate. These bounds are tested on locally initialized GPT-2 with
nonzero gains and biases. They are not a guarantee for every checkpoint or an
excuse to ignore a failing native comparison.

Existing read-only HookPoint observers remain installed. Activation edits,
external computational-module hooks, active relevance rules, and parameter
changes during capture are rejected. Scoped observation and lazily created
Bridge/HF collector hooks are removed on success and exceptions; pre-existing
collectors are preserved. Analysis does not consume the model's RNG stream or
create/clear parameter gradients. Arbitrary observers that mutate unrelated
external state are outside this guarantee.

Compatibility/processed weights, sharding, non-fp32 parameters, cross-attention,
repeated blocks, split-input attention forks, cached decoding, alternate scaling,
and SDPA are explicitly unsupported. Inputs are token tensors with optional
aligned boolean key masks and explicit position IDs; fully masked causal query
rows are rejected. This is a private executor, not a public source-selection API.

## Private single-source selection and targets

Single-source execution accepts a head write or a residual activation at block
input (`pre`), the attention residual merge (`mid`), or block output (`post`).
Head sources always anchor at `mid`. The selected head's native context is
projected with only that head's output-weight slice; the shared output bias is
not included in the source write. At the anchor, beta is the selected write minus
a supplied residual-space reference, and gamma is the entire native residual
minus beta, retaining all other writes and shared constants.

Source positions are unique nonnegative absolute indices shared across the batch.
Only selected positions receive beta; all downstream query positions are still
traversed. Selecting padding is rejected, including all-position sources on
padded batches. Supplied references must match `[batch, sequence, width]`, fp32,
and the model device. Empty, overlapping, negative, or out-of-range selectors
are rejected; no joint-source semantics are implied.

Targets must be strictly downstream of the anchor. Internal selectors use an
explicit supported Bridge-native hook name and its actual tensor space. Q/K/V
and context targets can select a head; attention probabilities are head-major
and must currently be selected as a whole tensor to avoid ambiguous position
semantics. Logit and logit-difference targets accept a shared selector or a tuple
with one position/token ID per example. A difference between equal token IDs
returns zero beta and gamma. The final normalization remains part of the
combined-input decomposition, not a separate norm of a beta vector.

A native-equal reference makes the selected source difference zero and removes its beta.
Native reconstruction is checked before installing each source anchor and at
every downstream intermediate. Highly cancelling constituents can exceed the
stated fp32 tolerances; such cases raise rather than silently adjusting gamma or
loosening the threshold. Source-write diagnostics and target constituents own
independent output storage. The public source/target constructors validate these same selector conventions.

## What the tests establish

Tests compare both constituents against independently evaluated scalar equations,
including nonzero bias, zero denominators, mixed signs, cancellation, and saturated
activations. Separate checks compare the reconstructed total with the ordinary
affine or GELU computation. A deliberately incorrect beta paired with a compensating
gamma demonstrates why reconstruction alone cannot verify attribution.

These checks establish the stated allocation rules. They do not establish causal
faithfulness of a source ranking or a circuit, nor applicability to an entire
TransformerBridge model.
