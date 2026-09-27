# SVD Circuits

SVD Circuits decomposes a single attention head's QK and OV weight maps into orthogonal
singular directions, so a head can be examined as a sum of low-rank subfunctions rather
than as one indivisible unit. Every claimed subfunction is then causally gated by
patching activations along its direction.

A weight-space decomposition is not a causal claim. A direction can project onto
plausible tokens while carrying no role in the model's behaviour, which is why the
causal gate is part of the tool rather than an optional extra.

## Definition

For a head at layer $\ell$ with head index $h$, the two maps are

$$
\mathrm{QK} = W_Q W_K^\top, \qquad \mathrm{OV} = W_V W_O,
$$

each of shape $d_{\text{model}} \times d_{\text{model}}$ but of rank at most
$d_{\text{head}}$. Their SVDs are

$$
W_V W_O = U \Sigma V^\top, \qquad \operatorname{rank} \le d_{\text{head}}.
$$

The convention matters, because both factors are $d_{\text{model}}$-wide and a swap
raises no shape error:

- For OV, the columns of $V$ are the residual-stream **output/write** directions, the
  ones projected through $W_U$ for a vocab readout. The columns of $U$ span the
  value-computation **input** space.
- For QK, the columns of $V$ are the **source/key-read** directions and the columns of
  $U$ the **destination/query-read** directions. QK produces no write direction, so it
  has no vocab readout.

`FactoredMatrix` computes both SVDs without materialising the $d_{\text{model}}^2$
product.

## The degeneracy guard

Singular directions are unique only when the singular values are distinct. Equal or
near-equal consecutive singular values leave the corresponding subspace defined only up
to an arbitrary rotation, so any statement of the form "direction 3 is the surname
subfunction" is a statement about a basis choice rather than about the model.

`decompose_head` returns a rank report marking each direction, and the consumers refuse
per-direction attribution inside a degenerate block, reporting the block as a subspace
instead. Numerically null directions are refused too, since they are arbitrary vectors
from the map's null space.

## The causal gate

`patch_along_directions` reconstructs the head's output onto a chosen singular subspace
and reports the resulting change in a caller-supplied metric:

- `delta_metric`: patched metric minus original metric.
- `baseline_delta_metric`: the mean magnitude over random same-width subspaces drawn
  inside the head's own OV span. Drawing the control in-span rather than from the full
  residual stream makes it the effect of an arbitrary subspace of *this head's* output,
  which is the comparison the gate needs.
- `gated`: whether the retained subspace passed the causal test.

The success condition depends on the mode the caller expressed, because `keep=S` and
`ablate=complement(S)` resolve to the same retained set:

| Mode | Meaning | `gated` is |
|---|---|---|
| `keep=S` | retain only $S$ | `abs(delta_metric) < threshold` |
| `ablate=S` | zero $S$, retain the rest | `abs(delta_metric) > threshold` |

With `keep`, a direction that alone reconstructs the head's behaviour moves the metric
*less* than an arbitrary same-width subspace. With `ablate`, a load-bearing direction
moves it *more*. A single "moved more than the control" test cannot answer both
questions, which is why the mode is explicit.

The threshold defaults to `baseline_delta_metric`. Because the per-draw control
magnitudes are heavy-tailed, the averaged threshold still varies with the seed, so a
direction whose delta sits near it can gate either way. Pass an explicit `rng` for a
reproducible verdict, and raise `n_baseline` when the verdict is close.

## Compatibility mode

The vocab readout projects through the final LayerNorm folded into $W_U$, which requires
`enable_compatibility_mode()` on an adapter that supports folding. A decomposition
records the folded-LayerNorm state it was taken under, and the readout and patch
consumers refuse a decomposition whose state no longer matches the model. See
[Compatibility Mode](compatibility_mode.md).

## Worked example

```python
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.svd_circuits import (
    decompose_head,
    patch_along_directions,
    vocab_readout,
)

model = TransformerBridge.boot_transformers("gpt2", dtype=torch.float32, device="cpu")
model.enable_compatibility_mode()
model.eval()

prompt = "When Mary and John went to the store, John gave a drink to"
ov = decompose_head(model, layer=9, head=9, which=("OV",)).OV

readout = vocab_readout(model, ov, k=10)

mary, john = model.to_single_token(" Mary"), model.to_single_token(" John")
metric = lambda logits: float(logits[0, -1, mary] - logits[0, -1, john])

result = patch_along_directions(
    model, ov, prompt, metric, keep=[0], rng=torch.Generator().manual_seed(0)
)
print(result.delta_metric, result.baseline_delta_metric, result.gated)
```

## What this does not establish

- A passing gate validates one subspace on one prompt, not a named subfunction. The
  vocab readout suggests what a direction moves; the gate tests whether it matters.
- Results are single-head. Nothing here assembles a multi-head circuit.
- The paper's subfunction taxonomy is scale- and model-dependent, so a small model
  reproduces the mechanism rather than the paper's exact inventory.
- Verdicts near the baseline threshold are seed-sensitive and should be read as
  borderline.

## Links

- [SVD Circuits demo](../generated/demos/SVD_Circuits_Demo.html)
- Areeb Ahmad, Abhinav Joshi, Ashutosh Modi, "Beyond Components: Singular Vector-Based
  Interpretability of Transformer Circuits", [arXiv 2511.20273](https://arxiv.org/abs/2511.20273)
