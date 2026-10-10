# SVD Circuits

SVD Circuits decomposes a single attention head's QK and OV weight maps into orthogonal
singular directions. Vocab readouts and activation projections describe those directions;
patching measures prompt-specific changes in a caller-selected output metric.

Mathematically distinct directions need not be semantically or causally distinct
subfunctions. A weight-space decomposition or a plausible token projection does not
establish a direction's role. Interventions provide additional measurements, but the
reported gate comparison is not by itself a validation of a named mechanism.

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

`decompose_head` returns a rank report marking degenerate and numerically null
directions. Null directions are arbitrary vectors from the map's null space and are
also unsuitable for individual attribution. The consumers have different guards:

- `logit_signature` requires an isolated, non-null direction and refuses individual
  directions inside a degenerate block.
- `patch_along_directions` refuses selections that split a degenerate block. Complete
  blocks can be retained or removed as subspaces; empty or full-span retained sets
  require an explicit `threshold` because their random controls coincide with them.
- `vocab_readout` and `project_activations` return raw numerical projections, including
  columns inside degenerate blocks. They neither refuse these columns nor replace
  them with block summaries. Consult the rank report and exclude both `is_degenerate`
  and `is_null` before interpreting individual directions.

## The intervention comparison

`patch_along_directions` reconstructs the head's output onto a chosen singular subspace
and reports the resulting change in a caller-supplied metric:

- `delta_metric`: patched metric minus original metric.
- `baseline_delta_metric`: the mean magnitude over random same-width subspaces drawn
  inside the head's own OV span. Drawing the control in-span rather than from the full
  residual stream makes it the effect of an arbitrary subspace of *this head's* output,
  which is the comparison the gate needs.
- `gated`: whether the metric change passed the mode-specific threshold comparison.

The comparison depends on the mode the caller expressed, because `keep=S` and
`ablate=complement(S)` resolve to the same retained set:

| Mode | Meaning | `gated` is |
|---|---|---|
| `keep=S` | retain only $S$ | `abs(delta_metric) < threshold` |
| `ablate=S` | zero $S$, retain the rest | `abs(delta_metric) > threshold` |

With `keep`, a passing comparison means retaining the selected subspace changes the
metric less than the threshold. With `ablate`, it means removing the selected subspace
changes the metric more than the threshold. These describe effects on the chosen metric,
not reconstruction of the head's entire behaviour or identification of its function.

The threshold defaults to `baseline_delta_metric`, a sampled mean magnitude rather than
a confidence bound or p-value. It does not establish statistical separation from
arbitrary directions. The sampled mean can vary with the seed and draw count, so verdicts
near it can change. Pass an explicit `rng` for repeatability and increase `n_baseline` to
sample the mean more thoroughly; neither makes a verdict scientifically conclusive.

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

- A passing gate reports a metric comparison for one intervention on one prompt. Vocab
  readouts and activation coefficients are descriptive projections, not semantic labels.
- Results are single-head and prompt-specific. Nothing here assembles a multi-head circuit.
- The worked example and demo do not establish named subfunctions, statistical separation
  from arbitrary directions, or replication of the paper's causal subfunction taxonomy.
- Output-metric changes include downstream responses. A small final-logit effect does not
  imply a small direct write contribution, nor identify which downstream components alter
  the effect.
- Verdicts near the mean control magnitude can change with the seed or draw count. Fixing
  a seed makes the comparison repeatable, not scientifically conclusive.

## Links

- [SVD Circuits demo](../generated/demos/SVD_Circuits_Demo.ipynb)
- Areeb Ahmad, Abhinav Joshi, Ashutosh Modi, "Beyond Components: Singular Vector-Based
  Interpretability of Transformer Circuits", [arXiv 2511.20273](https://arxiv.org/abs/2511.20273)
