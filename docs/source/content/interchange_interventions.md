# Interchange Interventions

An interchange intervention runs the model on a *base* input while setting the base's
activation at one hook point, inside a linear subspace, to its value on a *source* input:

$$
h' = h_{\mathrm{base}} + s \, (h_{\mathrm{source}} - h_{\mathrm{base}}) B B^\top,
$$

where $B$ is an orthonormal `[d_model, rank]` basis (the convention of
[`orthonormal_subspace`](projection_kernel.md)) and $s$ is the strength, 1 by default. The base
keeps its own coordinates outside the subspace.

The question it answers is causal. Suppose a high-level model says the output depends on a
variable, such as a subject's grammatical number. If interchanging a subspace makes the model
produce the output that the high-level model predicts for the base with that variable taken from
the source, the subspace carries the variable at that hook point (Geiger et al., 2021). The share
of base/source pairs where this happens is the *interchange intervention accuracy* (IIA).

## Run an intervention

```python
import torch

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import (
    SubspaceInterchange,
    collect_activations,
    intervened_logits,
    mean_difference_basis,
)

bridge = TransformerBridge.boot_transformers("gpt2")
answers = [bridge.to_single_token(" is"), bridge.to_single_token(" are")]
task_logits = lambda tokens: bridge(tokens)[:, -1, answers]

base = bridge.to_tokens(["The key near the cabinet", "The keys near the cabinet"])
source = base.flip(0)  # each base takes the other prompt's subject number
subject = torch.full((2,), 2)  # the subject token, after the BOS token
hook = "blocks.1.hook_out"

source_acts = collect_activations(bridge, hook, source, task_logits, positions=subject)
basis = mean_difference_basis(source_acts, labels=[1, 0])
logits = intervened_logits(
    bridge, hook, SubspaceInterchange(basis), base, source_acts, task_logits, positions=subject
)
print(task_logits(base).argmax(-1), logits.argmax(-1))  # tensor([0, 1]) tensor([1, 0])
```

`task_logits` runs the model in one forward pass and reads the task out of its output, so the
tools do not assume next-token prediction; the hook must fire exactly once in that pass.
Activations at the hook must be `[batch, d_model]`, or `[batch, pos, d_model]` with one position
index per example. A real experiment fits the basis on held-out pairs, not on the pairs it
evaluates.

`collect_activations` and `intervened_logits` run in batches of `batch_size` without gradients and
return CPU tensors, so memory does not grow with the number of pairs. `run_with_intervention` is
the same intervention as one differentiable forward pass, for training an intervention's
parameters.

## Bases and baselines

- `mean_difference_basis(activations, labels)`: the span of the centred class means; for two
  classes, the difference-in-means direction.
- `probe_basis(weights)`: the directions of a linear probe's weights. Undo any input scaling
  before passing them.
- `random_basis(d_model, rank, seed)`: a seeded random orthonormal basis.

Run every basis on the same pairs. A basis whose IIA does not clearly exceed a random basis of
the same rank is not evidence that the subspace carries the variable.

## Placebo

`NormMatchedPlacebo(reference, basis)` moves each base exactly as far as `reference` does, but
inside another subspace, usually `random_basis` with the same rank. If the placebo also changes the
output, part of the effect comes from displacing the activation by that much, not from the
aligned subspace.

## Interpretation

An IIA measures what an intervention does under the chosen pairs, hook point, positions and
basis. It does not by itself show that the model uses the variable: subspace interventions can
activate pathways the model does not use on its own inputs (Makelov et al., 2024). Report the
IIA beside the placebo and the baselines.

## References

- Geiger, Lu, Icard and Potts (2021). Causal Abstractions of Neural Networks. NeurIPS.
- Makelov, Lange and Nanda (2024). Is This the Subspace You Are Looking for? An Interpretability
  Illusion for Subspace Activation Patching. ICLR.
