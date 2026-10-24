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
hook = "blocks.1.hook_out"


def pairs(nouns, place):
    """Bases and sources differ only in the subject's number; the target is the source's verb."""
    singular = bridge.to_tokens([f"The {n} {place}" for n in nouns])
    plural = bridge.to_tokens([f"The {n}s {place}" for n in nouns])
    base, source = torch.cat([singular, plural]), torch.cat([plural, singular])
    targets = (torch.arange(len(base)) < len(nouns)).long()  # " are" when the source is plural
    subject = torch.full((len(base),), 2)  # the subject token, after the BOS token and "The"
    return base, source, targets, subject


nouns = ["key", "dog", "cat", "book", "car", "boy", "girl", "bird"]
fit_base, fit_source, fit_targets, fit_at = pairs(nouns, "near the cabinet")
test_base, test_source, test_targets, test_at = pairs(nouns, "behind the table")

# Fit the basis on the fit pairs' sources, labelled by subject number ...
fit_acts = collect_activations(bridge, hook, fit_source, task_logits, positions=fit_at)
basis = mean_difference_basis(fit_acts, labels=fit_targets.tolist())

# ... and measure the interchange accuracy (IIA) on held-out pairs
test_acts = collect_activations(bridge, hook, test_source, task_logits, positions=test_at)
logits = intervened_logits(
    bridge, hook, SubspaceInterchange(basis), test_base, test_acts, task_logits, positions=test_at
)
print((logits.argmax(-1) == test_targets).float().mean())  # tensor(1.)
```

`task_logits` runs the model in one forward pass and reads the task out of its output, so the
tools do not assume next-token prediction; the hook must fire exactly once in that pass.
Activations at the hook must be `[batch, d_model]`, or `[batch, pos, d_model]` with one position
index per example. Fit a basis on one set of pairs and evaluate it on pairs it was not fit on, as
here: the held-out pairs put the same nouns in a new frame. Reusing the nouns gives the
arbitrary-label control below keys it can learn; holding out nouns as well is a stronger test of
the real alignment.

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

## Neuron bases

`neuron_basis(d_mlp, neurons)` turns a set of MLP neurons into a basis: interchanging it sets
exactly those neurons to the source's values, at a neuron hook such as `blocks.N.mlp.hook_post`.
Its placebo is `random_neuron_basis(d_mlp, k, seed, exclude=...)`, the same number of neurons drawn
at random. A random direction is the wrong placebo for a claim about neurons, because it spreads
over every neuron.

A sparse probe (`fit_sparse_probe`; Gurnee et al., 2023) finds the neurons that best separate a
variable, but that is decodability, not use. Interchanging them is the causal test:

```python
from transformer_lens.tools.analysis import fit_sparse_probe, neuron_basis, random_neuron_basis

mlp_hook = "blocks.0.mlp.hook_post"
fit_neurons = collect_activations(bridge, mlp_hook, fit_source, task_logits, positions=fit_at)
test_neurons = collect_activations(bridge, mlp_hook, test_source, task_logits, positions=test_at)
probe = fit_sparse_probe(fit_neurons, fit_targets, k=1)

d_mlp = bridge.cfg.d_mlp
for basis in (
    neuron_basis(d_mlp, probe.selected_features),
    random_neuron_basis(d_mlp, 1, seed=0, exclude=probe.selected_features),
):
    logits = intervened_logits(
        bridge, mlp_hook, SubspaceInterchange(basis), test_base, test_neurons, task_logits,
        positions=test_at,
    )
    print((logits.argmax(-1) == test_targets).float().mean())  # tensor(1.), then tensor(0.)
```

Compare the selected neurons with random neurons of the same size, and with difference-in-means and
DAS at the same hook. Where DAS succeeds but neither the model's own directions nor its neurons
do, suspect the illusion Makelov et al. (2024) describe for subspace patching of MLP activations.

## Learn the subspace with DAS

Distributed alignment search (DAS; Geiger et al., 2024) learns the basis instead of fixing it.
`fit_das` trains an orthonormal basis so that the interchange makes the model produce, for each
base/source pair, the class the high-level model predicts:

```python
from transformer_lens.tools.analysis import fit_das

result = fit_das(
    bridge, hook, fit_base, fit_source, fit_targets, task_logits, rank=1,
    base_positions=fit_at, source_positions=fit_at, epochs=10,
)
logits = intervened_logits(
    bridge, hook, SubspaceInterchange(result.subspace), test_base, test_acts, task_logits,
    positions=test_at,
)
print((logits.argmax(-1) == test_targets).float().mean())  # tensor(1.)
```

`targets[i]` is the high-level model's output for base `i` with the variable taken from source
`i`. Fit on one set of pairs and evaluate the IIA on pairs the basis was not trained on. Include
training pairs where several variables differ between base and source, labelled by the high-level
model's interchange: otherwise the basis can learn whatever the training pairs happen to move
together. The model's parameters are frozen during training and restored afterwards;
`micro_batch_size` accumulates gradients over smaller parts of each batch to save memory.

DAS is expressive enough to fit arbitrary label mappings and to find effects in randomly
initialised models (Arora et al., 2024), so its IIA needs controls trained the same way.

## Trained controls

Train each control with the same pairs, hook point, rank, budget and seeds as the real alignment,
and evaluate it on the same held-out pairs:

- **Arbitrary labels.** `arbitrary_targets(keys, n_classes, seed)` gives each key a fixed
  pseudo-random class. DAS trained on these shows how much IIA the method's expressivity alone
  produces. Key the labels by something that recurs between the fit and held-out pairs, such as
  the source's subject word: on keys the fit pairs never contained, the labels are fresh coin
  flips, every subspace scores chance, and the control can reject nothing.
- **An untrained model.** `random_init_bridge(model_name, seed=...)` builds the architecture with
  its HuggingFace initialisation and no training; `reinitialized(model, seed)` does the same for a
  `HookedRootModule`. An effect DAS finds there does not come from anything the model learned.

```python
from transformer_lens.tools.analysis import arbitrary_targets, random_init_bridge

# Arbitrary labels: a fixed pseudo-random class per source subject word, which recurs between the
# fit and held-out pairs
labels = lambda sources: arbitrary_targets([bridge.to_string(s[2]) for s in sources], 2, seed=0)
arbitrary = fit_das(
    bridge, hook, fit_base, fit_source, labels(fit_source), task_logits, rank=1,
    base_positions=fit_at, source_positions=fit_at, epochs=10,
)
logits = intervened_logits(
    bridge, hook, SubspaceInterchange(arbitrary.subspace), test_base, test_acts, task_logits,
    positions=test_at,
)
print((logits.argmax(-1) == labels(test_source)).float().mean())  # tensor(0.6250)

# An untrained model: the same pairs and targets, its own activations and readout
untrained = random_init_bridge("gpt2", seed=0)
untrained_logits = lambda tokens: untrained(tokens)[:, -1, answers]
random_init = fit_das(
    untrained, hook, fit_base, fit_source, fit_targets, untrained_logits, rank=1,
    base_positions=fit_at, source_positions=fit_at, epochs=10,
)
untrained_acts = collect_activations(
    untrained, hook, test_source, untrained_logits, positions=test_at
)
logits = intervened_logits(
    untrained, hook, SubspaceInterchange(random_init.subspace), test_base, untrained_acts,
    untrained_logits, positions=test_at,
)
print((logits.argmax(-1) == test_targets).float().mean())  # tensor(0.8125)
```

Here the arbitrary labels reach 0.62 and the untrained model 0.81, against 1.0 for the real
alignment: DAS finds a subspace that moves even an untrained model's output, which is why the real
alignment's IIA has to be read against both. Report the real alignment's IIA minus each control's
(its selectivity), with uncertainty over the evaluation pairs. A subspace that does not clearly
beat both controls is not evidence that the model carries the variable there.

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
- Arora, Jurafsky and Potts (2024). CausalGym: Benchmarking causal interpretability methods on
  linguistic tasks. ACL.
- Geiger, Wu, Potts, Icard and Goodman (2024). Finding Alignments Between Interpretable Causal
  Variables and Distributed Neural Representations. CLeaR.
- Gurnee, Nanda, Pauly, Harvey, Troitskii and Bertsimas (2023). Finding Neurons in a Haystack:
  Case Studies with Sparse Probing. arXiv:2305.01610.
- Makelov, Lange, Geiger and Nanda (2024). Is This the Subspace You Are Looking for? An
  Interpretability Illusion for Subspace Activation Patching. ICLR.
