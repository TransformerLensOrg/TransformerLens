# Attribution Notes

Note DLA breakdown by source only, is really the "how information is moved" to get the correct
answer. The why is a bit more complex and it depends on QK which needs a different type of
attribution (e.g. DLA by attention pattern, which can highlight either smart queries or smart keys).
We could call these "smart queries and smart keys detectors".

Breakdown DLA by:

- Components
- Attention:
  - Layers
  - Heads
  - Source tokens (i.e. decompose heads by source token which is linear, i.e. L_H_T_)
  - Source components (i.e. break down source tokens by source tokens at each layer by their
    contributing components, which is just the stack of components up until that point inc. embed,
    i.e. L_H_T_[MLP_/L_H_]). Note layer norm - for the position x there are no layer norms to deal
    with so easy!
  - Where source components are heads, their source tokens... and then their source components...
- MLPS:
  - Layers
  - Neurons
  - Not layer source component (because the neurons have a non-linearity so the output is not a sum
    of the input tensors transformed).
  - Could be neuron source component (assumes all other neurons are fixed so it's approximately linear), but
    note that this doesn't really capture things with superposition as it would then be non-linear.
    Since we expect superposition, this may only work with SoLu models. Otherwise we could stick the
    limit of DLA here.

We want:

- Logit attribution function (mostly works so just document better)
- Document function to get residual directions
- Residual decomposition going way deeper (this should become it's own set of functions?)
- Recursive logit attribution