"""Attribution patching — linearized activation patching on ``TransformerBridge``.

This module estimates the causal effect of model components on a task metric with
a *gradient-based linearization* of activation patching: instead of one forward
pass per intervention, it reads a single gradient cache. This first commit ships
only the substrate — a names-filtered forward pass that caches activations
together with the gradient of a custom metric with respect to each of them.

Only the ``TransformerBridge`` API is targeted; TransformerLens v4 deprecates
``HookedTransformer``.

Sign/direction convention (denoising form): the gradient is taken on the
*corrupt* run and the estimate points *toward* the clean activation. Node/edge
scoring built on top of this substrate lands in follow-on commits.

Memory note: gradients are retained only for hook points passing ``names_filter``.
Retaining gradients at every hook point roughly doubles cache memory, so callers
should filter to the hook families their analysis actually reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import torch

MetricFn = Callable[[torch.Tensor], torch.Tensor]
NamesFilter = Union[str, Sequence[str], Callable[[str], bool], None]


@dataclass
class GradientCache:
    """Activations and their metric-gradients from one forward + backward pass.

    Attributes:
        activations: Detached activation tensor per cached hook name.
        gradients: ``d(metric)/d(activation)`` per cached hook name.
        metric: The scalar metric value at this run (detached).
    """

    activations: dict[str, torch.Tensor]
    gradients: dict[str, Optional[torch.Tensor]]
    metric: torch.Tensor


def _as_predicate(names_filter: NamesFilter) -> Callable[[str], bool]:
    if names_filter is None:
        return lambda name: True
    if isinstance(names_filter, str):
        target = names_filter
        return lambda name: name == target
    if callable(names_filter):
        return names_filter
    wanted = set(names_filter)
    return lambda name: name in wanted


def cache_activation_and_gradient(
    model: Any,
    tokens: torch.Tensor,
    metric_fn: MetricFn,
    names_filter: NamesFilter = None,
) -> GradientCache:
    """Run one forward + one manual backward, caching activations and gradients.

    ``run_with_cache(..., incl_bwd=True)`` only backpropagates the model's own
    scalar output, so a custom (non-scalar-output) metric such as a logit-diff
    needs the backward call done by hand — that is what this helper does.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``hook_dict`` and
            the ``hooks()`` context manager.
        tokens: Input token ids for a single forward pass.
        metric_fn: Maps the model logits to a scalar to differentiate.
        names_filter: Restricts which hook points are cached (and have gradients
            retained). ``None`` caches every hook point.

    Returns:
        A :class:`GradientCache` with per-hook activations and gradients.
    """
    if not torch.is_grad_enabled():
        raise ValueError(
            "cache_activation_and_gradient needs autograd, but gradient tracking "
            "is off (torch.no_grad(), set_grad_enabled(False), or inference mode)."
        )

    predicate = _as_predicate(names_filter)
    names = [name for name in model.hook_dict if predicate(name)]
    if not names:
        raise ValueError("names_filter matched no hook points")

    live: dict[str, torch.Tensor] = {}
    activations: dict[str, torch.Tensor] = {}

    def make_hook(name: str) -> Callable[..., None]:
        def hook(tensor: torch.Tensor, *, hook: Any) -> None:
            del hook
            if isinstance(tensor, torch.Tensor):
                tensor.retain_grad()
                live[name] = tensor
                activations[name] = tensor.detach().clone()
            return None

        return hook

    fwd_hooks = [(name, make_hook(name)) for name in names]

    model.zero_grad(set_to_none=True)
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)

    metric = metric_fn(logits)
    if metric.dim() != 0:
        raise ValueError(f"metric_fn must return a scalar tensor, got shape {tuple(metric.shape)}")
    metric.backward()

    gradients: dict[str, Optional[torch.Tensor]] = {
        name: live[name].grad for name in names if name in live
    }
    return GradientCache(activations=activations, gradients=gradients, metric=metric.detach())
