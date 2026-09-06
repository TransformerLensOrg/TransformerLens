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
from typing import Any, Callable, Literal, Optional, Sequence, Union

import torch

MetricFn = Callable[[torch.Tensor], torch.Tensor]
NamesFilter = Union[str, Sequence[str], Callable[[str], bool], None]

NodeKind = Literal["embed", "attn_head_out", "mlp_out"]


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


@dataclass(frozen=True)
class Node:
    """A node in the residual-stream computational graph at node granularity.

    Nodes are the typed, hashable keys the attribution sweep scores. Each node
    is identified by ``(kind, layer, position, head)``; ``kind`` selects the node
    family and constrains which of ``layer``/``head`` apply:

    - ``"embed"``: the token embedding write. ``layer`` and ``head`` are ``None``.
    - ``"attn_head_out"``: one attention head's output. ``layer`` and ``head`` set.
    - ``"mlp_out"``: one layer's MLP output. ``layer`` set, ``head`` is ``None``.

    ``position`` is the sequence index the node is read at. The invariants above
    are enforced in ``__post_init__`` so a malformed key raises rather than
    silently producing a wrong graph (Risk 1: the explicit-graph guard).
    """

    kind: NodeKind
    position: int
    layer: Optional[int] = None
    head: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == "embed":
            if self.layer is not None or self.head is not None:
                raise ValueError("embed nodes take neither layer nor head")
        elif self.kind == "attn_head_out":
            if self.layer is None or self.head is None:
                raise ValueError("attn_head_out nodes need both layer and head")
        elif self.kind == "mlp_out":
            if self.layer is None:
                raise ValueError("mlp_out nodes need a layer")
            if self.head is not None:
                raise ValueError("mlp_out nodes take no head")
        else:
            raise ValueError(f"unknown node kind {self.kind!r}")

    @property
    def hook_name(self) -> str:
        """The cache hook point this node reads from.

        Uses the standard ``TransformerBridge`` alias names (``hook_embed``,
        ``blocks.{l}.attn.hook_z``, ``blocks.{l}.hook_mlp_out``); the per-head
        ``attn_head_out`` node slices head ``self.head`` out of the shared
        ``hook_z`` tensor.
        """
        if self.kind == "embed":
            return "hook_embed"
        if self.kind == "attn_head_out":
            return f"blocks.{self.layer}.attn.hook_z"
        return f"blocks.{self.layer}.hook_mlp_out"


def _required_hook_names(n_layers: int) -> list[str]:
    """Hook points the node graph reads: embed plus per-layer attn-z and mlp-out."""
    names = ["hook_embed"]
    for layer in range(n_layers):
        names.append(f"blocks.{layer}.attn.hook_z")
        names.append(f"blocks.{layer}.hook_mlp_out")
    return names


def enumerate_nodes(model: Any, cache: GradientCache) -> list[Node]:
    """Enumerate the full node-granularity graph from the Bridge hook graph.

    The graph is *explicit*: for ``n_layers`` layers it always contains the embed
    write, every attention head's output, and every layer's MLP output, at every
    sequence position. Sequence length and head count are read from the cached
    tensor shapes; ``n_layers`` from ``model.cfg``.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg.n_layers``.
        cache: A :class:`GradientCache` holding at least the required hook points.

    Returns:
        The node list, ordered embed-then-layerwise for deterministic ranking.

    Raises:
        ValueError: if any required hook point is absent from ``cache`` — the
            graph is never silently truncated (Risk 1).
    """
    n_layers = int(model.cfg.n_layers)
    missing = [name for name in _required_hook_names(n_layers) if name not in cache.activations]
    if missing:
        raise ValueError(
            "node graph requires hook points missing from the cache: "
            + ", ".join(missing)
            + ". Cache with a names_filter that keeps hook_embed, "
            "blocks.*.attn.hook_z, and blocks.*.hook_mlp_out."
        )

    seq_len = cache.activations["hook_embed"].shape[1]

    nodes: list[Node] = [Node(kind="embed", position=position) for position in range(seq_len)]
    for layer in range(n_layers):
        n_heads = cache.activations[f"blocks.{layer}.attn.hook_z"].shape[2]
        for position in range(seq_len):
            for head in range(n_heads):
                nodes.append(Node(kind="attn_head_out", layer=layer, head=head, position=position))
            nodes.append(Node(kind="mlp_out", layer=layer, position=position))
    return nodes


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
