"""Attribution patching — linearized activation patching on ``TransformerBridge``.

Attribution patching estimates the causal effect of every model component on a
task metric with a *gradient-based linearization* of activation patching: rather
than one forward pass per intervention, it reads a single gradient cache. For a
clean/corrupt prompt pair it runs a clean forward (for ``a_clean``) and a corrupt
forward whose backward hooks capture ``g = d(metric)/d(a)`` (for ``a_corrupt`` and
its gradient), and scores each node with the
first-order Taylor estimate ``effect(node) = (a_clean - a_corrupt) . g``. Scores
over a batch of clean/corrupt pairs are averaged before ranking.

Only the ``TransformerBridge`` API is targeted; TransformerLens v4 deprecates
``HookedTransformer``.

Sign/direction convention (denoising form): the gradient is taken on the
*corrupt* run and the estimate points *toward* the clean activation, so a
positive score means patching that node from corrupt toward clean moves the
metric in the positive direction. The oracle-parity test (PR5) maps this
convention onto the pinned reference rather than assuming the two agree.

Memory note: gradients are retained only for hook points passing ``names_filter``.
Retaining gradients at every hook point roughly doubles cache memory, so callers
should filter to the hook families their analysis actually reads.

Scope: this PR ships node granularity with plain attribution (``ig_steps=1``).
Edge scoring (EAP), the integrated-gradient path (EAP-IG, ``ig_steps>1``), and
ablate-outside faithfulness land in follow-on PRs; their API is declared here —
``granularity="edge"`` and ``ig_steps>1`` raise :class:`NotImplementedError` — so
downstream code can pin against a stable surface now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence, Union

import torch

MetricFn = Callable[[torch.Tensor], torch.Tensor]
NamesFilter = Union[str, Sequence[str], Callable[[str], bool], None]

NodeKind = Literal["embed", "attn_head_out", "mlp_out"]
Granularity = Literal["node", "edge"]


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


@dataclass(frozen=True)
class EdgeAttributionConfig:
    """Configuration for an attribution-patching sweep.

    The two axes are deliberately orthogonal:

    - ``granularity`` selects what is scored: ``"node"`` scores each residual-stream
      write, ``"edge"`` scores each ``(source, destination)`` write->read pair.
    - ``ig_steps`` selects gradient fidelity. ``ig_steps=1`` is plain attribution
      patching / EAP: a single first-order Taylor gradient taken at the corrupt
      point. ``ig_steps>1`` is EAP-IG: the integrated gradient averaged over that
      many points along the corrupt->clean path, which corrects the gradient
      saturation that makes plain attribution unfaithful.

    There is intentionally **no** ``method`` field. An earlier design had both a
    ``method`` enum (``"attribution"``/``"EAP"``/``"EAP-IG"``) and ``ig_steps``,
    which overlap: the method is fully determined by ``granularity`` and whether
    ``ig_steps`` exceeds 1. Collapsing them removes the invalid states (e.g.
    ``method="attribution", ig_steps=5``).

    This substrate PR implements node granularity with plain attribution only.
    ``granularity="edge"`` and ``ig_steps>1`` are accepted by the type but raise
    :class:`NotImplementedError` at construction, so downstream code can import and
    reference this API now while the edge sweep (PR2) and the integrated-gradient
    path (PR3) land later. When PR3 ships EAP-IG, the default flips to the
    proposal's ``ig_steps=5`` (EAP-IG is the faithful default); until then the
    default is the only executable value, ``ig_steps=1``.

    Attributes:
        granularity: ``"node"`` or ``"edge"``. Defaults to ``"node"``.
        ig_steps: Integrated-gradient path steps (``>=1``). Defaults to ``1``.
    """

    granularity: Granularity = "node"
    ig_steps: int = 1

    def __post_init__(self) -> None:
        if self.ig_steps < 1:
            raise ValueError(f"ig_steps must be >= 1, got {self.ig_steps}")
        if self.granularity == "edge":
            raise NotImplementedError(
                "granularity='edge' (EAP edge scoring) lands in PR2; this substrate "
                "PR implements granularity='node' only."
            )
        if self.ig_steps > 1:
            raise NotImplementedError(
                "ig_steps>1 (EAP-IG integrated gradients) lands in PR3; this substrate "
                "PR implements ig_steps=1 (plain attribution) only."
            )


@dataclass
class AttributionResult:
    """Scored output of an attribution-patching sweep.

    Attributes:
        node_scores: Signed first-order effect estimate per node,
            ``(a_clean - a_corrupt) . d(metric)/d(a)``. A positive score means
            patching that node from corrupt toward clean moves the metric in the
            positive direction (the denoising convention pinned in the module
            docstring).
        edge_scores: Per-edge effect estimate keyed by ``(source, destination)``.
            Declared here so the result API is stable across the PR series; it is
            populated only from PR2 (edge sweep) and is empty for a node sweep.
    """

    node_scores: dict[Node, float]
    edge_scores: dict[tuple[Node, Node], float] = field(default_factory=dict)

    def top_nodes(self, k: int = 10) -> list[tuple[Node, float]]:
        """The ``k`` nodes with the largest effect magnitude, strongest first.

        Ranking is by absolute score: a node with a large negative effect is as
        causally important as one with a large positive effect, so magnitude — not
        signed value — orders the circuit. Ties keep enumeration order (stable
        sort). Requesting more than the available nodes returns all of them.
        """
        ranked = sorted(self.node_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        return ranked[:k]

    def top_edges(self, k: int = 10) -> list[tuple[Node, Node, float]]:
        """The ``k`` highest-magnitude edges — populated from PR2's edge sweep."""
        raise NotImplementedError(
            "edge scoring lands in PR2; run a node-granularity sweep and use "
            "top_nodes() in this PR."
        )


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
    compute_gradient: bool = True,
) -> GradientCache:
    """Run one forward and capture activations plus (optionally) their metric-gradients.

    Registers a forward hook and a backward hook at each cached point, runs one
    grad-enabled forward, then drives a single backward with
    :func:`torch.autograd.grad` to fire the backward hooks.
    ``run_with_cache(..., incl_bwd=True)`` only backpropagates the model's own
    scalar output, so a custom (non-scalar-output) metric such as a logit-diff
    needs the backward driven here.

    Gradients come from the backward hooks, not from ``.grad`` on the cached
    tensors. ``TransformerBridge`` reshapes the tensor handed to a forward hook at
    a converted point (``attn.hook_z``, ``hook_q/k/v``, ``hook_attn_out``), so that
    tensor is a view the model's forward never consumes: ``retain_grad()`` on it is
    inert and its ``.grad`` stays ``None``. A backward hook goes through the same
    conversion and delivers the real gradient in canonical shape. Driving the
    backward with :func:`torch.autograd.grad` instead of ``metric.backward()``
    keeps it off every parameter's ``.grad`` buffer — no caller grads clobbered, no
    model-sized buffer allocated.

    With ``compute_gradient=False`` the backward is skipped entirely: only forward
    hooks are registered, no backward is driven, and every cached point's gradient
    is ``None``. Callers that read activations only (the clean pass of
    :func:`attribution_patch`, which pairs these with the *corrupt* run's gradients)
    use this to run a plain forward instead of a needless forward + backward.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg.n_layers``,
            ``hook_dict``, and the ``hooks()`` context manager.
        tokens: Input token ids for a single forward pass.
        metric_fn: Maps the model logits to a scalar to differentiate.
        names_filter: Restricts which hook points are cached (and have gradients
            retained). ``None`` (the default) caches the node-granularity hook set —
            ``hook_embed`` plus each layer's ``attn.hook_z`` and ``hook_mlp_out``;
            pass an explicit filter to cache any hook point outside that set. On a
            real Bridge, ``None`` cannot mean "every hook point": the gated points
            (``hook_mlp_in``, ``attn.hook_result``, the split-QKV inputs) that
            ``hook_dict`` exposes raise in ``add_hook`` unless their ``set_use_*``
            flag is on.
        compute_gradient: When ``True`` (default) capture gradients via backward
            hooks. When ``False`` run an activation-only forward and leave every
            gradient ``None``.

    Returns:
        A :class:`GradientCache` with per-hook activations, and gradients when
        ``compute_gradient`` is ``True`` (all ``None`` otherwise).
    """
    if compute_gradient and not torch.is_grad_enabled():
        raise ValueError(
            "cache_activation_and_gradient needs autograd, but gradient tracking "
            "is off (torch.no_grad(), set_grad_enabled(False), or inference mode)."
        )

    if names_filter is None:
        # Default to the node-granularity hook set rather than every hook point: a
        # real Bridge's hook_dict holds gated points (hook_mlp_in, attn.hook_result,
        # the split-QKV inputs) that add_hook rejects unless their set_use_* flag is
        # on, so caching "everything" raises. Pass an explicit filter to reach them.
        names_filter = _required_hook_names(int(model.cfg.n_layers))
    predicate = _as_predicate(names_filter)
    names = [name for name in model.hook_dict if predicate(name)]
    if not names:
        raise ValueError("names_filter matched no hook points")

    live: dict[str, torch.Tensor] = {}
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, Optional[torch.Tensor]] = {}

    def make_fwd_hook(name: str) -> Callable[..., None]:
        def hook(tensor: torch.Tensor, *, hook: Any) -> None:
            del hook
            if isinstance(tensor, torch.Tensor):
                live[name] = tensor
                activations[name] = tensor.detach().clone()
            return None

        return hook

    def make_bwd_hook(name: str) -> Callable[..., None]:
        def hook(grad: torch.Tensor, *, hook: Any) -> None:
            del hook
            if isinstance(grad, torch.Tensor):
                gradients[name] = grad.detach().clone()
            return None

        return hook

    fwd_hooks = [(name, make_fwd_hook(name)) for name in names]

    if not compute_gradient:
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(tokens)
            metric = metric_fn(logits)
            if metric.dim() != 0:
                raise ValueError(
                    f"metric_fn must return a scalar tensor, got shape {tuple(metric.shape)}"
                )
        return GradientCache(
            activations=activations,
            gradients={name: None for name in activations},
            metric=metric.detach(),
        )

    bwd_hooks = [(name, make_bwd_hook(name)) for name in names]

    with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        logits = model(tokens)
        metric = metric_fn(logits)
        if metric.dim() != 0:
            raise ValueError(
                f"metric_fn must return a scalar tensor, got shape {tuple(metric.shape)}"
            )
        captured_names = [name for name in names if name in live]
        # torch.autograd.grad only *drives* the backward; the gradients we keep are
        # the converted-shape ones the backward hooks write into `gradients`. Unlike
        # metric.backward() it touches no parameter `.grad` buffer.
        returned = (
            torch.autograd.grad(
                metric,
                inputs=[live[name] for name in captured_names],
                allow_unused=True,
                retain_graph=False,
            )
            if captured_names
            else ()
        )

    # The most-upstream cached point's own backward hook never fires — nothing
    # requested is upstream of it, so the backward stops at it — but its cached
    # tensor is on-path and unconverted, so torch.autograd.grad returns that
    # gradient directly. Backfill any point the hooks missed from that return.
    for name, grad in zip(captured_names, returned):
        if gradients.get(name) is None:
            gradients[name] = None if grad is None else grad.detach().clone()

    return GradientCache(activations=activations, gradients=gradients, metric=metric.detach())


def _node_effects(
    clean_cache: GradientCache,
    corrupt_cache: GradientCache,
    nodes: Sequence[Node],
) -> dict[Node, float]:
    """Score every node with the first-order attribution ``(a_clean - a_corrupt) . g``.

    The gradient ``g`` is taken from the corrupt cache (denoising convention). For
    each node the feature dimension (``d_model``, or ``d_head`` for an attention
    head) is contracted at the node's position/head, giving one signed scalar.
    """
    scores: dict[Node, float] = {}
    for node in nodes:
        name = node.hook_name
        grad = corrupt_cache.gradients.get(name)
        if grad is None:
            raise ValueError(
                f"node {node} reads {name!r}, but the corrupt cache holds no gradient "
                "there; cache with a names_filter that retains this hook point."
            )
        delta = clean_cache.activations[name] - corrupt_cache.activations[name]
        contribution = delta * grad
        if node.kind == "attn_head_out":
            value = contribution[0, node.position, node.head].sum()
        else:
            value = contribution[0, node.position].sum()
        scores[node] = float(value)
    return scores


def attribution_patch(
    model: Any,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    metric_fn: MetricFn,
    config: EdgeAttributionConfig = EdgeAttributionConfig(),
) -> AttributionResult:
    """Estimate every node's causal effect on ``metric_fn`` in two forwards + one backward.

    For each clean/corrupt pair this runs a clean forward (for ``a_clean``) and a
    corrupt forward whose backward hooks capture ``g = d(metric)/d(a)`` (for
    ``a_corrupt`` and its gradient), then scores each node with the
    first-order Taylor estimate ``effect(node) = (a_clean - a_corrupt) . g``.

    Sign/direction convention (denoising form): gradients are taken on the *corrupt*
    run and the estimate points *toward* the clean activation, so a positive score
    means patching that node from corrupt toward clean moves the metric in the
    positive direction. PR5's oracle-parity test maps this convention onto the
    pinned reference rather than assuming the two agree.

    Dataset averaging: ``clean``/``corrupt`` may hold a batch of prompt pairs. Each
    pair is scored independently (per-example forward/backward, so its own
    reconstruction identity holds) and per-node scores are averaged across the batch
    before ranking (proposal step 6).

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg.n_layers``,
            ``hook_dict``, and ``hooks()``.
        clean: Clean token ids, shape ``[batch, seq]``.
        corrupt: Corrupt token ids, shape ``[batch, seq]``, paired row-by-row with
            ``clean``.
        metric_fn: Maps single-example logits to a scalar to differentiate.
        config: Sweep configuration. This PR supports node granularity with plain
            attribution (``ig_steps=1``) only; other values raise at construction.

    Returns:
        An :class:`AttributionResult` whose ``node_scores`` are averaged over the
        batch. ``edge_scores`` stays empty until PR2.

    Raises:
        ValueError: if ``clean``/``corrupt`` are not 2D, hold a different number of
            pairs, or a pair tokenizes to different lengths (activations must align
            position-by-position).
    """
    del config  # node granularity + ig_steps=1 only this PR; enforced at construction.

    if clean.ndim != 2 or corrupt.ndim != 2:
        raise ValueError(
            "attribution_patch expects 2D [batch, seq] token tensors, got clean "
            f"{tuple(clean.shape)} and corrupt {tuple(corrupt.shape)}"
        )
    if clean.shape[0] != corrupt.shape[0]:
        raise ValueError(
            "clean and corrupt must hold the same number of prompt pairs, got "
            f"{clean.shape[0]} and {corrupt.shape[0]}"
        )
    if clean.shape[1] != corrupt.shape[1]:
        raise ValueError(
            "each clean/corrupt pair must tokenize to the same length; got clean "
            f"length {clean.shape[1]} and corrupt length {corrupt.shape[1]}. "
            "Attribution patching aligns activations position-by-position."
        )

    node_hook_names = _required_hook_names(int(model.cfg.n_layers))
    batch = int(clean.shape[0])
    totals: dict[Node, float] = {}

    for index in range(batch):
        clean_cache = cache_activation_and_gradient(
            model,
            clean[index : index + 1],
            metric_fn,
            names_filter=node_hook_names,
            compute_gradient=False,
        )
        corrupt_cache = cache_activation_and_gradient(
            model, corrupt[index : index + 1], metric_fn, names_filter=node_hook_names
        )
        nodes = enumerate_nodes(model, corrupt_cache)
        for node, score in _node_effects(clean_cache, corrupt_cache, nodes).items():
            totals[node] = totals.get(node, 0.0) + score

    node_scores = {node: total / batch for node, total in totals.items()}
    return AttributionResult(node_scores=node_scores)
