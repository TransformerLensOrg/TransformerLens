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
metric in the positive direction. An oracle-parity test maps this convention
onto a pinned reference rather than assuming the two agree.

Memory note: gradients are retained only for hook points passing ``names_filter``.
Retaining gradients at every hook point roughly doubles cache memory, so callers
should filter to the hook families their analysis actually reads.

Scope: this build ships node and edge granularity with plain attribution
(``ig_steps=1``). The integrated-gradient path (EAP-IG, ``ig_steps>1``) and
ablate-outside faithfulness are not implemented yet; their API is declared here,
and ``ig_steps>1`` raises :class:`NotImplementedError`, so downstream code can pin
against a stable surface now.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Union

import torch

from transformer_lens.tools.analysis._model_state import require_eval_mode

MetricFn = Callable[[torch.Tensor], torch.Tensor]
NamesFilter = Union[str, Sequence[str], Callable[[str], bool], None]

NodeKind = Literal[
    "embed", "attn_head_out", "mlp_out", "q_input", "k_input", "v_input", "mlp_in", "logits"
]
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
    """A node in the residual-stream computational graph.

    Nodes are the typed, hashable keys the attribution sweep scores. Each node
    is identified by ``(kind, layer, position, head)``; ``kind`` selects the node
    family and constrains which of ``layer``/``head`` apply. Three kinds are
    *writers* -- they contribute a value into the residual stream:

    - ``"embed"``: the token embedding write. ``layer`` and ``head`` are ``None``.
    - ``"attn_head_out"``: one attention head's output. ``layer`` and ``head`` set.
    - ``"mlp_out"``: one layer's MLP output. ``layer`` set, ``head`` is ``None``.

    Five kinds are *readers* -- they consume the residual stream as an edge's
    destination (see :func:`enumerate_edges`):

    - ``"q_input"`` / ``"k_input"`` / ``"v_input"``: one attention head's split
      Q/K/V input. ``layer`` and ``head`` set.
    - ``"mlp_in"``: one layer's MLP entry. ``layer`` set, ``head`` is ``None``.
    - ``"logits"``: the terminal readout of the final residual, read at
      ``blocks.{n_layers-1}.hook_resid_post``. ``layer`` is that final layer and
      ``head`` is ``None``. Every writer feeds this reader, so a writer's
      aggregate over its outgoing edges equals its direct node score.

    ``position`` is the sequence index the node is read at. The invariants above
    are enforced in ``__post_init__`` so a malformed key raises rather than
    silently producing a wrong graph.
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
        elif self.kind in ("q_input", "k_input", "v_input"):
            if self.layer is None or self.head is None:
                raise ValueError(f"{self.kind} nodes need both layer and head")
        elif self.kind == "mlp_in":
            if self.layer is None:
                raise ValueError("mlp_in nodes need a layer")
            if self.head is not None:
                raise ValueError("mlp_in nodes take no head")
        elif self.kind == "logits":
            if self.layer is None:
                raise ValueError("logits nodes need a layer")
            if self.head is not None:
                raise ValueError("logits nodes take no head")
        else:
            raise ValueError(f"unknown node kind {self.kind!r}")

    @property
    def hook_name(self) -> str:
        """The cache hook point this node reads from.

        Uses the standard ``TransformerBridge`` alias names (``hook_embed``,
        ``blocks.{l}.attn.hook_z``, ``blocks.{l}.hook_mlp_out``,
        ``blocks.{l}.attn.hook_q_input``/``hook_k_input``/``hook_v_input``,
        ``blocks.{l}.hook_mlp_in``, ``blocks.{l}.hook_resid_post``); the per-head
        nodes (``attn_head_out``, ``q_input``, ``k_input``, ``v_input``) slice
        head ``self.head`` out of the shared per-head tensor.
        """
        if self.kind == "embed":
            return "hook_embed"
        if self.kind == "attn_head_out":
            return f"blocks.{self.layer}.attn.hook_z"
        if self.kind == "mlp_out":
            return f"blocks.{self.layer}.hook_mlp_out"
        if self.kind in ("q_input", "k_input", "v_input"):
            return f"blocks.{self.layer}.attn.hook_{self.kind}"
        if self.kind == "logits":
            return f"blocks.{self.layer}.hook_resid_post"
        return f"blocks.{self.layer}.hook_mlp_in"


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

    This build implements node and edge granularity with plain attribution.
    ``ig_steps>1`` is accepted by the type but raises :class:`NotImplementedError`
    at construction, so downstream code can import and reference this API now
    while the integrated-gradient path is not implemented yet. Once EAP-IG lands,
    the default flips to ``ig_steps=5`` (EAP-IG is the faithful default); until
    then the default is the only executable value, ``ig_steps=1``.

    Attributes:
        granularity: ``"node"`` or ``"edge"``. Defaults to ``"node"``.
        ig_steps: Integrated-gradient path steps (``>=1``). Defaults to ``1``.
    """

    granularity: Granularity = "node"
    ig_steps: int = 1

    def __post_init__(self) -> None:
        if self.ig_steps < 1:
            raise ValueError(f"ig_steps must be >= 1, got {self.ig_steps}")
        if self.ig_steps > 1:
            raise NotImplementedError(
                "ig_steps>1 (EAP-IG integrated gradients) is not implemented yet; this "
                "build supports ig_steps=1 (plain attribution) only."
            )


@dataclass
class AttributionResult:
    """Scored output of an attribution-patching sweep.

    Attributes:
        node_scores: Signed first-order effect estimate per node,
            ``(a_clean - a_corrupt) . d(metric)/d(a)``. A positive score means
            patching that node from corrupt toward clean moves the metric in the
            positive direction (the denoising convention pinned in the module
            docstring). For an edge-granularity sweep this is instead each
            writer's aggregate over its own outgoing edge scores, which omits
            its direct skip-connection contribution to the metric (see
            :func:`attribution_patch`).
        edge_scores: Per-edge effect estimate keyed by ``(source, destination)``.
            Populated for an edge-granularity sweep; empty for a node sweep.
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
        """The ``k`` edges with the largest effect magnitude, strongest first.

        Ranking is by absolute score, matching ``top_nodes``: a large negative
        edge effect is as causally important as a large positive one. Ties keep
        enumeration order (stable sort). Requesting more than the available
        edges returns all of them.
        """
        ranked = sorted(self.edge_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        return [(writer, reader, score) for (writer, reader), score in ranked[:k]]


def _required_hook_names(n_layers: int, granularity: Granularity = "node") -> list[str]:
    """Hook points a sweep at ``granularity`` reads.

    Node granularity needs the embed write plus each layer's attn-z and
    mlp-out. Edge granularity additionally needs the per-head hook points on
    both sides of an edge into or out of an attention head: ``attn.hook_result``
    (writer -- a head's own contribution before the sum into the residual
    stream) and the split ``attn.hook_q_input``/``hook_k_input``/``hook_v_input``
    (reader -- the residual each head's Q/K/V projection reads separately).
    """
    names = ["hook_embed"]
    for layer in range(n_layers):
        names.append(f"blocks.{layer}.attn.hook_z")
        names.append(f"blocks.{layer}.hook_mlp_out")
        if granularity == "edge":
            names.append(f"blocks.{layer}.attn.hook_result")
            names.append(f"blocks.{layer}.attn.hook_q_input")
            names.append(f"blocks.{layer}.attn.hook_k_input")
            names.append(f"blocks.{layer}.attn.hook_v_input")
    return names


def _ensure_edge_hook_flags(model: Any) -> None:
    """Turn on the Bridge flags edge granularity's hook points require.

    ``attn.hook_result``, the split ``attn.hook_q_input``/``hook_k_input``/
    ``hook_v_input``, and ``hook_mlp_in`` all exist on the Bridge
    unconditionally but only fire when their owning flag
    (``cfg.use_attn_result`` / ``cfg.use_split_qkv_input`` /
    ``cfg.use_hook_mlp_in``) is on, so an edge sweep must enable all three
    before caching or the writer- and reader-side hook points it needs never
    populate.

    Memory caveat: enabling ``use_attn_result``/``use_split_qkv_input`` makes
    every cached per-head tensor ``[batch, seq, n_heads, d_model]`` instead of
    the summed ``[batch, seq, d_model]`` residual. That is fine on a model the
    size of gpt2-small; it does not scale to models with many heads or layers.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg`` and
            ``set_use_attn_result``/``set_use_split_qkv_input``/
            ``set_use_hook_mlp_in``.
    """
    if not model.cfg.use_attn_result:
        model.set_use_attn_result(True)
    if not model.cfg.use_split_qkv_input:
        model.set_use_split_qkv_input(True)
    if not model.cfg.use_hook_mlp_in:
        model.set_use_hook_mlp_in(True)


@contextmanager
def _edge_hook_flags(model: Any) -> Iterator[None]:
    """Enable the Bridge flags an edge sweep needs, then restore the caller's state.

    ``attn.hook_result``, the split ``attn.hook_q_input``/``hook_k_input``/
    ``hook_v_input``, and ``hook_mlp_in`` all exist on the Bridge unconditionally
    but only fire when their owning flag (``cfg.use_attn_result`` /
    ``cfg.use_split_qkv_input`` / ``cfg.use_hook_mlp_in``) is on, so an edge sweep
    must enable all three before caching or the writer- and reader-side hook
    points it needs never populate.

    ``use_split_qkv_input`` is mutually exclusive with ``use_attn_in``, so a
    caller who arrives with ``use_attn_in`` on would otherwise trip the
    exclusivity error. This turns ``use_attn_in`` off before enabling the split
    input, and restores it after ``use_split_qkv_input`` has been turned back off.

    Invariant: the caller's flag state (``use_attn_result``,
    ``use_split_qkv_input``, ``use_hook_mlp_in``, ``use_attn_in``) is unchanged on
    return, including when the body raises. Restoration runs in a ``finally``
    block so a raise mid-sweep -- or in the caller's own later code -- cannot
    leave the per-head tensors materialized on the model.

    Memory caveat: enabling ``use_attn_result``/``use_split_qkv_input`` makes
    every cached per-head tensor ``[batch, seq, n_heads, d_model]`` instead of
    the summed ``[batch, seq, d_model]`` residual. That is fine on a model the
    size of gpt2-small; it does not scale to models with many heads or layers.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg`` and
            ``set_use_attn_result``/``set_use_split_qkv_input``/
            ``set_use_hook_mlp_in``/``set_use_attn_in``.
    """
    cfg = model.cfg
    saved_attn_result = cfg.use_attn_result
    saved_split_qkv_input = cfg.use_split_qkv_input
    saved_hook_mlp_in = cfg.use_hook_mlp_in
    saved_attn_in = bool(getattr(cfg, "use_attn_in", False))
    try:
        # use_attn_in and use_split_qkv_input are mutually exclusive; clear
        # use_attn_in first so enabling the split input cannot raise.
        if saved_attn_in:
            model.set_use_attn_in(False)
        if not cfg.use_attn_result:
            model.set_use_attn_result(True)
        if not cfg.use_split_qkv_input:
            model.set_use_split_qkv_input(True)
        if not cfg.use_hook_mlp_in:
            model.set_use_hook_mlp_in(True)
        yield
    finally:
        model.set_use_attn_result(saved_attn_result)
        model.set_use_split_qkv_input(saved_split_qkv_input)
        model.set_use_hook_mlp_in(saved_hook_mlp_in)
        # Re-enabling use_attn_in requires use_split_qkv_input already off; the
        # line above restored it, and a caller with use_attn_in on cannot also
        # have had use_split_qkv_input on, so this cannot trip the exclusivity.
        if saved_attn_in:
            model.set_use_attn_in(True)


def _check_required_hooks(
    cache: GradientCache, required_names: Sequence[str], graph: str, hint: str
) -> None:
    """Raise if any of ``required_names`` is absent from ``cache.activations``.

    Shared by every granularity's graph-construction step, so a cache built
    with too narrow a ``names_filter`` -- or one produced while a required
    Bridge flag was off -- fails loudly instead of silently producing a
    truncated graph.
    """
    missing = [name for name in required_names if name not in cache.activations]
    if missing:
        raise ValueError(
            f"{graph} requires hook points missing from the cache: "
            + ", ".join(missing)
            + f". {hint}"
        )


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
            graph is never silently truncated.
    """
    n_layers = int(model.cfg.n_layers)
    _check_required_hooks(
        cache,
        _required_hook_names(n_layers),
        "node graph",
        "Cache with a names_filter that keeps hook_embed, blocks.*.attn.hook_z, "
        "and blocks.*.hook_mlp_out.",
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


def _assert_edges_unique(edges: Sequence[tuple[Node, Node]]) -> None:
    """Raise if any writer -> reader pair appears more than once in ``edges``.

    A writer feeding two distinct readers (a head's output feeding both the
    next layer's attention input and this layer's MLP input, say) is two
    edges; this guards the enumeration against a construction bug that
    collapses or duplicates a single ``(writer, reader)`` pair instead.
    """
    seen: set[tuple[Node, Node]] = set()
    for edge in edges:
        if edge in seen:
            raise ValueError(f"edge {edge} enumerated more than once")
        seen.add(edge)


def _required_edge_reader_hook_names(n_layers: int) -> list[str]:
    """The reader hook points edge enumeration additionally requires.

    ``_required_hook_names(..., granularity="edge")`` covers the per-head
    attention hooks a writer/reader pair into or out of a head needs. Edge
    enumeration reads two reader points those miss:

    - each layer's MLP entry, ``attn.hook_mlp_in``'s layer-level sibling
      ``hook_mlp_in``, gated on ``cfg.use_hook_mlp_in`` the same way the per-head
      hooks are gated on their own flags, and
    - the terminal ``blocks.{n_layers-1}.hook_resid_post``, where the logits
      reader takes its gradient. This final residual hook fires unconditionally,
      so no Bridge flag gates it.
    """
    names = [f"blocks.{layer}.hook_mlp_in" for layer in range(n_layers)]
    names.append(f"blocks.{n_layers - 1}.hook_resid_post")
    return names


def _edge_hook_names(n_layers: int) -> list[str]:
    """Every hook point an edge-granularity sweep must cache.

    The per-head attention hooks from ``_required_hook_names(..., granularity="edge")``
    plus each layer's MLP-entry reader hook and the terminal logits reader hook.
    """
    return _required_hook_names(n_layers, granularity="edge") + _required_edge_reader_hook_names(
        n_layers
    )


def enumerate_edges(model: Any, cache: GradientCache) -> list[tuple[Node, Node]]:
    """Enumerate every writer -> reader edge in the residual-stream graph.

    At a fixed sequence position, the residual stream is a running sum: a
    reader (a head's split Q/K/V input, a layer's MLP entry, or the terminal
    logits readout) is fed by every writer (the embed write, every attention
    head's output, every layer's MLP output) that precedes it. Building the
    graph position-by-position tracks which writers are "available" so far and
    connects each new reader to all of them, then adds that layer's writers to
    the available set before moving on -- so a writer never edges to a reader
    upstream of it, and a writer feeding both a direct edge and a through-MLP
    edge produces two distinct ``(u, v)`` pairs rather than one summed together.

    A terminal ``logits`` reader (read at the final ``hook_resid_post``) closes
    the graph: after the per-layer loop every remaining writer -- including the
    final layer's MLP output, which no per-layer reader sees -- edges to it. That
    edge carries the writer's direct skip-connection contribution to the metric,
    so a writer's aggregate over its outgoing edges equals its direct node score.

    On ``cfg.parallel_attn_mlp`` models (Pythia, GPT-J, Falcon, Phi) the MLP
    reads the layer input, not the post-attention residual, so a layer's own
    heads are not writers into that layer's MLP; those same-layer head->mlp_in
    edges are dropped while the heads still feed later readers and the logits
    reader.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg.n_layers``
            and, optionally, ``cfg.parallel_attn_mlp``.
        cache: A :class:`GradientCache` holding at least the required hook points
            for edge granularity.

    Returns:
        The edge list as ``(writer, reader)`` node pairs; no pair repeats.

    Raises:
        ValueError: if any required hook point is absent from ``cache`` -- the
            graph is never silently truncated.
    """
    n_layers = int(model.cfg.n_layers)
    _check_required_hooks(
        cache,
        _edge_hook_names(n_layers),
        "edge graph",
        "Cache with a names_filter that keeps the edge-granularity hook set: "
        "hook_embed, blocks.*.attn.hook_z, blocks.*.hook_mlp_out, "
        "blocks.*.attn.hook_result, blocks.*.attn.hook_q_input, "
        "blocks.*.attn.hook_k_input, blocks.*.attn.hook_v_input, "
        "blocks.*.hook_mlp_in, and blocks.{n_layers-1}.hook_resid_post.",
    )

    parallel_attn_mlp = bool(getattr(model.cfg, "parallel_attn_mlp", False))

    seq_len = cache.activations["hook_embed"].shape[1]
    edges: list[tuple[Node, Node]] = []

    for position in range(seq_len):
        available: list[Node] = [Node(kind="embed", position=position)]
        for layer in range(n_layers):
            n_heads = cache.activations[f"blocks.{layer}.attn.hook_z"].shape[2]

            attn_reader_kinds: tuple[NodeKind, NodeKind, NodeKind] = (
                "q_input",
                "k_input",
                "v_input",
            )
            attn_readers = [
                Node(kind=kind, layer=layer, head=head, position=position)
                for kind in attn_reader_kinds
                for head in range(n_heads)
            ]
            for reader in attn_readers:
                edges.extend((writer, reader) for writer in available)

            layer_heads = [
                Node(kind="attn_head_out", layer=layer, head=head, position=position)
                for head in range(n_heads)
            ]

            mlp_reader = Node(kind="mlp_in", layer=layer, position=position)
            if parallel_attn_mlp:
                # The MLP reads the layer input, not the post-attention residual,
                # so this layer's heads are not writers into its MLP. They still
                # become available to later readers and the logits reader.
                edges.extend((writer, mlp_reader) for writer in available)
                available = available + layer_heads
            else:
                available = available + layer_heads
                edges.extend((writer, mlp_reader) for writer in available)

            available = available + [Node(kind="mlp_out", layer=layer, position=position)]

        logits_reader = Node(kind="logits", layer=n_layers - 1, position=position)
        edges.extend((writer, logits_reader) for writer in available)

    _assert_edges_unique(edges)
    return edges


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


def _writer_hook_name(node: Node) -> str:
    """The residual-stream hook point holding a writer node's own contribution.

    Distinct from ``Node.hook_name``: an ``attn_head_out`` node's ``hook_name``
    resolves to ``attn.hook_z``, the pre-``hook_result`` value node granularity
    scores. An edge's writer contribution must instead be measured in the same
    ``d_model`` space a reader's gradient lives in, which is ``attn.hook_result``
    -- the per-head decomposition of the head's contribution after it is
    projected into the residual stream.
    """
    if node.kind == "embed":
        return "hook_embed"
    if node.kind == "attn_head_out":
        return f"blocks.{node.layer}.attn.hook_result"
    if node.kind == "mlp_out":
        return f"blocks.{node.layer}.hook_mlp_out"
    raise ValueError(f"{node.kind} is a reader kind and has no writer contribution")


def _edge_effects(
    clean_cache: GradientCache,
    corrupt_cache: GradientCache,
    edges: Sequence[tuple[Node, Node]],
) -> dict[tuple[Node, Node], float]:
    """Score every edge with ``(a_clean[u] - a_corrupt[u]) . d(metric)/d(input of v)``.

    Mirrors ``_node_effects``: the delta is the writer's own residual
    contribution (clean minus corrupt cache), dotted with the reader's
    corrupt-run gradient -- the same denoising convention ``_node_effects``
    uses. Unlike a node score, the delta and the gradient are read from two
    different hook points (the writer's and the reader's), since an edge
    measures how much of one component's output reaches another component's
    input.
    """
    scores: dict[tuple[Node, Node], float] = {}
    for writer, reader in edges:
        writer_name = _writer_hook_name(writer)
        reader_name = reader.hook_name
        grad = corrupt_cache.gradients.get(reader_name)
        if grad is None:
            raise ValueError(
                f"edge {(writer, reader)} reads its gradient at {reader_name!r}, but "
                "the corrupt cache holds none there; cache with a names_filter that "
                "retains this hook point."
            )
        delta = clean_cache.activations[writer_name] - corrupt_cache.activations[writer_name]
        if writer.kind == "attn_head_out":
            delta_vec = delta[0, writer.position, writer.head]
        else:
            delta_vec = delta[0, writer.position]
        if reader.kind in ("q_input", "k_input", "v_input"):
            grad_vec = grad[0, reader.position, reader.head]
        else:
            grad_vec = grad[0, reader.position]
        scores[(writer, reader)] = float((delta_vec * grad_vec).sum())
    return scores


def _aggregate_edge_scores_to_writer_nodes(
    edge_scores: dict[tuple[Node, Node], float],
) -> dict[Node, float]:
    """Sum each writer's outgoing edge scores into that writer's aggregate node score.

    A writer's aggregate is the sum of its effects along every edge it feeds.
    This is not the same quantity a node-granularity sweep measures directly at
    the writer's own hook point: enumerate_edges' reader kinds (the per-head
    Q/K/V inputs and the MLP entry) do not include a final-readout reader, so a
    writer's direct skip-connection contribution to the metric -- the part of
    its residual-stream write that is never read by a later component, only
    carried forward by addition -- is absent from the aggregate. A writer with
    no outgoing edge at all (the final layer's MLP output, which nothing in
    this graph reads) has no aggregate entry, even though its direct node score
    is generally nonzero.
    """
    totals: dict[Node, float] = {}
    for (writer, _reader), score in edge_scores.items():
        totals[writer] = totals.get(writer, 0.0) + score
    return totals


def attribution_patch(
    model: Any,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    metric_fn: MetricFn,
    config: EdgeAttributionConfig = EdgeAttributionConfig(),
) -> AttributionResult:
    """Estimate every component's causal effect on ``metric_fn`` in two forwards + one backward.

    For each clean/corrupt pair this runs a clean forward (for ``a_clean``) and a
    corrupt forward whose backward hooks capture ``g = d(metric)/d(a)`` (for
    ``a_corrupt`` and its gradient). At node granularity (``config.granularity ==
    "node"``) each node is scored with the first-order Taylor estimate
    ``effect(node) = (a_clean - a_corrupt) . g``. At edge granularity
    (``config.granularity == "edge"``) each writer -> reader edge is scored with
    ``effect(edge) = (a_clean[writer] - a_corrupt[writer]) . d(metric)/d(input of
    reader)``, and ``node_scores`` holds each writer's aggregate effect (the sum
    of its outgoing edge scores).

    Sign/direction convention (denoising form): gradients are taken on the *corrupt*
    run and the estimate points *toward* the clean activation, so a positive score
    means patching that node from corrupt toward clean moves the metric in the
    positive direction. An oracle-parity test maps this convention onto a pinned
    reference rather than assuming the two agree.

    Dataset averaging: ``clean``/``corrupt`` may hold a batch of prompt pairs. Each
    pair is scored independently (per-example forward/backward, so its own
    reconstruction identity holds) and per-node (or per-edge) scores are averaged
    across the batch before ranking.

    The model and every submodule must be in evaluation mode. Separate clean and
    corrupt forwards cannot produce meaningful activation differences if stochastic
    training layers such as dropout remain active.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg.n_layers``,
            ``hook_dict``, and ``hooks()``.
        clean: Clean token ids, shape ``[batch, seq]``.
        corrupt: Corrupt token ids, shape ``[batch, seq]``, paired row-by-row with
            ``clean``.
        metric_fn: Maps single-example logits to a scalar to differentiate.
        config: Sweep configuration. Node and edge granularity are both
            supported with plain attribution (``ig_steps=1``); ``ig_steps>1``
            raises at construction.

    Returns:
        An :class:`AttributionResult` whose ``node_scores`` are averaged over the
        batch. For an edge-granularity sweep, ``edge_scores`` is populated too and
        ``node_scores`` is the per-writer aggregate of those edge scores. This
        aggregate is not the same quantity a node-granularity sweep on the same
        model returns: it omits each writer's direct skip-connection contribution
        to the metric, since no reader kind models a final readout (a writer with
        no outgoing edge at all, such as the final layer's MLP output, has no
        entry here even though its direct node score is generally nonzero).

    Raises:
        ValueError: if ``clean``/``corrupt`` are not 2D, hold a different number of
            pairs, a pair tokenizes to different lengths (activations must align
            position-by-position), or the model or one of its submodules is in
            training mode.
    """
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

    require_eval_mode(model, operation="attribution_patch()")

    batch = int(clean.shape[0])
    n_layers = int(model.cfg.n_layers)

    if config.granularity == "edge":
        _ensure_edge_hook_flags(model)
        hook_names = _edge_hook_names(n_layers)
        edge_totals: dict[tuple[Node, Node], float] = {}

        for index in range(batch):
            clean_cache = cache_activation_and_gradient(
                model,
                clean[index : index + 1],
                metric_fn,
                names_filter=hook_names,
                compute_gradient=False,
            )
            corrupt_cache = cache_activation_and_gradient(
                model, corrupt[index : index + 1], metric_fn, names_filter=hook_names
            )
            edges = enumerate_edges(model, corrupt_cache)
            for edge, score in _edge_effects(clean_cache, corrupt_cache, edges).items():
                edge_totals[edge] = edge_totals.get(edge, 0.0) + score

        edge_scores = {edge: total / batch for edge, total in edge_totals.items()}
        node_scores = _aggregate_edge_scores_to_writer_nodes(edge_scores)
        return AttributionResult(node_scores=node_scores, edge_scores=edge_scores)

    node_hook_names = _required_hook_names(n_layers)
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
