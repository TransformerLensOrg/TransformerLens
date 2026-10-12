"""Attribution patching and edge-ablation faithfulness on ``TransformerBridge``.

Node and edge scores are the first-order ``(a_clean - a_corrupt) . g`` with ``g`` taken on the
corrupt run (denoising form), so a positive score means patching corrupt toward clean raises the
metric. :func:`faithfulness` ablates every edge outside a circuit by rewriting each reader's pre-LN
fork input; the residual stream is a running sum, so that rewrite is exact.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Collection,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Union,
)

import torch

from transformer_lens.tools.analysis._model_state import require_eval_mode

MetricFn = Callable[[torch.Tensor], torch.Tensor]
NamesFilter = Union[str, Sequence[str], Callable[[str], bool], None]

NodeKind = Literal[
    "embed", "attn_head_out", "mlp_out", "q_input", "k_input", "v_input", "mlp_in", "logits"
]
Granularity = Literal["node", "edge"]
AblationMode = Literal["corrupt", "mean"]
EdgeClass = Literal["into_qk", "into_v", "into_mlp", "into_logits"]


@dataclass
class GradientCache:
    """Activations and their metric-gradients from one forward + backward pass."""

    activations: dict[str, torch.Tensor]
    gradients: dict[str, Optional[torch.Tensor]]
    metric: torch.Tensor


@dataclass(frozen=True)
class Node:
    """Hashable graph key; ``__post_init__`` enforces which kinds take ``layer``/``head``."""

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
        """The hook point this node reads; per-head kinds slice ``head`` out of it."""
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
    """Sweep config; ``ig_steps>1`` (EAP-IG) is reserved and raises ``NotImplementedError``."""

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
    """Pair-averaged scores; an edge sweep's ``node_scores`` are per-writer edge-score sums."""

    node_scores: dict[Node, float]
    edge_scores: dict[tuple[Node, Node], float] = field(default_factory=dict)

    def top_nodes(self, k: int = 10) -> list[tuple[Node, float]]:
        """The ``k`` largest-magnitude nodes, strongest first; ties keep enumeration order."""
        ranked = sorted(self.node_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        return ranked[:k]

    def top_edges(self, k: int = 10) -> list[tuple[Node, Node, float]]:
        """The ``k`` largest-magnitude edges, strongest first; ties keep enumeration order."""
        ranked = sorted(self.edge_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        return [(writer, reader, score) for (writer, reader), score in ranked[:k]]


@dataclass(frozen=True)
class FaithfulnessConfig:
    """How out-of-circuit writers are replaced: the corrupt run's outputs or a held-out mean."""

    ablation: AblationMode = "corrupt"


@dataclass
class FaithfulnessResult:
    """Gap recovery for a circuit; ``edge_class_recovered`` also ablates one class's kept edges
    (classes absent from the circuit reuse ``recovered``) and is not additive across classes."""

    recovered: float  # (m_ablated - m_corrupt) / (m_clean - m_corrupt); may leave [0, 1]
    full_metric: float
    corrupt_metric: float
    circuit_size: int
    total_edges: int
    edge_class_recovered: dict[EdgeClass, float] = field(default_factory=dict)
    edge_class_counts: dict[EdgeClass, int] = field(default_factory=dict)  # over the full graph


def _required_hook_names(n_layers: int, granularity: Granularity = "node") -> list[str]:
    """Hook points a node sweep reads; ``"edge"`` adds per-head results and split-QKV inputs."""
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


@contextmanager
def _edge_hook_flags(model: Any) -> Iterator[None]:
    """Turn on the Bridge flags gating the edge hook points; restore the caller's flags on exit."""
    cfg = model.cfg
    saved_attn_result = cfg.use_attn_result
    saved_split_qkv_input = cfg.use_split_qkv_input
    saved_hook_mlp_in = cfg.use_hook_mlp_in
    saved_attn_in = bool(getattr(cfg, "use_attn_in", False))
    try:
        # use_attn_in and use_split_qkv_input are mutually exclusive.
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
        # Only after use_split_qkv_input is restored, which a use_attn_in caller had off.
        if saved_attn_in:
            model.set_use_attn_in(True)


def _check_required_hooks(
    cache: GradientCache, required_names: Sequence[str], graph: str, hint: str
) -> None:
    """Raise on a missing hook point rather than silently building a truncated graph."""
    missing = [name for name in required_names if name not in cache.activations]
    if missing:
        raise ValueError(
            f"{graph} requires hook points missing from the cache: "
            + ", ".join(missing)
            + f". {hint}"
        )


def enumerate_nodes(model: Any, cache: GradientCache) -> list[Node]:
    """Embed, every head and every MLP at every position, ordered embed-then-layerwise."""
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
    """Guard enumeration against a construction bug that repeats a ``(writer, reader)`` pair."""
    seen: set[tuple[Node, Node]] = set()
    for edge in edges:
        if edge in seen:
            raise ValueError(f"edge {edge} enumerated more than once")
        seen.add(edge)


def _edge_hook_names(n_layers: int) -> list[str]:
    """Every hook point an edge sweep caches, including the MLP-entry and logits readers."""
    names = _required_hook_names(n_layers, granularity="edge")
    names.extend(f"blocks.{layer}.hook_mlp_in" for layer in range(n_layers))
    names.append(f"blocks.{n_layers - 1}.hook_resid_post")
    return names


def _writer_hook_names(n_layers: int) -> list[str]:
    """The writer-output hook family: embed, per-head ``attn.hook_result`` and MLP out."""
    names = ["hook_embed"]
    for layer in range(n_layers):
        names.append(f"blocks.{layer}.attn.hook_result")
        names.append(f"blocks.{layer}.hook_mlp_out")
    return names


def enumerate_edges(model: Any, cache: GradientCache) -> list[tuple[Node, Node]]:
    """Every per-position ``(writer, reader)`` edge; each reader is fed by all upstream writers,
    and every writer feeds a terminal logits reader, so its edge sum equals its node score."""
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", None)
    # K/V reader nodes are indexed by query head, which overruns the shared K/V heads.
    if n_kv_heads is not None and n_kv_heads != model.cfg.n_heads:
        raise ValueError(
            "Edge granularity requires n_key_value_heads == n_heads, but this model has "
            f"n_key_value_heads={n_kv_heads} and n_heads={model.cfg.n_heads}; GQA/MQA is "
            "unsupported here (node granularity still works)."
        )
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
                # The MLP reads the layer input, so this layer's heads do not feed it; they
                # still feed later readers and the logits reader.
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


def _select_hook_names(model: Any, names_filter: NamesFilter) -> list[str]:
    if names_filter is None:
        names = list(model.hook_dict)
    elif isinstance(names_filter, str):
        names = [name for name in model.hook_dict if name == names_filter]
    elif callable(names_filter):
        names = [name for name in model.hook_dict if names_filter(name)]
    else:
        wanted = set(names_filter)
        names = [name for name in model.hook_dict if name in wanted]
    if not names:
        raise ValueError("names_filter matched no hook points")
    return names


def cache_activation_and_gradient(
    model: Any,
    tokens: torch.Tensor,
    metric_fn: MetricFn,
    names_filter: NamesFilter = None,
    compute_gradient: bool = True,
) -> GradientCache:
    """Run one forward caching activations and, unless ``compute_gradient=False``, their
    ``d(metric)/d(activation)``; ``metric_fn`` must return a scalar."""
    if compute_gradient and not torch.is_grad_enabled():
        raise ValueError(
            "cache_activation_and_gradient needs autograd, but gradient tracking "
            "is off (torch.no_grad(), set_grad_enabled(False), or inference mode)."
        )

    if names_filter is None:
        # Not every hook point: gated Bridge hooks (hook_mlp_in, attn.hook_result, split-QKV
        # inputs) raise in add_hook until their set_use_* flag is on.
        names_filter = _required_hook_names(int(model.cfg.n_layers))
    names = _select_hook_names(model, names_filter)

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
    # run_with_cache(incl_bwd=True) backprops only the model's own output, not metric_fn's.
    # Backward hooks, not retain_grad(): at converted points (attn.hook_z, hook_q/k/v) the Bridge
    # hands forward hooks a reshaped view the forward never consumes, so its .grad stays None.
    bwd_hooks = [(name, make_bwd_hook(name)) for name in names] if compute_gradient else []

    with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        logits = model(tokens)
        metric = metric_fn(logits)
        if metric.dim() != 0:
            raise ValueError(
                f"metric_fn must return a scalar tensor, got shape {tuple(metric.shape)}"
            )
        if not compute_gradient:
            return GradientCache(
                activations=activations,
                gradients={name: None for name in activations},
                metric=metric.detach(),
            )
        captured_names = [name for name in names if name in live]
        # autograd.grad only drives the backward for the hooks, and unlike metric.backward()
        # leaves every parameter's .grad untouched.
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

    # The most-upstream cached point's own backward hook never fires -- nothing
    # requested is upstream of it, so the backward stops at it -- but its cached
    # tensor is on-path and unconverted, so torch.autograd.grad returns that
    # gradient directly. Backfill any point the hooks missed from that return.
    for name, grad in zip(captured_names, returned):
        if gradients.get(name) is None:
            gradients[name] = None if grad is None else grad.detach().clone()

    return GradientCache(activations=activations, gradients=gradients, metric=metric.detach())


def capture_writer_outputs(
    model: Any,
    tokens: torch.Tensor,
    names_filter: NamesFilter = None,
) -> dict[str, torch.Tensor]:
    """Forward-only, no-grad capture of the writer-output hooks (or ``names_filter``); gated
    hooks such as ``attn.hook_result`` fire only inside :func:`_edge_hook_flags`."""
    if names_filter is None:
        names_filter = _writer_hook_names(int(model.cfg.n_layers))
    names = _select_hook_names(model, names_filter)

    captured: dict[str, torch.Tensor] = {}

    def make_fwd_hook(name: str) -> Callable[..., None]:
        def hook(tensor: torch.Tensor, *, hook: Any) -> None:
            del hook
            if isinstance(tensor, torch.Tensor):
                captured[name] = tensor.detach().clone()
            return None

        return hook

    fwd_hooks = [(name, make_fwd_hook(name)) for name in names]
    with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
        model(tokens)

    return captured


def _node_slice(node: Node, tensor: torch.Tensor) -> torch.Tensor:
    """``tensor[0, position]``, further indexed by ``head`` for per-head kinds (``Node`` allows a
    head on no other kind)."""
    if node.head is None:
        return tensor[0, node.position]
    return tensor[0, node.position, node.head]


def _node_effects(
    clean_cache: GradientCache,
    corrupt_cache: GradientCache,
    nodes: Sequence[Node],
) -> dict[Node, float]:
    """Score each node ``(a_clean - a_corrupt) . g`` with ``g`` from the corrupt run."""
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
        scores[node] = float(_node_slice(node, delta * grad).sum())
    return scores


def _writer_hook_name(node: Node) -> str:
    """A writer's ``d_model`` output hook; heads use ``attn.hook_result``, not ``hook_z``."""
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
    """Score each edge ``(a_clean[w] - a_corrupt[w]) . d(metric)/d(input of r)``, one matvec per
    reader since a reader's incoming edges share its gradient."""
    scores: dict[tuple[Node, Node], float] = {}
    writer_deltas: dict[str, torch.Tensor] = {}

    def writer_delta_vec(writer: Node) -> torch.Tensor:
        name = _writer_hook_name(writer)
        delta = writer_deltas.get(name)
        if delta is None:
            delta = clean_cache.activations[name] - corrupt_cache.activations[name]
            writer_deltas[name] = delta
        return _node_slice(writer, delta)

    writers_by_reader: dict[Node, list[Node]] = {}
    reader_order: list[Node] = []
    for writer, reader in edges:
        bucket = writers_by_reader.get(reader)
        if bucket is None:
            writers_by_reader[reader] = bucket = []
            reader_order.append(reader)
        bucket.append(writer)

    for reader in reader_order:
        reader_name = reader.hook_name
        grad = corrupt_cache.gradients.get(reader_name)
        writers = writers_by_reader[reader]
        if grad is None:
            raise ValueError(
                f"edge {(writers[0], reader)} reads its gradient at {reader_name!r}, but "
                "the corrupt cache holds none there; cache with a names_filter that "
                "retains this hook point."
            )
        delta_matrix = torch.stack([writer_delta_vec(writer) for writer in writers])
        edge_values = delta_matrix @ _node_slice(reader, grad)
        for writer, value in zip(writers, edge_values.tolist()):
            scores[(writer, reader)] = value
    return scores


def _aggregate_edge_scores_to_writer_nodes(
    edge_scores: dict[tuple[Node, Node], float],
) -> dict[Node, float]:
    """Sum each writer's outgoing edge scores; its logits edge makes this its node score."""
    totals: dict[Node, float] = {}
    for (writer, _reader), score in edge_scores.items():
        totals[writer] = totals.get(writer, 0.0) + score
    return totals


def _excluded_writers_by_reader(
    edges: Sequence[tuple[Node, Node]],
    circuit: Collection[tuple[Node, Node]],
) -> dict[Node, list[Node]]:
    """Out-of-circuit writers per reader, so each reader is corrected once, not once per edge."""
    circuit_set = set(circuit)
    excluded: dict[Node, list[Node]] = {}
    for writer, reader in edges:
        if (writer, reader) not in circuit_set:
            excluded.setdefault(reader, []).append(writer)
    return excluded


def _make_writer_capture_hook(name: str, sink: dict[str, torch.Tensor]) -> Callable[..., None]:
    """Record a writer's live output; writers fire before the readers they feed, which only read
    it, so it is detached but not cloned."""

    def hook(tensor: torch.Tensor, *, hook: Any) -> None:
        del hook
        if isinstance(tensor, torch.Tensor):
            sink[name] = tensor.detach()
        return None

    return hook


def _make_edge_ablation_hook(
    reader: Node,
    excluded_writers: Sequence[Node],
    live_writers: dict[str, torch.Tensor],
    replacement_writers: dict[str, torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Swap each excluded writer's live output for its replacement in the reader's own
    position/head slice, so readers sharing a hook point never double-correct."""
    index: tuple[int, ...] = (
        (0, reader.position) if reader.head is None else (0, reader.position, reader.head)
    )

    def hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
        del hook
        rebuilt = tensor.clone()
        for writer in excluded_writers:
            name = _writer_hook_name(writer)
            live = live_writers.get(name)
            replacement = replacement_writers.get(name)
            if live is None or replacement is None:
                raise ValueError(
                    f"cannot ablate the edge from {writer} into {reader}: no "
                    f"contribution captured at {name!r}. Enable the writer's hook "
                    "family before running the ablation."
                )
            rebuilt[index] = (
                rebuilt[index] - _node_slice(writer, live) + _node_slice(writer, replacement)
            )
        return rebuilt

    return hook


def _ablate_edges(
    model: Any,
    tokens: torch.Tensor,
    edges: Sequence[tuple[Node, Node]],
    circuit: Collection[tuple[Node, Node]],
    replacement_writers: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Logits of a no-grad forward with every edge outside ``circuit`` ablated; the caller must
    hold :func:`_edge_hook_flags` so the gated writer and reader hooks fire."""
    excluded_by_reader = _excluded_writers_by_reader(edges, circuit)

    live_writers: dict[str, torch.Tensor] = {}
    fwd_hooks: list[tuple[str, Callable[..., Any]]] = [
        (name, _make_writer_capture_hook(name, live_writers))
        for name in _writer_hook_names(int(model.cfg.n_layers))
    ]
    fwd_hooks.extend(
        (
            reader.hook_name,
            _make_edge_ablation_hook(reader, writers, live_writers, replacement_writers),
        )
        for reader, writers in excluded_by_reader.items()
    )

    with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
        logits = model(tokens)
    return logits


def attribution_patch(
    model: Any,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    metric_fn: MetricFn,
    config: EdgeAttributionConfig = EdgeAttributionConfig(),
) -> AttributionResult:
    """Score every node or edge on ``[batch, seq]`` pairs, averaged over pairs. ``metric_fn`` gets
    one pair's logits and no pair index, so per-pair labels need one call per pair."""
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
        hook_names = _edge_hook_names(n_layers)
        edge_totals: dict[tuple[Node, Node], float] = {}

        with _edge_hook_flags(model):
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


def _normalize_circuit(
    circuit: Sequence[tuple[Node, Node]] | Sequence[tuple[Node, Node, float]],
) -> list[tuple[Node, Node]]:
    """Dedupe circuit pairs in first-seen order; ``top_edges``' scored triples are accepted."""
    normalized: list[tuple[Node, Node]] = []
    seen: set[tuple[Node, Node]] = set()
    for entry in circuit:
        if len(entry) == 2:
            writer, reader = entry
        elif len(entry) == 3:
            writer, reader, _score = entry
        else:
            raise ValueError(
                f"circuit entries must be (writer, reader) or (writer, reader, score), "
                f"got a {len(entry)}-tuple"
            )
        edge = (writer, reader)
        if edge not in seen:
            normalized.append(edge)
            seen.add(edge)
    return normalized


def _edge_class(edge: tuple[Node, Node]) -> EdgeClass:
    """An edge's class by what its reader consumes; Q and K share one, as both feed scores."""
    reader = edge[1]
    if reader.kind in ("q_input", "k_input"):
        return "into_qk"
    if reader.kind == "v_input":
        return "into_v"
    if reader.kind == "mlp_in":
        return "into_mlp"
    if reader.kind == "logits":
        return "into_logits"
    raise ValueError(f"{reader.kind} is a writer kind and has no edge class")


def _mean_writer_contributions(
    model: Any,
    mean_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Average calibration examples, preserving position, head, and singleton batch axes."""
    count = int(mean_tokens.shape[0])
    if count == 0:
        raise ValueError("mean_tokens must contain at least one calibration example")
    totals: dict[str, torch.Tensor] = {}
    for index in range(count):
        captured = capture_writer_outputs(model, mean_tokens[index : index + 1])
        for name, tensor in captured.items():
            running = totals.get(name)
            totals[name] = tensor if running is None else running + tensor
    return {name: total / count for name, total in totals.items()}


def faithfulness(
    model: Any,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    metric_fn: MetricFn,
    circuit: Sequence[tuple[Node, Node]] | Sequence[tuple[Node, Node, float]],
    config: FaithfulnessConfig = FaithfulnessConfig(),
    *,
    mean_tokens: Optional[torch.Tensor] = None,
) -> FaithfulnessResult:
    """Forward-only: ablate every edge outside ``circuit`` on one ``[1, seq]`` pair and report the
    gap recovered. ``"mean"`` ablation needs held-out ``mean_tokens`` of the same length."""
    if clean.ndim != 2 or corrupt.ndim != 2:
        raise ValueError(
            "faithfulness expects 2D [batch, seq] token tensors, got clean "
            f"{tuple(clean.shape)} and corrupt {tuple(corrupt.shape)}"
        )
    if clean.shape[0] != corrupt.shape[0]:
        raise ValueError(
            "clean and corrupt must hold the same number of prompt pairs, got "
            f"{clean.shape[0]} and {corrupt.shape[0]}"
        )
    if clean.shape[0] != 1:
        raise ValueError(
            "faithfulness measures one clean/corrupt pair at a time, since the "
            f"metric reads a single example; got {clean.shape[0]} pairs. Loop over "
            "pairs and aggregate the recovered fractions yourself."
        )
    if clean.shape[1] != corrupt.shape[1]:
        raise ValueError(
            "each clean/corrupt pair must tokenize to the same length; got clean "
            f"length {clean.shape[1]} and corrupt length {corrupt.shape[1]}. "
            "Faithfulness aligns activations position-by-position."
        )

    if config.ablation == "mean":
        if mean_tokens is None:
            raise ValueError("mean ablation requires independent calibration via mean_tokens")
        if mean_tokens.ndim != 2:
            raise ValueError("mean_tokens must be a 2D [n_calibration, seq] token tensor")
        if mean_tokens.shape[0] == 0:
            raise ValueError("mean_tokens must contain at least one calibration example")
        if mean_tokens.shape[1] != clean.shape[1]:
            raise ValueError("mean_tokens must have the same sequence length as the evaluated pair")
        if mean_tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "mean_tokens must contain token ids with dtype torch.int32 or torch.int64"
            )
        if mean_tokens.device != clean.device or mean_tokens.device != corrupt.device:
            raise ValueError("mean_tokens must be on the same device as the clean/corrupt tokens")
        if (mean_tokens == clean).all(dim=1).any() or (mean_tokens == corrupt).all(dim=1).any():
            raise ValueError("mean_tokens must not contain either evaluated clean/corrupt endpoint")
    elif mean_tokens is not None:
        raise ValueError("mean_tokens is only accepted with mean ablation")

    require_eval_mode(model, operation="faithfulness()")

    kept = _normalize_circuit(circuit)
    n_layers = int(model.cfg.n_layers)
    hook_names = _edge_hook_names(n_layers)

    with torch.no_grad(), _edge_hook_flags(model):
        corrupt_cache = cache_activation_and_gradient(
            model, corrupt, metric_fn, names_filter=hook_names, compute_gradient=False
        )
        edges = enumerate_edges(model, corrupt_cache)
        graph = set(edges)
        unknown = [edge for edge in kept if edge not in graph]
        if unknown:
            raise ValueError(
                f"circuit holds {len(unknown)} edge(s) that are not in the graph, "
                f"starting with {unknown[0]}; a stale or mistyped circuit would "
                "otherwise ablate nothing."
            )

        if mean_tokens is not None:
            replacements = _mean_writer_contributions(model, mean_tokens)
        else:
            replacements = capture_writer_outputs(model, corrupt)

        clean_metric = float(metric_fn(model(clean)))
        corrupt_metric = float(corrupt_cache.metric)
        gap = clean_metric - corrupt_metric
        if gap == 0.0:
            raise ValueError(
                "the clean and corrupt runs produce the same metric, so the recovered "
                "fraction is undefined; choose a pair the metric separates."
            )
        ablated_metric = float(metric_fn(_ablate_edges(model, clean, edges, kept, replacements)))

        recovered = (ablated_metric - corrupt_metric) / gap
        class_counts: dict[EdgeClass, int] = {}
        for edge in edges:
            edge_class = _edge_class(edge)
            class_counts[edge_class] = class_counts.get(edge_class, 0) + 1

        kept_classes = {_edge_class(edge) for edge in kept}
        class_recovered: dict[EdgeClass, float] = {}
        for edge_class in class_counts:
            if edge_class not in kept_classes:
                class_recovered[edge_class] = recovered
                continue
            # Removing a class must not restore edges excluded by the candidate.
            reduced = [edge for edge in kept if _edge_class(edge) != edge_class]
            class_metric = float(
                metric_fn(_ablate_edges(model, clean, edges, reduced, replacements))
            )
            class_recovered[edge_class] = (class_metric - corrupt_metric) / gap

    return FaithfulnessResult(
        recovered=recovered,
        full_metric=clean_metric,
        corrupt_metric=corrupt_metric,
        circuit_size=len(kept),
        total_edges=len(edges),
        edge_class_recovered=class_recovered,
        edge_class_counts=class_counts,
    )
