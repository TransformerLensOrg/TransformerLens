"""Attribution patching — linearized activation patching on ``TransformerBridge``.

Attribution patching estimates the causal effect of every model component on a
task metric with a *gradient-based linearization* of activation patching: rather
than one forward pass per intervention, it reads a single gradient cache. For a
clean/corrupt prompt pair it runs a clean forward (for ``a_clean``) and a corrupt
forward whose backward hooks capture ``g = d(metric)/d(a)`` (for ``a_corrupt`` and
its gradient), and scores each node with the
first-order Taylor estimate ``effect(node) = (a_clean - a_corrupt) . g``. Scores
over a batch of clean/corrupt pairs are averaged before ranking.

Only the ``TransformerBridge`` API is targeted.

Sign/direction convention (denoising form): the gradient is taken on the
*corrupt* run and the estimate points *toward* the clean activation, so a
positive score means patching that node from corrupt toward clean moves the
metric in the positive direction. An oracle-parity test maps this convention
onto a pinned reference rather than assuming the two agree.

Memory note: gradients are retained only for hook points passing ``names_filter``.
Retaining gradients at every hook point roughly doubles cache memory, so callers
should filter to the hook families their analysis actually reads.

Scope: this build ships node and edge granularity with plain attribution
(``ig_steps=1``), plus ablate-outside faithfulness. The integrated-gradient path
(EAP-IG, ``ig_steps>1``) is not implemented yet; its API is declared here, and
``ig_steps>1`` raises :class:`NotImplementedError`, so downstream code can pin
against a stable surface now.

Faithfulness intervenes rather than estimating: it ablates every edge outside a
candidate circuit and reports how much of the clean-to-corrupt metric gap the
circuit recovers. Because the residual stream is a running sum and the reader
hooks fire before the layer norm, each reader's input is rebuilt exactly.
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
    cast,
    get_args,
)

import torch

from transformer_lens.model_bridge.generalized_components.block import (
    VARIANT_SUBMODULE_NAMES,
)
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
    while the integrated-gradient path is not implemented. Once EAP-IG lands,
    the default flips to ``ig_steps=5`` (EAP-IG is the faithful default); until
    then the default is the only executable value, ``ig_steps=1``.

    Attributes:
        granularity: ``"node"`` or ``"edge"``. Defaults to ``"node"``.
        ig_steps: Integrated-gradient path steps (``>=1``). Defaults to ``1``.
    """

    granularity: Granularity = "node"
    ig_steps: int = 1

    def __post_init__(self) -> None:
        if self.granularity not in get_args(Granularity):
            raise ValueError(
                f"granularity must be one of {get_args(Granularity)}, got {self.granularity!r}"
            )
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
            writer's aggregate over its own outgoing edge scores. Because the
            graph has a terminal logits reader that every writer feeds, that
            aggregate equals the writer's direct node score (see
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


@dataclass(frozen=True)
class FaithfulnessConfig:
    """Configuration for an ablate-outside faithfulness measurement.

    Attributes:
        ablation: How an out-of-circuit edge's writer is replaced.
            ``"corrupt"`` (default) substitutes the corrupt run's own contribution;
            ``"mean"`` substitutes that writer's mean over the held-out
            ``mean_tokens`` batch passed to :func:`faithfulness`.
    """

    ablation: AblationMode = "corrupt"

    def __post_init__(self) -> None:
        if self.ablation not in get_args(AblationMode):
            raise ValueError(
                f"ablation must be one of {get_args(AblationMode)}, got {self.ablation!r}"
            )


@dataclass
class FaithfulnessResult:
    """How much of the clean-to-corrupt metric gap a candidate circuit recovers.

    Attributes:
        recovered: ``(m_ablated - m_corrupt) / (m_clean - m_corrupt)``, a fraction
            (``1.0`` = clean metric, ``0.0`` = corrupt metric); values outside
            ``[0, 1]`` are possible, since a circuit can overshoot the clean run.
        full_metric: The clean run's metric, the top of the gap.
        corrupt_metric: The corrupt run's metric, the bottom of the gap.
        circuit_size: Number of edges kept.
        total_edges: Number of edges in the graph.
        edge_class_recovered: Recovery after also removing this class's retained
            edges, with the same replacements and gap; a class absent from the
            circuit has the aggregate recovery. Conditional dependence, not
            additive and not bounded to ``[0, 1]``.
        edge_class_counts: Full-graph edge counts per class, not retained counts.
    """

    recovered: float
    full_metric: float
    corrupt_metric: float
    circuit_size: int
    total_edges: int
    edge_class_recovered: dict[EdgeClass, float] = field(default_factory=dict)
    edge_class_counts: dict[EdgeClass, int] = field(default_factory=dict)


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


# Pre-norm blocks measured at most 2.7e-3 (gpt2, pythia-70m and Qwen2.5-0.5B in bfloat16, CPU
# and MPS; float32 stays below 1e-7); post-norm blocks measured 0.05 to 19.
_ATTN_WRITE_RTOL = 2.5e-2


def _attn_write_hook_names(n_layers: int) -> list[str]:
    """Each layer's attention residual write, cached only to check it against the head sum."""
    return [f"blocks.{layer}.hook_attn_out" for layer in range(n_layers)]


_EDGE_UNSUPPORTED = (
    "edge granularity and faithfulness are unsupported here. Node granularity "
    '(EdgeAttributionConfig(granularity="node")) is available.'
)


def _non_additive_attention_error(detail: str) -> ValueError:
    return ValueError(
        "per-head attention outputs are normalized or scaled before the residual add on this "
        f"architecture ({detail}), so attn.hook_result is not a head's residual contribution: "
        f"{_EDGE_UNSUPPORTED}"
    )


def _attention_write_obstruction(
    model: Any, layers: Optional[Collection[int]] = None
) -> Optional[str]:
    """Describe the first block declaring a norm, scale or parallel mixer after its head sum."""
    for layer, block in enumerate(getattr(model, "blocks", ())):
        if layers is not None and layer not in layers:
            continue
        submodules = getattr(block, "submodules", None)
        if not isinstance(submodules, dict) or "attn" not in submodules:
            continue
        scale = getattr(block, "residual_contribution_scale", 1.0)
        if scale != 1.0:
            return f"blocks.{layer} scales its attention output by {scale}"
        # Falcon-H1 scales attention and adds a parallel Mamba output inline, after
        # attn.hook_out, so the cached head-sum check cannot see either.
        mixers = [name for name in VARIANT_SUBMODULE_NAMES if name != "attn" and name in submodules]
        if mixers:
            return f"blocks.{layer} shares that add with a parallel {mixers[0]} mixer"
        aliases = getattr(block, "hook_aliases", None)
        target = aliases.get("hook_attn_out") if isinstance(aliases, dict) else None
        candidates = [target] if isinstance(target, str) else list(target or [])
        # A fallback chain resolves to its first target whose owner the block declares.
        for candidate in candidates:
            owner = candidate.split(".", 1)[0]
            if owner not in submodules:
                continue
            if owner != "attn":
                return f"blocks.{layer}.hook_attn_out reads {candidate}, after the head sum"
            break
    return None


def _check_attention_write_structure(model: Any) -> None:
    """Refuse blocks that declare a norm, scale or parallel mixer between head sum and residual."""
    detail = _attention_write_obstruction(model)
    if detail is not None:
        raise _non_additive_attention_error(detail)


def _check_kv_heads_match_query_heads(model: Any) -> None:
    """Refuse GQA/MQA attention layers, whose K/V reader inputs are shared across query heads."""
    cfg = model.cfg
    blocks = list(getattr(model, "blocks", ()))
    scalar_kv = getattr(cfg, "n_key_value_heads", None)
    per_layer_kv = getattr(cfg, "per_layer_num_key_value_heads", None) or []
    for layer in range(cfg.n_layers):
        submodules = getattr(blocks[layer], "submodules", None) if layer < len(blocks) else None
        # A hybrid's SSM or linear-attention block inherits the scalar count but has no K/V heads.
        if isinstance(submodules, dict) and "attn" not in submodules:
            continue
        kv = per_layer_kv[layer] if layer < len(per_layer_kv) else None
        if kv is None:
            kv = scalar_kv
        if kv is not None and kv != cfg.n_heads:
            raise ValueError(
                f"blocks.{layer} has n_key_value_heads={kv} for n_heads={cfg.n_heads} (GQA/MQA), "
                f"so its K/V inputs are not per query head: {_EDGE_UNSUPPORTED}"
            )


def _check_attention_write_is_head_sum(cache: GradientCache, n_layers: int) -> None:
    """Refuse a layer whose ``hook_attn_out`` is not its head-result sum plus a constant."""
    for layer in range(n_layers):
        per_head = cache.activations.get(f"blocks.{layer}.attn.hook_result")
        written = cache.activations.get(f"blocks.{layer}.hook_attn_out")
        if per_head is None or written is None:
            continue
        work_dtype = torch.promote_types(written.dtype, torch.float32)
        written = written.to(work_dtype)
        head_sum = per_head.sum(dim=2, dtype=work_dtype)
        if head_sum.shape != written.shape:
            raise _non_additive_attention_error(
                f"blocks.{layer}.hook_attn_out has shape {tuple(written.shape)}, the summed "
                f"attn.hook_result {tuple(head_sum.shape)}"
            )
        # The output bias belongs to no head and cancels in every clean-minus-corrupt delta,
        # so only the position-varying part of the remainder breaks additivity.
        remainder = written - head_sum
        remainder = remainder - remainder.mean(dim=(0, 1), keepdim=True)
        scale = written.norm().clamp_min(torch.finfo(work_dtype).tiny)
        relative = float(remainder.norm() / scale)
        if relative > _ATTN_WRITE_RTOL:
            raise _non_additive_attention_error(
                f"blocks.{layer}.hook_attn_out differs from the summed attn.hook_result by "
                f"{relative:.2g} of its norm"
            )


def _writer_hook_names(n_layers: int) -> list[str]:
    """Hook points holding each writer's residual contribution: embed, head result, MLP out."""
    names = ["hook_embed"]
    for layer in range(n_layers):
        names.append(f"blocks.{layer}.attn.hook_result")
        names.append(f"blocks.{layer}.hook_mlp_out")
    return names


def _reader_hook_names(n_layers: int) -> list[str]:
    """Hook points holding each reader's input; the logits reader's is listed separately."""
    names: list[str] = []
    for layer in range(n_layers):
        names.append(f"blocks.{layer}.attn.hook_q_input")
        names.append(f"blocks.{layer}.attn.hook_k_input")
        names.append(f"blocks.{layer}.attn.hook_v_input")
        names.append(f"blocks.{layer}.hook_mlp_in")
    return names


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
        ValueError: if any required hook point is absent from ``cache`` — the
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


def capture_writer_outputs(
    model: Any,
    tokens: torch.Tensor,
    names_filter: NamesFilter = None,
) -> dict[str, torch.Tensor]:
    """Capture writer contributions in one forward-only pass, with no gradient retained.

    Memory caveat: ``attn.hook_result`` is a per-head ``[batch, seq, n_heads, d_model]``
    tensor, so this does not scale to models with many heads or layers.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg.n_layers``,
            ``hook_dict``, and the ``hooks()`` context manager.
        tokens: Input token ids for a single forward pass.
        names_filter: Restricts which hook points are captured. ``None`` (the
            default) captures ``hook_embed``, ``attn.hook_result`` and
            ``hook_mlp_out``. Gated points (``attn.hook_result``, the split-QKV
            inputs, ``hook_mlp_in``) raise in ``add_hook`` unless their
            ``set_use_*`` flag is on; see :func:`_edge_hook_flags`.

    Returns:
        Detached activation tensors keyed by hook name, for every hook point
        matching ``names_filter``.

    Raises:
        ValueError: if ``names_filter`` matches no hook point.
    """
    if names_filter is None:
        names_filter = _writer_hook_names(int(model.cfg.n_layers))
    predicate = _as_predicate(names_filter)
    names = [name for name in model.hook_dict if predicate(name)]
    if not names:
        raise ValueError("names_filter matched no hook points")

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
    input. A ``logits`` reader takes its gradient at the final
    ``hook_resid_post`` and contracts over ``d_model`` at its position, the same
    shape path as an ``mlp_in`` reader.

    Two structural facts keep this off a per-edge recompute: a writer hook's
    clean-minus-corrupt delta is the same tensor for every edge that writer
    feeds, and every edge into one reader shares that reader's gradient vector.
    Each writer delta is therefore computed once, and each reader's incoming
    edges are scored with a single batched matrix-vector product over the
    stacked writer deltas -- reducing to the same per-edge scalars a
    ``float((delta_vec * grad_vec).sum())`` per edge would, reader gradient
    reused across all of that reader's edges.
    """
    scores: dict[tuple[Node, Node], float] = {}
    writer_deltas: dict[str, torch.Tensor] = {}

    def writer_delta_vec(writer: Node) -> torch.Tensor:
        name = _writer_hook_name(writer)
        delta = writer_deltas.get(name)
        if delta is None:
            delta = clean_cache.activations[name] - corrupt_cache.activations[name]
            writer_deltas[name] = delta
        if writer.kind == "attn_head_out":
            return delta[0, writer.position, writer.head]
        return delta[0, writer.position]

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
        if reader.kind in ("q_input", "k_input", "v_input"):
            grad_vec = grad[0, reader.position, reader.head]
        else:
            grad_vec = grad[0, reader.position]
        delta_matrix = torch.stack([writer_delta_vec(writer) for writer in writers])
        edge_values = delta_matrix @ grad_vec
        for writer, value in zip(writers, edge_values.tolist()):
            scores[(writer, reader)] = value
    return scores


def _aggregate_edge_scores_to_writer_nodes(
    edge_scores: dict[tuple[Node, Node], float],
) -> dict[Node, float]:
    """Sum each writer's outgoing edge scores into that writer's aggregate node score.

    A writer's aggregate is the sum of its effects along every edge it feeds.
    enumerate_edges gives every writer an edge to the terminal logits reader, so
    the aggregate includes the writer's direct skip-connection contribution to
    the metric -- the part of its residual-stream write that no intermediate
    component reads, only the final readout does -- and therefore equals the
    quantity a node-granularity sweep measures directly at the writer's own hook
    point. Every writer has at least that one outgoing edge, including the final
    layer's MLP output, whose only reader is the logits terminal, so no writer is
    missing from the aggregate.
    """
    totals: dict[Node, float] = {}
    for (writer, _reader), score in edge_scores.items():
        totals[writer] = totals.get(writer, 0.0) + score
    return totals


def _writer_contribution_slice(writer: Node, tensor: torch.Tensor) -> torch.Tensor:
    """The writer's ``d_model`` contribution at its position, and head if it has one."""
    if writer.kind == "attn_head_out":
        return tensor[0, writer.position, writer.head]
    return tensor[0, writer.position]


def _excluded_writers_by_reader(
    edges: Sequence[tuple[Node, Node]],
    circuit: Collection[tuple[Node, Node]],
) -> dict[Node, list[Node]]:
    """Group the out-of-circuit writers feeding each reader node, so each gets one hook."""
    circuit_set = set(circuit)
    excluded: dict[Node, list[Node]] = {}
    for writer, reader in edges:
        if (writer, reader) not in circuit_set:
            excluded.setdefault(reader, []).append(writer)
    return excluded


def _make_writer_capture_hook(name: str, sink: dict[str, torch.Tensor]) -> Callable[..., None]:
    """Record a writer's live contribution for the reader hooks that follow it.

    Detached, not cloned: it is only read, and the reader hook clones its own input.
    """

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
    """Rebuild one reader's input with its out-of-circuit writers replaced.

    Exact, not approximate: the pre-norm residual is a plain sum of writer contributions.
    """
    if reader.kind in ("q_input", "k_input", "v_input"):
        # Node.__post_init__ guarantees a head for the per-head reader kinds.
        index: tuple[int, ...] = (0, reader.position, cast(int, reader.head))
    else:
        index = (0, reader.position)

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
                rebuilt[index]
                - _writer_contribution_slice(writer, live)
                + _writer_contribution_slice(writer, replacement)
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
    """Return logits from one no-grad forward with every edge outside ``circuit`` ablated.

    The caller must have the writer and reader hook families enabled (see :func:`_edge_hook_flags`).
    """
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


def _validate_pair(
    clean: torch.Tensor, corrupt: torch.Tensor, *, operation: str, method: str
) -> None:
    """Require 2D clean/corrupt token tensors holding equal numbers of equal-length pairs."""
    if clean.ndim != 2 or corrupt.ndim != 2:
        raise ValueError(
            f"{operation} expects 2D [batch, seq] token tensors, got clean "
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
            f"{method} aligns activations position-by-position."
        )


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
        ``node_scores`` is the per-writer aggregate of those edge scores. The
        graph's terminal logits reader gives every writer an edge carrying its
        direct skip-connection contribution to the metric, so this aggregate
        equals the quantity a node-granularity sweep on the same model returns
        (including for the final layer's MLP output, whose only outgoing edge is
        the one to the logits reader).

    Raises:
        ValueError: if ``clean``/``corrupt`` are not 2D, hold a different number of
            pairs, a pair tokenizes to different lengths (activations must align
            position-by-position), or the model or one of its submodules is in
            training mode. At edge granularity, also if per-head attention outputs
            are normalized or scaled before the residual add (post-norm,
            scaled-residual or parallel-mixer blocks), where ``attn.hook_result`` is not
            a head's residual contribution, or if an attention layer's K/V head count
            differs from ``n_heads`` (GQA/MQA).
    """
    _validate_pair(clean, corrupt, operation="attribution_patch", method="Attribution patching")
    require_eval_mode(model, operation="attribution_patch()")

    batch = int(clean.shape[0])
    n_layers = int(model.cfg.n_layers)

    if config.granularity == "edge":
        _check_attention_write_structure(model)
        _check_kv_heads_match_query_heads(model)
        hook_names = _edge_hook_names(n_layers)
        edge_totals: dict[tuple[Node, Node], float] = {}

        # Scope the per-head hook-flag mutation to the caching loop so the
        # caller's flag state is restored on both the normal and the error path.
        with _edge_hook_flags(model):
            for index in range(batch):
                clean_cache = cache_activation_and_gradient(
                    model,
                    clean[index : index + 1],
                    metric_fn,
                    names_filter=hook_names + _attn_write_hook_names(n_layers),
                    compute_gradient=False,
                )
                _check_attention_write_is_head_sum(clean_cache, n_layers)
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
    """Deduplicate circuit pairs in first-occurrence order, dropping ``top_edges`` scores."""
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
    """The edge's class, keyed by what its reader consumes; Q and K share one (both feed scores)."""
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
    """Measure how much of the clean-to-corrupt metric gap a circuit recovers.

    Ablates every edge *outside* ``circuit`` (forward-only, exact rather than linearized) and
    reports the metric the model then produces. Feed a ranked edge set straight in from
    ``attribution_patch(..., config=EdgeAttributionConfig(granularity="edge")).top_edges(k=...)``.

    Under mean ablation recovery is still normalized to the clean/corrupt gap, so an empty
    circuit need not have zero recovery.

    Args:
        model: A ``TransformerBridge`` (or compatible) exposing ``cfg.n_layers``,
            ``hook_dict``, and ``hooks()``.
        clean: Clean token ids for one evaluated example, shape ``[1, seq]``.
        corrupt: Paired corrupt token ids, shape ``[1, seq]``.
        metric_fn: Maps single-example logits to a scalar.
        circuit: The edges to keep, as ``(writer, reader)`` pairs or the
            ``(writer, reader, score)`` triples ``top_edges`` returns. Every
            other edge in the graph is ablated. Duplicate entries count once.
        config: Ablation configuration. Defaults to replacing an ablated writer
            with the corrupt run's own contribution.
        mean_tokens: Required only for ``ablation="mean"``. Held-out ``int32``/``int64``
            calibration token ids, ``[n_calibration, seq]`` at the evaluated length and
            on the clean/corrupt device; writer activations are averaged per position
            and head. Rows equal to either endpoint are rejected; avoid semantic leakage too.

    Returns:
        A :class:`FaithfulnessResult`. The per-edge-class breakout costs one extra
        forward per retained class.

    Raises:
        ValueError: if ``clean``/``corrupt`` are not 2D, hold a different number
            of pairs, hold more than one pair, a pair tokenizes to different
            lengths, the model or a submodule is in training mode, a circuit edge
            is not in the graph, the clean and corrupt metrics are equal (so
            the recovered fraction is undefined), mean calibration is missing,
            empty, misaligned, on a different input device, has an unsupported
            dtype, contains an evaluated endpoint, or is supplied in corrupt mode;
            or per-head attention outputs are normalized or scaled before the
            residual add (post-norm, scaled-residual or parallel-mixer blocks), or an
            attention layer's K/V head count differs from ``n_heads`` (GQA/MQA).
    """
    _validate_pair(clean, corrupt, operation="faithfulness", method="Faithfulness")
    if clean.shape[0] != 1:
        raise ValueError(
            "faithfulness measures one clean/corrupt pair at a time, since the "
            f"metric reads a single example; got {clean.shape[0]} pairs. Loop over "
            "pairs and aggregate the recovered fractions yourself."
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
    _check_attention_write_structure(model)
    _check_kv_heads_match_query_heads(model)

    kept = _normalize_circuit(circuit)
    n_layers = int(model.cfg.n_layers)
    hook_names = _edge_hook_names(n_layers) + _attn_write_hook_names(n_layers)

    with torch.no_grad(), _edge_hook_flags(model):
        corrupt_cache = cache_activation_and_gradient(
            model, corrupt, metric_fn, names_filter=hook_names, compute_gradient=False
        )
        _check_attention_write_is_head_sum(corrupt_cache, n_layers)
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

        with torch.no_grad():
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
