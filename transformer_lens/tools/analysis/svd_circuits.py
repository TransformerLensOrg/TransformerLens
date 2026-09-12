"""Singular-vector decomposition of a single attention head's QK and OV maps.

An attention head is characterized by two low-rank linear maps: the query-key
map ``W_Q W_K^T`` that scores source positions, and the output-value map
``W_V W_O`` that writes the attended value back into the residual stream. This
tool takes the singular value decomposition of each map for one head and exposes
the singular values (how much each direction matters) together with the left and
right singular vectors that span the map's input and output spaces.

The decomposition is weight-space only: it reads ``W_Q``/``W_K``/``W_V``/``W_O``
and needs no forward pass, no activation cache, and no compatibility mode. Each
map is kept factored through
:class:`~transformer_lens.FactoredMatrix.FactoredMatrix`, so the
``d_model x d_model`` product is never materialized and the returned rank is
bounded by ``d_head``.

For a factored map ``A @ B`` (``A: [ldim, mdim]``, ``B: [mdim, rdim]``), the SVD's
``U`` columns live in ``A``'s input space (``ldim``) and ``V`` columns live in
``B``'s output space (``rdim``): feeding ``x = U[:, i]`` through the map gives
``x @ (A @ B) == S[i] * V[:, i]``, never the reverse. For OV (``A = W_V_h``,
``B = W_O_h``), ``U``'s columns are therefore the value-computation *input*
directions this head reads from the residual stream, and ``V``'s columns are the
*output* directions it writes back into the residual stream - the ones to
project through ``W_U`` for a vocab or logit readout. For QK (``A = W_Q_h``,
``B = W_K_h.transpose(-1, -2)``), both ``U`` (destination/query-read) and ``V``
(source/key-read) are read directions; QK only ever produces a scalar attention
score, so neither is a write direction. The historical ``.Vh`` alias returns the
same tensor as ``.V`` and is never used here.

Adjacent singular values closer than a relative gap ``eps`` leave their singular
directions defined only up to a rotation, so the result carries a per-direction
degeneracy report. Directions are grouped into contiguous blocks that end only
at a gap of at least ``eps``; callers attribute such a block as a subspace
instead of trusting a single, rotation-dependent direction inside it.

A singular value near zero relative to the top of the spectrum is null rather
than near-equal: its singular vector is an arbitrary null-space direction, not a
rotation of a comparable neighbour. Null directions are flagged under their own
tolerance, ``null_rtol``, keyed to the spectrum's top value the way
:func:`torch.linalg.matrix_rank` keys its default tolerance.

Example::

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis.svd_circuits import decompose_head

    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    decomposition = decompose_head(model, layer=0, head=0)
    ov = decomposition.OV
    for row in ov.rank_report:
        print(f"direction {row.idx}: sigma={row.sigma:.3f} ratio={row.sigma_ratio:.3f}")
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Float

from transformer_lens.FactoredMatrix import FactoredMatrix

# Which of a head's two maps to decompose: the query-key map or the output-value map.
Which = Literal["QK", "OV"]

_VALID_WHICH: Tuple[Which, ...] = ("QK", "OV")

# Relative-gap threshold: neighbouring singular values closer than this (in
# relative terms) are treated as one rotation-ambiguous block.
_DEFAULT_EPS = 1e-2

# Absolute floor so relative-gap and ratio computations never divide by ~0.
_SIGMA_FLOOR = 1e-12


class DegenerateDirectionError(ValueError):
    """Raised when per-direction attribution is requested for a degenerate direction.

    Subclasses ``ValueError`` so callers that already ``except ValueError`` keep
    working, mirroring how the other analysis tools raise ``ValueError`` for
    their input guards.
    """


@dataclass
class RankReportRow:
    """One singular direction's summary, aligned with column ``idx`` of ``U``/``V``.

    Attributes:
        idx: Position of the direction, matching the column index in ``U`` and ``V``.
        sigma: The singular value for this direction.
        sigma_ratio: ``sigma`` normalized by the largest singular value, in ``[0, 1]``.
        is_degenerate: True when this direction is not attributable on its own: it
            shares a block with a neighbour (defined only up to a rotation within
            that block) or it is numerically null.
        is_null: True when ``sigma_ratio`` falls below ``null_rtol``, so the singular
            vector is an arbitrary direction from the map's null space.
        block_id: Index of the contiguous block this direction belongs to.
    """

    idx: int
    sigma: float
    sigma_ratio: float
    is_degenerate: bool
    is_null: bool
    block_id: int


@dataclass
class HeadSVD:
    """Factored SVD of one head map (``"QK"`` or ``"OV"``) with a degeneracy report.

    Attributes:
        which: Which map this decomposes, ``"QK"`` or ``"OV"``.
        layer: Layer of the decomposed head.
        head: Head index within the layer.
        U: Left singular vectors, ``[d_model, rank]``: column i is the map's input
            direction i (for OV, the residual-stream direction this head's value
            computation reads from; for QK, the destination/query-read direction).
        S: Singular values, ``[rank]``, sorted descending.
        V: Right singular vectors, ``[d_model, rank]``: column i is the map's output
            direction i for OV (the residual-stream direction this head writes into,
            the one to project through ``W_U``), or the source/key-read direction for
            QK (QK produces no write direction). The reconstruction is
            ``U @ S.diag() @ V.transpose(-2, -1)``.
        rank_report: Per-direction :class:`RankReportRow` list, aligned with the
            columns of ``U``/``V``.
        eps: Relative gap below which adjacent directions share a block; every block
            boundary sits at a gap of at least ``eps``.
        null_rtol: Relative-to-top-singular-value tolerance below which a direction
            is numerically null.
    """

    which: Which
    layer: int
    head: int
    U: Float[torch.Tensor, "d_model rank"]
    S: Float[torch.Tensor, "rank"]
    V: Float[torch.Tensor, "d_model rank"]
    rank_report: List[RankReportRow]
    eps: float
    null_rtol: float

    def is_degenerate(self, i: int) -> bool:
        """Whether direction ``i`` is refused: rotation-ambiguous inside a block, or null."""
        return self.rank_report[i].is_degenerate

    def block_of(self, i: int) -> List[int]:
        """Return every direction index sharing direction ``i``'s degeneracy block.

        The result is a singleton ``[i]`` for an isolated direction and the full
        run for a degenerate one.
        """
        block_id = self.rank_report[i].block_id
        return [row.idx for row in self.rank_report if row.block_id == block_id]

    def degenerate_blocks(self) -> List[List[int]]:
        """Return every degenerate block's indices: the subspaces to attribute whole or skip."""
        blocks: List[List[int]] = []
        current: List[int] = []
        current_block_id: Optional[int] = None
        degenerate = False
        # block_ids are contiguous and ascending, so one scan recovers the blocks.
        for row in self.rank_report:
            if row.block_id != current_block_id:
                if degenerate:
                    blocks.append(current)
                current = []
                degenerate = False
                current_block_id = row.block_id
            current.append(row.idx)
            degenerate = degenerate or row.is_degenerate
        if degenerate:
            blocks.append(current)
        return blocks

    def require_isolated(self, i: int) -> None:
        """Raise :class:`DegenerateDirectionError` unless direction ``i`` is attributable alone.

        The message names the cause, since a rotation-ambiguous block is still a
        subspace worth attributing while a null block carries no signal.
        """
        row = self.rank_report[i]
        if not row.is_degenerate:
            return
        where = f"Direction {i} of the {self.which} SVD of head L{self.layer}H{self.head}"
        block = self.block_of(i)
        if row.is_null:
            raise DegenerateDirectionError(
                f"{where} is numerically null (sigma_ratio {row.sigma_ratio:.2e} < "
                f"null_rtol {self.null_rtol:.2e}), so its singular vector is an arbitrary "
                f"null-space direction. Attribute the null block {block} as a subspace, "
                f"or skip it."
            )
        raise DegenerateDirectionError(
            f"{where} lies in block {block}, whose members are not separated by a "
            f"relative gap of eps={self.eps:g} and so are defined only up to a rotation. "
            f"Attribute the block as a subspace instead of the single direction."
        )


@dataclass
class HeadDecomposition:
    """Container returned by :func:`decompose_head`.

    ``QK`` and ``OV`` hold the :class:`HeadSVD` for each requested map, or
    ``None`` when that map was not requested.
    """

    layer: int
    head: int
    QK: Optional[HeadSVD] = None
    OV: Optional[HeadSVD] = None


def _read_weight(weight: torch.Tensor) -> torch.Tensor:
    """Detach one per-head weight and put it in SVD precision.

    Detached so the returned factors carry no autograd graph into the model. fp16/bf16
    are promoted because reduced-precision SVD is unstable; float64 is kept so the
    default ``null_rtol`` matches the precision actually decomposed.
    """
    weight = weight.detach()
    return weight if weight.dtype == torch.float64 else weight.float()


def _head_weights(
    model, layer: int, head: int
) -> Tuple[
    Float[torch.Tensor, "d_model d_head"],
    Float[torch.Tensor, "d_model d_head"],
    Float[torch.Tensor, "d_model d_head"],
    Float[torch.Tensor, "d_head d_model"],
]:
    """Return ``(W_Q_h, W_K_h, W_V_h, W_O_h)`` for one head, detached, in SVD precision.

    Reads the single block's per-head weights rather than the full-model
    ``W_Q``/``W_K``/``W_V``/``W_O`` stacks, so only one layer is materialized. On
    grouped-query attention ``W_K``/``W_V`` carry one row per key-value head, so the
    query head is mapped to its key-value head (query head ``h`` reads kv head
    ``h // (n_heads // n_kv_heads)``); a no-op for multi-head attention, where the
    head counts already match.
    """
    attn = model.blocks[layer].attn
    n_kv_heads = attn.W_K.shape[0]
    kv_head = head // (model.cfg.n_heads // n_kv_heads)
    W_Q_h = _read_weight(attn.W_Q[head])  # [d_model, d_head]
    W_K_h = _read_weight(attn.W_K[kv_head])  # [d_model, d_head]
    W_V_h = _read_weight(attn.W_V[kv_head])  # [d_model, d_head]
    W_O_h = _read_weight(attn.W_O[head])  # [d_head, d_model]
    return W_Q_h, W_K_h, W_V_h, W_O_h


def _degeneracy_blocks(
    S: Float[torch.Tensor, "rank"], eps: float, null_rtol: float
) -> List[List[int]]:
    """Group singular directions into contiguous blocks separated by relative gaps of ``eps``.

    ``S`` is sorted descending. Direction ``i`` joins the open block when
    ``1 - S[i]/S[i-1] < eps``, or when it and its predecessor are both null relative to
    the top value. Only the gap to the previous direction counts: a block may span far
    more than ``eps`` end to end, but every boundary is an ``eps`` gap, and that
    separation from the rest of the spectrum is what makes a block stable under
    perturbation; no smaller contiguous group inside it is. ``_SIGMA_FLOOR`` keeps the
    ratios finite when a divisor is ~0.
    """
    n = int(S.shape[0])
    if n == 0:
        return []
    values = [float(x) for x in S.tolist()]
    top = max(values[0], _SIGMA_FLOOR)
    blocks: List[List[int]] = []
    current = [0]
    for i in range(1, n):
        prev = max(values[i - 1], _SIGMA_FLOOR)
        near_equal = (1.0 - values[i] / prev) < eps
        null_run = (values[i] / top) < null_rtol and (values[i - 1] / top) < null_rtol
        if near_equal or null_run:
            current.append(i)
        else:
            blocks.append(current)
            current = [i]
    blocks.append(current)
    return blocks


def _build_rank_report(
    S: Float[torch.Tensor, "rank"], blocks: List[List[int]], null_rtol: float
) -> List[RankReportRow]:
    """Summarize each singular direction and tag its degeneracy block.

    ``blocks`` must partition ``range(len(S))``. Nullness flags a direction on its own:
    a lone null singular vector is arbitrary even though nothing groups with it.
    """
    values = [float(x) for x in S.tolist()]
    top = max(values) if values else 0.0
    denominator = top if top > _SIGMA_FLOOR else _SIGMA_FLOOR
    rows: List[Optional[RankReportRow]] = [None] * len(values)
    for block_id, block in enumerate(blocks):
        shared = len(block) > 1
        for idx in block:
            sigma_ratio = values[idx] / denominator
            is_null = sigma_ratio < null_rtol
            rows[idx] = RankReportRow(
                idx=idx,
                sigma=values[idx],
                sigma_ratio=sigma_ratio,
                is_degenerate=shared or is_null,
                is_null=is_null,
                block_id=block_id,
            )
    return [row for row in rows if row is not None]


def _factored_head_svd(
    A: Float[torch.Tensor, "d_model d_head"],
    B: Float[torch.Tensor, "d_head d_model"],
    *,
    which: Which,
    layer: int,
    head: int,
    eps: float,
    null_rtol: Optional[float] = None,
) -> HeadSVD:
    """Decompose the factored map ``A @ B`` for one head into a :class:`HeadSVD`.

    For OV pass ``A = W_V_h`` and ``B = W_O_h``; for QK pass ``A = W_Q_h`` and
    ``B = W_K_h.transpose(-1, -2)``. The map stays factored through
    :class:`FactoredMatrix`, so the ``d_model x d_model`` product is never
    materialized and the rank is bounded by ``d_head``.

    ``null_rtol`` of ``None`` resolves to ``d_model * torch.finfo(S.dtype).eps``,
    the same relative tolerance :func:`torch.linalg.matrix_rank` uses by default
    for a square ``d_model x d_model`` map, so the null cutoff tracks the
    decomposition's own numerical rank rather than a hand-tuned constant.
    """
    U, S, V = FactoredMatrix(A, B).svd()
    d_model = U.shape[0]
    resolved_null_rtol = null_rtol if null_rtol is not None else d_model * torch.finfo(S.dtype).eps
    blocks = _degeneracy_blocks(S, eps, null_rtol=resolved_null_rtol)
    rank_report = _build_rank_report(S, blocks, null_rtol=resolved_null_rtol)
    return HeadSVD(
        which=which,
        layer=layer,
        head=head,
        U=U,
        S=S,
        V=V,
        rank_report=rank_report,
        eps=eps,
        null_rtol=resolved_null_rtol,
    )


def decompose_head(
    model,
    layer: int,
    head: int,
    *,
    which: Sequence[str] = ("QK", "OV"),
    eps: float = _DEFAULT_EPS,
    null_rtol: Optional[float] = None,
) -> HeadDecomposition:
    """Decompose a head's QK (``W_Q W_K^T``) and/or OV (``W_V W_O``) maps via SVD.

    Weight-space only: this reads the head's per-block weights via the bridge's
    ``model.blocks[layer].attn`` accessors and needs no forward pass and no
    compatibility mode. The returned factors are detached from the model.

    Args:
        model: A ``TransformerBridge``.
        layer: Layer of the head to decompose.
        head: Head index within the layer.
        which: Which maps to decompose, a non-empty sequence drawn from
            ``("QK", "OV")``. A bare string is rejected rather than iterated.
        eps: Relative gap at which a block of adjacent directions ends; directions
            closer than this share a block.
        null_rtol: Relative-to-top-singular-value tolerance below which a direction
            counts as numerically null. Defaults to ``None``, which resolves to
            ``d_model * torch.finfo(S.dtype).eps`` per map, matching
            :func:`torch.linalg.matrix_rank`'s default tolerance.

    Returns:
        A :class:`HeadDecomposition` whose ``QK``/``OV`` fields hold a
        :class:`HeadSVD` for each requested map.

    Raises:
        ValueError: If ``layer`` or ``head`` is out of range, or ``which`` is
            empty, a bare string, or contains an unknown entry.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer must be in [0, {n_layers}), got {layer!r}")
    if not 0 <= head < n_heads:
        raise ValueError(f"head must be in [0, {n_heads}), got {head!r}")
    if isinstance(which, str):
        raise ValueError(f"which must be a sequence of map names such as ('QK',), got {which!r}")
    requested = tuple(which)
    if not requested:
        raise ValueError("which must request at least one of 'QK' or 'OV'")
    invalid = [entry for entry in requested if entry not in _VALID_WHICH]
    if invalid:
        raise ValueError(f"which entries must be in {_VALID_WHICH}, got {invalid!r}")

    W_Q_h, W_K_h, W_V_h, W_O_h = _head_weights(model, layer, head)
    qk = None
    ov = None
    if "QK" in requested:
        qk = _factored_head_svd(
            W_Q_h,
            W_K_h.transpose(-1, -2),
            which="QK",
            layer=layer,
            head=head,
            eps=eps,
            null_rtol=null_rtol,
        )
    if "OV" in requested:
        ov = _factored_head_svd(
            W_V_h, W_O_h, which="OV", layer=layer, head=head, eps=eps, null_rtol=null_rtol
        )
    return HeadDecomposition(layer=layer, head=head, QK=qk, OV=ov)


def _validate_bridge_compatibility(model) -> None:
    """Reject a ``TransformerBridge`` whose ``W_U`` would give a silently wrong projection.

    ``HookedTransformer`` always has the final LayerNorm folded into ``W_U``, so this
    only fires for ``TransformerBridge``. Mirrors the compatibility-mode check other
    unembedding-touching analysis tools already run, without any hybrid-architecture
    restriction: projecting a rank-1 OV direction through ``W_U`` does not depend on
    the block-layout assumptions that check exists for elsewhere.
    """
    # Lazy import - keeps the module importable without the bridge as a hard dependency.
    from transformer_lens.model_bridge import TransformerBridge

    if not isinstance(model, TransformerBridge):
        return
    if not getattr(model, "compatibility_mode", False):
        raise ValueError(
            "Projecting an OV direction through W_U on a TransformerBridge requires "
            "compatibility mode, so that LayerNorm weights are folded into W_U. Call "
            "`model.enable_compatibility_mode()` after loading the bridge, then retry."
        )


def vocab_readout(
    model, head_svd: HeadSVD, *, k: int = 10
) -> Float[torch.Tensor, "d_vocab k"]:
    """Project the top-k OV output directions through the unembedding.

    Requires ``head_svd.which == "OV"``: QK produces no write direction to project
    (see the module docstring). On a ``TransformerBridge``, compatibility mode must
    be enabled so ``W_U`` carries the folded final LayerNorm weights;
    ``HookedTransformer`` always has this folding applied.

    Does not call ``head_svd.require_isolated``: a degenerate direction's vocab
    readout is still a well-defined projection, unlike a per-direction causal claim,
    so it is not gated here. The contract that no direction is reported without a
    passing causal patch is enforced by :func:`patch_along_directions`.

    Args:
        model: A ``TransformerBridge`` (with compatibility mode enabled) or a
            ``HookedTransformer``; only its ``W_U`` is read.
        head_svd: An OV :class:`HeadSVD` from :func:`decompose_head`.
        k: Number of top singular directions to project.

    Returns:
        ``W_U.T @ head_svd.V[:, :k]``, shape ``[d_vocab, k]``: column i is
        direction i's projection through the unembedding.

    Raises:
        ValueError: If ``head_svd.which != "OV"``, if ``k`` is not in
            ``(0, rank]``, or if ``model`` is a ``TransformerBridge`` without
            compatibility mode enabled.
    """
    if head_svd.which != "OV":
        raise ValueError(f"vocab_readout requires an OV HeadSVD, got which={head_svd.which!r}")
    rank = head_svd.V.shape[1]
    if not 0 < k <= rank:
        raise ValueError(f"k must be in (0, {rank}], got {k!r}")
    _validate_bridge_compatibility(model)
    return model.W_U.T @ head_svd.V[:, :k].float()


@dataclass
class LogitSignature:
    """Rank-1-reconstruction logit effect for one OV direction, per requested token.

    Attributes:
        direction: Which ``HeadSVD`` column this reconstructs.
        values: Signed logit contribution, aligned with the requested tokens.
    """

    direction: int
    values: Float[torch.Tensor, "token"]


def logit_signature(
    model,
    head_svd: HeadSVD,
    direction: int,
    tokens: Union[int, Sequence[int], torch.Tensor],
) -> LogitSignature:
    """Signed logit effect of one OV direction's rank-1 reconstruction on the given tokens.

    Pure weight-space computation: reconstructs the head's OV output along a single
    singular direction (``S[direction] * V[:, direction]``, never ``U`` - see the
    module docstring) and projects it through ``W_U`` restricted to ``tokens``. Runs
    no forward pass and builds no cache.

    Args:
        model: A ``TransformerBridge`` (with compatibility mode enabled) or a
            ``HookedTransformer``; only its ``W_U`` is read.
        head_svd: An OV :class:`HeadSVD` from :func:`decompose_head`.
        direction: Column index of the singular direction to reconstruct.
        tokens: Token id(s) to read the logit effect for.

    Returns:
        A :class:`LogitSignature` with one value per requested token.

    Raises:
        ValueError: If ``head_svd.which != "OV"``, or if ``model`` is a
            ``TransformerBridge`` without compatibility mode enabled.
        DegenerateDirectionError: If ``direction`` is not attributable alone (see
            :meth:`HeadSVD.require_isolated`).
    """
    if head_svd.which != "OV":
        raise ValueError(f"logit_signature requires an OV HeadSVD, got which={head_svd.which!r}")
    head_svd.require_isolated(direction)
    _validate_bridge_compatibility(model)
    token_ids = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
    reconstruction = (head_svd.S[direction] * head_svd.V[:, direction]).float()
    values = reconstruction @ model.W_U[:, token_ids]
    return LogitSignature(direction=direction, values=values)
