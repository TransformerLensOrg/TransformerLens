"""Singular-vector decomposition of a single attention head's QK and OV maps.

An attention head is characterized by two low-rank linear maps: the query-key
map ``W_Q W_K^T`` that scores source positions, and the output-value map
``W_V W_O`` that writes the attended value back into the residual stream. This
tool takes the singular value decomposition of each map for one head and exposes
the singular values (how much each direction matters) together with the left and
right singular vectors (the output and input directions they act on).

The decomposition is weight-space only: it reads ``W_Q``/``W_K``/``W_V``/``W_O``
and needs no forward pass, no activation cache, and no compatibility mode. Each
map is kept factored through
:class:`~transformer_lens.FactoredMatrix.FactoredMatrix`, so the
``d_model x d_model`` product is never materialized and the returned rank is
bounded by ``d_head``.

Right singular vectors are read from :attr:`~transformer_lens.FactoredMatrix.FactoredMatrix.V`
(its columns are the right singular vectors). The historical ``.Vh`` alias is
deprecated and returns the same tensor, so it is never used here.

Near-equal singular values leave their singular directions defined only up to a
rotation, so the result carries a per-direction degeneracy report. Callers can
use it to attribute an ambiguous block as a subspace instead of trusting a
single, rotation-dependent direction.

Example::

    from transformer_lens import HookedTransformer
    from transformer_lens.tools.analysis.svd_circuits import decompose_head

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    decomposition = decompose_head(model, layer=0, head=0)
    ov = decomposition.OV
    for row in ov.rank_report:
        print(f"direction {row.idx}: sigma={row.sigma:.3f} ratio={row.sigma_ratio:.3f}")
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

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
    """Raised when per-direction attribution is requested inside a degenerate block.

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
        is_degenerate: True when this direction shares its block with a neighbour,
            so it is defined only up to a rotation within that block.
        block_id: Index of the contiguous block this direction belongs to.
    """

    idx: int
    sigma: float
    sigma_ratio: float
    is_degenerate: bool
    block_id: int


@dataclass
class HeadSVD:
    """Factored SVD of one head map (``"QK"`` or ``"OV"``) with a degeneracy report.

    Attributes:
        which: Which map this decomposes, ``"QK"`` or ``"OV"``.
        layer: Layer of the decomposed head.
        head: Head index within the layer.
        U: Left singular vectors, ``[d_model, rank]`` (column i is output direction i).
        S: Singular values, ``[rank]``, sorted descending.
        V: Right singular vectors, ``[d_model, rank]`` (column i is input direction i).
            The reconstruction is ``U @ S.diag() @ V.transpose(-2, -1)``.
        rank_report: Per-direction :class:`RankReportRow` list, aligned with the
            columns of ``U``/``V``.
        eps: Relative-gap threshold used to group the degeneracy blocks.
    """

    which: Which
    layer: int
    head: int
    U: Float[torch.Tensor, "d_model rank"]
    S: Float[torch.Tensor, "rank"]
    V: Float[torch.Tensor, "d_model rank"]
    rank_report: List[RankReportRow]
    eps: float

    def is_degenerate(self, i: int) -> bool:
        """Return whether direction ``i`` shares its block with another direction.

        A degenerate direction is defined only up to a rotation within its block,
        so per-direction attribution against it is not meaningful.
        """
        return self.rank_report[i].is_degenerate

    def block_of(self, i: int) -> List[int]:
        """Return every direction index sharing direction ``i``'s degeneracy block.

        The result is a singleton ``[i]`` for an isolated direction and the full
        run for a degenerate one.
        """
        block_id = self.rank_report[i].block_id
        return [row.idx for row in self.rank_report if row.block_id == block_id]

    def degenerate_blocks(self) -> List[List[int]]:
        """Return the index groups for blocks holding more than one direction.

        Rows carry contiguous, ascending ``block_id`` values, so consecutive rows
        with a shared id form one block. Isolated directions are omitted, leaving
        only the rotation-ambiguous subspaces a caller must attribute as a whole.
        """
        blocks: List[List[int]] = []
        current: List[int] = []
        current_block_id: Optional[int] = None
        for row in self.rank_report:
            if row.block_id != current_block_id:
                if len(current) > 1:
                    blocks.append(current)
                current = []
                current_block_id = row.block_id
            current.append(row.idx)
        if len(current) > 1:
            blocks.append(current)
        return blocks

    def require_isolated(self, i: int) -> None:
        """Raise :class:`DegenerateDirectionError` if direction ``i`` is not isolated.

        Callers project or attribute a single singular direction only after this
        passes; inside a degenerate block the direction is rotation-ambiguous and
        the block must be attributed as a subspace instead.
        """
        if self.is_degenerate(i):
            block = self.block_of(i)
            raise DegenerateDirectionError(
                f"Direction {i} of the {self.which} SVD of head L{self.layer}H{self.head} "
                f"lies in a degenerate block {block} (near-equal singular values, "
                f"rotation-ambiguous). Attribute the block as a subspace instead of the "
                f"single direction."
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


def _head_weights(
    model, layer: int, head: int
) -> Tuple[
    Float[torch.Tensor, "d_model d_head"],
    Float[torch.Tensor, "d_model d_head"],
    Float[torch.Tensor, "d_model d_head"],
    Float[torch.Tensor, "d_head d_model"],
]:
    """Return ``(W_Q_h, W_K_h, W_V_h, W_O_h)`` for one head as float tensors.

    Reads the single block's per-head weights rather than the full-model
    ``W_Q``/``W_K``/``W_V``/``W_O`` stacks, so only one layer is materialized. On
    grouped-query attention ``W_K``/``W_V`` carry one row per key-value head, so the
    query head is mapped to its key-value head (query head ``h`` reads kv head
    ``h // (n_heads // n_kv_heads)``); a no-op for multi-head attention, where the
    head counts already match. The upcast to ``float`` promotes fp16/bf16 weights
    before the SVD, where reduced precision is a known source of instability.
    """
    attn = model.blocks[layer].attn
    W_Q_h = attn.W_Q[head].float()  # [d_model, d_head]
    W_O_h = attn.W_O[head].float()  # [d_head, d_model]
    n_kv_heads = attn.W_K.shape[0]
    kv_head = head // (model.cfg.n_heads // n_kv_heads)
    W_K_h = attn.W_K[kv_head].float()  # [d_model, d_head]
    W_V_h = attn.W_V[kv_head].float()  # [d_model, d_head]
    return W_Q_h, W_K_h, W_V_h, W_O_h


def _degeneracy_blocks(S: Float[torch.Tensor, "rank"], eps: float) -> List[List[int]]:
    """Group singular directions into contiguous blocks by their relative gap.

    ``S`` holds singular values sorted in descending order. Direction ``i`` joins the
    open block when it is within relative gap ``eps`` of both the previous direction
    and the block's anchor (its first, largest member), or when both directions sit in
    a near-zero (null) run relative to the top singular value. The anchor constraint
    bounds how far a block can spread: without it, a spectrum decaying by just under
    ``eps`` at every step would chain every direction into one block whose extremes
    differ by far more than ``eps``, purely because each gap-to-previous is individually
    small. A block of more than one direction is degenerate: its directions are defined
    only up to a rotation within the block. ``_SIGMA_FLOOR`` keeps the ratios finite when
    a divisor is ~0.
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
        anchor = max(values[current[0]], _SIGMA_FLOOR)
        near_equal = (1.0 - values[i] / prev) < eps and (1.0 - values[i] / anchor) < eps
        null_run = (values[i] / top) < eps and (values[i - 1] / top) < eps
        if near_equal or null_run:
            current.append(i)
        else:
            blocks.append(current)
            current = [i]
    blocks.append(current)
    return blocks


def _build_rank_report(
    S: Float[torch.Tensor, "rank"], blocks: List[List[int]]
) -> List[RankReportRow]:
    """Summarize each singular direction and tag the degeneracy block it belongs to.

    ``sigma_ratio`` normalizes each singular value by the largest one, so the top
    direction has ratio 1.0. ``blocks`` must partition ``range(len(S))``; each
    direction is degenerate when its block holds more than one direction.
    """
    values = [float(x) for x in S.tolist()]
    top = max(values) if values else 0.0
    denominator = top if top > _SIGMA_FLOOR else _SIGMA_FLOOR
    rows: List[Optional[RankReportRow]] = [None] * len(values)
    for block_id, block in enumerate(blocks):
        degenerate = len(block) > 1
        for idx in block:
            rows[idx] = RankReportRow(
                idx=idx,
                sigma=values[idx],
                sigma_ratio=values[idx] / denominator,
                is_degenerate=degenerate,
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
) -> HeadSVD:
    """Decompose the factored map ``A @ B`` for one head into a :class:`HeadSVD`.

    For OV pass ``A = W_V_h`` and ``B = W_O_h``; for QK pass ``A = W_Q_h`` and
    ``B = W_K_h.transpose(-1, -2)``. The map stays factored through
    :class:`FactoredMatrix`, so the ``d_model x d_model`` product is never
    materialized and the rank is bounded by ``d_head``.
    """
    U, S, V = FactoredMatrix(A, B).svd()
    blocks = _degeneracy_blocks(S, eps)
    rank_report = _build_rank_report(S, blocks)
    return HeadSVD(
        which=which,
        layer=layer,
        head=head,
        U=U,
        S=S,
        V=V,
        rank_report=rank_report,
        eps=eps,
    )


def decompose_head(
    model,
    layer: int,
    head: int,
    *,
    which: Sequence[Which] = ("QK", "OV"),
    eps: float = _DEFAULT_EPS,
) -> HeadDecomposition:
    """Decompose a head's QK (``W_Q W_K^T``) and/or OV (``W_V W_O``) maps via SVD.

    Weight-space only: this reads the head's weights and needs no forward pass and
    no compatibility mode. Works with both ``HookedTransformer`` and
    ``TransformerBridge`` because they share the ``W_Q``/``W_K``/``W_V``/``W_O``
    layout and ``cfg``.

    Args:
        model: A ``HookedTransformer`` or ``TransformerBridge``.
        layer: Layer of the head to decompose.
        head: Head index within the layer.
        which: Which maps to decompose, any subset of ``("QK", "OV")``.
        eps: Relative-gap threshold for grouping degenerate singular directions.

    Returns:
        A :class:`HeadDecomposition` whose ``QK``/``OV`` fields hold a
        :class:`HeadSVD` for each requested map.

    Raises:
        ValueError: If ``layer`` or ``head`` is out of range, or ``which`` is
            empty or contains an unknown entry.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer must be in [0, {n_layers}), got {layer!r}")
    if not 0 <= head < n_heads:
        raise ValueError(f"head must be in [0, {n_heads}), got {head!r}")
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
            W_Q_h, W_K_h.transpose(-1, -2), which="QK", layer=layer, head=head, eps=eps
        )
    if "OV" in requested:
        ov = _factored_head_svd(W_V_h, W_O_h, which="OV", layer=layer, head=head, eps=eps)
    return HeadDecomposition(layer=layer, head=head, QK=qk, OV=ov)
