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
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

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

# Default seed for the causal gate's random baseline, so a bare patch_along_directions
# call is reproducible instead of drawing from the global RNG. Matches estimate_occupancy,
# which seeds its random controls by default.
_DEFAULT_BASELINE_SEED = 0

# Number of random in-span control subspaces the gate averages its baseline delta over,
# so one lucky or unlucky draw does not decide the gate.
_DEFAULT_N_BASELINE = 8


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
        compatibility_mode: The model's compatibility-mode state at the time this was
            decomposed. ``enable_compatibility_mode`` folds ``ln1`` into ``W_V`` and
            centres ``W_O``/``W_U``, so a decomposition describes the OV map only under
            the state it was built in; the readout and patch consumers refuse a
            decomposition whose state no longer matches the model.
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
    compatibility_mode: bool = False

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
    compatibility_mode: bool = False,
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

    ``compatibility_mode`` is the model's compatibility-mode state, recorded on the
    result so the consumers can refuse a decomposition taken under a different state.
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
        compatibility_mode=compatibility_mode,
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
    compatibility mode. The returned factors are detached from the model. The model's
    compatibility-mode state is recorded on each returned :class:`HeadSVD` so the readout
    and patch consumers can refuse a decomposition taken under a different state.

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
    compatibility_mode = getattr(model, "compatibility_mode", False)
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
            compatibility_mode=compatibility_mode,
        )
    if "OV" in requested:
        ov = _factored_head_svd(
            W_V_h,
            W_O_h,
            which="OV",
            layer=layer,
            head=head,
            eps=eps,
            null_rtol=null_rtol,
            compatibility_mode=compatibility_mode,
        )
    return HeadDecomposition(layer=layer, head=head, QK=qk, OV=ov)


def _validate_bridge_compatibility(model) -> None:
    """Reject a ``TransformerBridge`` whose ``W_U`` would give a silently wrong projection.

    Projecting an OV direction through ``W_U`` requires the final LayerNorm folded
    into ``W_U``. This mirrors :func:`direct_logit_attribution`, the other
    unembedding-touching analysis tool, but the check here is stricter: it reads the
    bridge's recorded weight-processing state rather than the ``compatibility_mode``
    flag. ``enable_compatibility_mode`` sets that flag before it processes weights and
    skips processing entirely under ``no_processing``, so the flag can be ``True`` while
    ``W_U`` is still unfolded. The fold only happens once the bridge processes its
    weights with ``fold_ln`` enabled, which is what ``_weights_processed`` and the
    adapter's ``_fold_ln_requested`` record. No hybrid-architecture restriction applies:
    projecting a rank-1 OV direction through ``W_U`` does not depend on the block-layout
    assumptions that check exists for elsewhere.
    """
    # Lazy import - keeps the module importable without the bridge as a hard dependency.
    from transformer_lens.model_bridge import TransformerBridge

    if not isinstance(model, TransformerBridge):
        return
    processed = getattr(model, "_weights_processed", False)
    fold_ln_requested = getattr(getattr(model, "adapter", None), "_fold_ln_requested", True)
    if not (processed and fold_ln_requested):
        raise ValueError(
            "Projecting an OV direction through W_U on a TransformerBridge requires the "
            "final LayerNorm folded into W_U, which only happens once the bridge has "
            "processed its weights with fold_ln enabled. Call "
            "`model.enable_compatibility_mode()` (its default folds LayerNorm) after "
            "loading the bridge, then retry."
        )


def _validate_decomposition_matches_model(model, head_svd: HeadSVD) -> None:
    """Refuse a decomposition built under a different compatibility-mode state than the model.

    ``enable_compatibility_mode`` folds ``ln1`` into ``W_V`` and centres ``W_O``/``W_U``,
    so a :class:`HeadSVD` decomposed before the call describes a different OV map than the
    model now computes: the cached ``U``/``S``/``V`` are stale, and the returned readout or
    patch would be silently wrong rather than raise a shape error. This guard is orthogonal
    to :func:`_validate_bridge_compatibility` (which checks that the bridge actually folded
    LayerNorm into ``W_U`` via its recorded processing state): here the model may be in
    either state, only mismatched from the decomposition's.
    """
    current = getattr(model, "compatibility_mode", False)
    if current != head_svd.compatibility_mode:
        raise ValueError(
            f"This HeadSVD was decomposed with compatibility_mode="
            f"{head_svd.compatibility_mode} but the model now has compatibility_mode="
            f"{current}; the cached singular vectors describe a different OV map. "
            f"Re-run decompose_head under the current state, then retry."
        )


def vocab_readout(model, head_svd: HeadSVD, *, k: int = 10) -> Float[torch.Tensor, "d_vocab k"]:
    """Project the top-k OV output directions through the unembedding.

    Requires ``head_svd.which == "OV"``: QK produces no write direction to project
    (see the module docstring). On a ``TransformerBridge``, compatibility mode must
    be enabled so ``W_U`` carries the folded final LayerNorm weights.

    Does not call ``head_svd.require_isolated``: a degenerate direction's vocab
    readout is still a well-defined projection, unlike a per-direction causal claim,
    so it is not gated here. The contract that no direction is reported without a
    passing causal patch is enforced by :func:`patch_along_directions`.

    Args:
        model: A ``TransformerBridge`` with compatibility mode enabled; only its
            ``W_U`` is read.
        head_svd: An OV :class:`HeadSVD` from :func:`decompose_head`.
        k: Number of top singular directions to project.

    Returns:
        ``W_U.T @ head_svd.V[:, :k]``, shape ``[d_vocab, k]``: column i is
        direction i's projection through the unembedding.

    Raises:
        ValueError: If ``head_svd.which != "OV"``, if ``k`` is not in
            ``(0, rank]``, if ``head_svd`` was decomposed under a different
            compatibility-mode state than ``model`` now has, or if ``model`` is a
            ``TransformerBridge`` without compatibility mode enabled.
    """
    if head_svd.which != "OV":
        raise ValueError(f"vocab_readout requires an OV HeadSVD, got which={head_svd.which!r}")
    _validate_decomposition_matches_model(model, head_svd)
    rank = head_svd.V.shape[1]
    if not 0 < k <= rank:
        raise ValueError(f"k must be in (0, {rank}], got {k!r}")
    _validate_bridge_compatibility(model)
    # Promote V to W_U's dtype rather than forcing float32: matmul does not promote its
    # operands, so a bare .float() raises a dtype mismatch on any bf16, fp16, or fp64 model.
    W_U = model.W_U
    return W_U.T @ head_svd.V[:, :k].to(W_U.dtype)


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
        model: A ``TransformerBridge`` with compatibility mode enabled; only its
            ``W_U`` is read.
        head_svd: An OV :class:`HeadSVD` from :func:`decompose_head`.
        direction: Column index of the singular direction to reconstruct.
        tokens: Token id(s) to read the logit effect for.

    Returns:
        A :class:`LogitSignature` with one value per requested token.

    Raises:
        ValueError: If ``head_svd.which != "OV"``, if ``direction`` is not in
            ``[0, rank)``, if ``head_svd`` was decomposed under a different
            compatibility-mode state than ``model`` now has, or if ``model`` is a
            ``TransformerBridge`` without compatibility mode enabled.
        DegenerateDirectionError: If ``direction`` is not attributable alone (see
            :meth:`HeadSVD.require_isolated`).
    """
    if head_svd.which != "OV":
        raise ValueError(f"logit_signature requires an OV HeadSVD, got which={head_svd.which!r}")
    _validate_decomposition_matches_model(model, head_svd)
    # Bounds-check before indexing: a negative direction would wrap into V/rank_report and a
    # too-large one would raise a bare IndexError, both hiding a caller mistake as a wrong or
    # cryptic result rather than a clear refusal.
    rank = head_svd.V.shape[1]
    direction = int(direction)
    if not 0 <= direction < rank:
        raise ValueError(f"direction index {direction} out of range [0, {rank})")
    head_svd.require_isolated(direction)
    _validate_bridge_compatibility(model)
    token_ids = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
    # Match the rank-1 reconstruction to W_U's dtype for the projection: matmul does not
    # promote its operands, so a float32 reconstruction breaks on bf16, fp16, or fp64 models.
    W_U = model.W_U
    reconstruction = (head_svd.S[direction] * head_svd.V[:, direction]).to(W_U.dtype)
    values = reconstruction @ W_U[:, token_ids]
    return LogitSignature(direction=direction, values=values)


@dataclass
class ActivationProjection:
    """Per-position coefficients of a head's actual output in its OV output basis.

    Attributes:
        head_svd: The OV decomposition this was projected against.
        coefficients: ``[pos, rank]``; ``coefficients[:, i]`` is the signed amount of
            singular direction ``i`` (``head_svd.V[:, i]``) present in the head's actual
            output at each position. Summing ``coefficients[:, i] * head_svd.V[:, i]``
            over ``i`` reconstructs the head's real per-position output to numerical
            precision, since ``V``'s columns are orthonormal and this projects onto the
            exact basis the head writes in.
        str_tokens: Tokenized prompt, aligned with the position axis, for display.
    """

    head_svd: HeadSVD
    coefficients: Float[torch.Tensor, "pos rank"]
    str_tokens: List[str]


def project_activations(
    model, head_svd: HeadSVD, prompt: Union[str, torch.Tensor]
) -> ActivationProjection:
    """Project a head's actual per-position output onto its OV singular directions.

    Requires ``head_svd.which == "OV"``: this projects onto the write/output basis
    ``V``, and QK has no such vector (see the module docstring). Runs a real forward
    pass with ``use_attn_result`` enabled to read the per-head output
    (``hook_result``), then projects it onto ``head_svd.V``. Restores the model's
    prior ``use_attn_result`` setting afterward, since flipping that config flag as a
    side effect of a read-only analysis call would surprise a caller who already had
    hooks or a cache built around its prior state.

    Args:
        model: A ``TransformerBridge``.
        head_svd: An OV :class:`HeadSVD` from :func:`decompose_head`.
        prompt: A single prompt (not a batch): a string or a ``[pos]`` or ``[1, pos]``
            token tensor.

    Returns:
        An :class:`ActivationProjection` with the per-position coefficients.

    Raises:
        ValueError: If ``head_svd.which != "OV"``, if ``head_svd`` was decomposed under
            a different compatibility-mode state than ``model`` now has, or if ``prompt``
            is a batched token tensor (leading dimension greater than one).
        NotImplementedError: If the model's attention adapter exposes no per-head result,
            so ``set_use_attn_result(True)`` cannot fork the attention output.
    """
    if head_svd.which != "OV":
        raise ValueError(
            f"project_activations requires an OV HeadSVD, got which={head_svd.which!r}"
        )
    _validate_decomposition_matches_model(model, head_svd)
    # A batched token tensor carries a leading batch dimension larger than one, so reject it
    # before the forward instead of running the model on every row only to discard the result.
    # A 1-D [pos] tensor has no batch axis and runs as a single prompt, matching what
    # patch_along_directions accepts. A list of prompt strings only reveals its batch size once
    # tokenized, so the post-cache check below still guards that path.
    if isinstance(prompt, torch.Tensor) and prompt.ndim >= 2 and prompt.shape[0] > 1:
        raise ValueError(
            "project_activations requires a single prompt, got a batched token tensor of shape "
            f"{tuple(prompt.shape)}; pass a [pos] or [1, pos] tensor, or a single string."
        )
    # Cache only the one hook this reads. run_with_cache otherwise retains every hook point of
    # the forward pass (about 1 GB against 19 MB on a 512-token gpt2-small prompt) to read a
    # single head's output.
    hook_name = f"blocks.{head_svd.layer}.attn.hook_result"
    previous = getattr(model.cfg, "use_attn_result", False)
    model.set_use_attn_result(True)
    try:
        _, cache = model.run_with_cache(prompt, names_filter=lambda name: name == hook_name)
    finally:
        model.set_use_attn_result(previous)
    result = cache[hook_name][..., head_svd.head, :]
    if result.shape[0] != 1:
        raise ValueError(
            f"project_activations requires a single prompt, got batch={result.shape[0]}"
        )
    result = result.squeeze(0).to(head_svd.V.dtype)
    coefficients = result @ head_svd.V
    str_tokens = model.to_str_tokens(prompt)
    return ActivationProjection(head_svd=head_svd, coefficients=coefficients, str_tokens=str_tokens)


def _validate_retained_blocks(head_svd: HeadSVD, retained: Sequence[int]) -> None:
    """Raise if ``retained`` splits a degenerate block instead of keeping it whole or empty.

    A degenerate block's members are defined only up to a rotation (or, for a null
    block, arbitrary null-space vectors), so attributing a causal effect to part of the
    block while dropping the rest would let a caller route around the guard
    :meth:`HeadSVD.require_isolated` already enforces per direction.
    """
    retained_set = set(retained)
    for block in head_svd.degenerate_blocks():
        block_set = set(block)
        overlap = retained_set & block_set
        if overlap and overlap != block_set:
            raise DegenerateDirectionError(
                f"retained directions {sorted(overlap)} split block {sorted(block_set)} of "
                f"the {head_svd.which} SVD of head L{head_svd.layer}H{head_svd.head}, whose "
                f"members are not separated by a relative gap of eps={head_svd.eps:g} or are "
                f"jointly null. Keep or ablate the whole block, not part of it."
            )


def _resolve_retained(
    head_svd: HeadSVD, keep: Optional[Sequence[int]], ablate: Optional[Sequence[int]]
) -> List[int]:
    """Resolve ``keep``/``ablate`` to the sorted list of retained direction indices.

    Exactly one of ``keep``/``ablate`` must be given; ``ablate``'s complement over the
    map's full rank becomes the retained set. Every supplied index is coerced to ``int``
    and bounds-checked against ``[0, rank)`` before use, so a negative index raises rather
    than wrapping into ``V[:, ...]`` and an out-of-range ``ablate`` raises rather than
    silently subtracting nothing from the complement. Raises
    :class:`DegenerateDirectionError` if the result would split a degenerate block (see
    :func:`_validate_retained_blocks`).
    """
    if (keep is None) == (ablate is None):
        raise ValueError("patch_along_directions requires exactly one of keep or ablate")
    rank = head_svd.V.shape[1]

    def _checked(indices: Sequence[int], name: str) -> List[int]:
        resolved: List[int] = []
        for raw in indices:
            index = int(raw)
            if not 0 <= index < rank:
                raise ValueError(f"{name} index {index} out of range [0, {rank})")
            resolved.append(index)
        return resolved

    if keep is not None:
        retained = sorted(set(_checked(keep, "keep")))
    else:
        assert ablate is not None
        ablate_set = set(_checked(ablate, "ablate"))
        retained = [i for i in range(rank) if i not in ablate_set]
    _validate_retained_blocks(head_svd, retained)
    return retained


def _make_subspace_hook(head: int, projector: Float[torch.Tensor, "d_model d_model"]):
    """Build a ``hook_result`` hook that reconstructs one head's output onto ``span(projector)``.

    Leaves every other head's slice of the ``[batch, pos, head_index, d_model]`` tensor
    untouched. Clones before mutating so the hook never writes into the activation the
    forward pass itself is still using.
    """

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        activation = activation.clone()
        # The projector is drawn on CPU (QR is unimplemented on MPS and a CUDA generator
        # cannot feed a CPU randn), so move it to the activation's device and dtype here.
        activation[:, :, head, :] = activation[:, :, head, :] @ projector.to(
            device=activation.device, dtype=activation.dtype
        )
        return activation

    return hook_fn


@dataclass
class PatchResult:
    """Result of causally patching a head's output onto a chosen OV singular subspace.

    Attributes:
        head_svd: The OV decomposition patched against.
        retained: Direction indices whose span the head's output was reconstructed
            onto; the complement was zeroed.
        original_metric: Metric value on the unmodified prompt.
        patched_metric: Metric value after the subspace reconstruction.
        delta_metric: ``patched_metric - original_metric``.
        baseline_delta_metric: the mean of the per-draw ``delta_metric`` magnitudes over
            several random control subspaces of the same width as ``retained``, each drawn
            inside the head's own OV span ``span(V)`` rather than from the full residual
            stream, so the control is the effect of an arbitrary same-size subspace of this
            head's output rather than of an unrelated residual-stream direction. Averaging
            the magnitudes rather than the signed deltas keeps this a typical control
            effect that does not shrink when the controls mix sign, so it is non-negative.
        gated: whether the retained subspace passed the causal test for the mode it was
            expressed in, comparing ``abs(delta_metric)`` against ``baseline_delta_metric``
            (or an explicit threshold, if one was passed). For ``ablate`` (retain the complement),
            removing a load-bearing subspace should move the metric more than removing an
            arbitrary same-size one, so ``gated`` is ``abs(delta_metric) > threshold``.
            For ``keep`` (retain only the given subspace), a subspace that reconstructs
            the head's behavior should move the metric less than keeping an arbitrary
            same-size one, so ``gated`` is ``abs(delta_metric) < threshold``. A single
            "moved more than baseline" test cannot answer both, since ``keep=S`` and
            ``ablate=complement(S)`` resolve to the same retained set.
    """

    head_svd: HeadSVD
    retained: List[int]
    original_metric: float
    patched_metric: float
    delta_metric: float
    baseline_delta_metric: float
    gated: bool


@torch.no_grad()
def patch_along_directions(
    model,
    head_svd: HeadSVD,
    prompt: Union[str, torch.Tensor],
    metric: Callable[[torch.Tensor], float],
    *,
    keep: Optional[Sequence[int]] = None,
    ablate: Optional[Sequence[int]] = None,
    threshold: Optional[float] = None,
    rng: Optional[torch.Generator] = None,
    n_baseline: int = _DEFAULT_N_BASELINE,
) -> PatchResult:
    """Causally validate a claimed OV subfunction by reconstructing the head's output onto it.

    Requires ``head_svd.which == "OV"``: this reconstructs the write/output basis
    ``V``, and QK has no such vector (see the module docstring). Runs the prompt with
    ``use_attn_result`` enabled once unmodified, once with the head's ``hook_result``
    slice reconstructed onto ``span(head_svd.V[:, retained])``, and once per random
    control subspace. Each control subspace is drawn *inside the head's own OV span*
    ``span(V)`` (not from the full residual stream, where a width-``w`` random subspace
    would keep only ``w/d_model`` of a head output that itself occupies only
    ``rank/d_model`` of the stream; an in-span control of width ``w`` keeps ``w/rank``
    of it), so a moved metric is compared against the effect of an arbitrary
    subspace of this head's output of the same width. The per-draw control delta magnitudes
    are averaged over ``n_baseline`` draws so one lucky or unlucky draw does not decide the
    gate, and so controls that mix sign do not cancel into a smaller threshold. Restores
    the model's prior ``use_attn_result`` setting afterward.

    The gate's success condition depends on the mode the caller expressed, because
    ``keep=S`` and ``ablate=complement(S)`` resolve to the same retained set and a single
    "moved more than the control" test would answer only the ``ablate`` question. See
    :attr:`PatchResult.gated`.

    Args:
        model: A ``TransformerBridge``.
        head_svd: An OV :class:`HeadSVD` from :func:`decompose_head`.
        prompt: A single prompt: a string or a ``[1, pos]`` token tensor.
        metric: A function from the model's logits to a scalar.
        keep: Direction indices to retain; the rest are zeroed. Exactly one of
            ``keep``/``ablate`` must be given.
        ablate: Direction indices to zero; the rest are retained.
        threshold: Explicit gate threshold. Defaults to ``None``, which uses
            ``baseline_delta_metric`` (already a magnitude) instead.
        rng: Optional generator for the random control subspaces, for reproducibility.
            Defaults to a generator seeded with ``_DEFAULT_BASELINE_SEED`` so a bare
            call is reproducible rather than drawing from the global RNG.
        n_baseline: Number of random in-span control subspaces to average the baseline
            delta over. Must be at least 1.

    Returns:
        A :class:`PatchResult` describing the patched, baseline, and original metrics.

    Raises:
        ValueError: If ``head_svd.which != "OV"``, if ``head_svd`` was decomposed under a
            different compatibility-mode state than ``model`` now has, if ``keep``/``ablate``
            are both given or both omitted, if any index is out of ``[0, rank)``, if the
            retained set is empty (``keep=[]`` or ``ablate`` over the full rank) or spans the
            full rank (``keep`` over every direction) and no explicit ``threshold`` is
            supplied, or if ``n_baseline < 1``.
        NotImplementedError: If the model's attention adapter exposes no per-head result,
            so ``set_use_attn_result(True)`` cannot fork the attention output.
        DegenerateDirectionError: If the retained directions split a degenerate block
            (see :func:`_validate_retained_blocks`).
    """
    if head_svd.which != "OV":
        raise ValueError(
            f"patch_along_directions requires an OV HeadSVD, got which={head_svd.which!r}"
        )
    if n_baseline < 1:
        raise ValueError(f"n_baseline must be at least 1, got {n_baseline}")
    _validate_decomposition_matches_model(model, head_svd)
    retained = _resolve_retained(head_svd, keep, ablate)
    # _resolve_retained has confirmed exactly one of keep/ablate is set, so the mode the
    # caller expressed is unambiguous and selects the gate's success condition (see below).
    mode = "keep" if keep is not None else "ablate"
    V = head_svd.V
    rank = V.shape[1]
    if (not retained or len(retained) == rank) and threshold is None:
        # An empty retained set reconstructs the head onto the zero subspace; a full-rank
        # retained set reconstructs it onto span(V) itself (V V^T). In both cases the kept
        # projector and the equal-width random control projector coincide: for the empty set
        # both are the zero projector, and for the full set every width-rank in-span control
        # is V Q Q^T V^T = V V^T. The two deltas then tie by construction and the gate
        # compares a quantity against itself. Require an explicit threshold to gate either
        # degenerate set on purpose.
        raise ValueError(
            "keep/ablate retains either no directions or every direction, so the "
            "reconstruction's kept projector coincides with the random control projector "
            "and their deltas tie by construction; pass an explicit threshold to gate this."
        )

    if rng is None:
        rng = torch.Generator().manual_seed(_DEFAULT_BASELINE_SEED)

    width = len(retained)
    kept_projector = V[:, retained] @ V[:, retained].transpose(-2, -1)

    hook_name = f"blocks.{head_svd.layer}.attn.hook_result"
    previous = getattr(model.cfg, "use_attn_result", False)
    model.set_use_attn_result(True)
    try:
        original_metric = float(metric(model(prompt)))
        patched_logits = model.run_with_hooks(
            prompt, fwd_hooks=[(hook_name, _make_subspace_hook(head_svd.head, kept_projector))]
        )
        patched_metric = float(metric(patched_logits))

        baseline_deltas: List[float] = []
        for _ in range(n_baseline):
            # Draw a random width-of-rank subspace inside the head's own OV span. The QR
            # is drawn on CPU (unimplemented on MPS, and a CUDA generator cannot feed a
            # CPU randn); the columns are mapped into span(V) on V's device, and the hook
            # moves the finished projector to the activation's device.
            random_rank = torch.randn(rank, rank, generator=rng, dtype=V.dtype)
            random_basis, _ = torch.linalg.qr(random_rank)
            baseline_directions = V @ random_basis[:, :width].to(V.device)
            baseline_projector = baseline_directions @ baseline_directions.transpose(-2, -1)
            baseline_logits = model.run_with_hooks(
                prompt,
                fwd_hooks=[(hook_name, _make_subspace_hook(head_svd.head, baseline_projector))],
            )
            baseline_deltas.append(float(metric(baseline_logits)) - original_metric)
    finally:
        model.set_use_attn_result(previous)

    delta_metric = patched_metric - original_metric
    baseline_delta_metric = sum(abs(d) for d in baseline_deltas) / len(baseline_deltas)
    gate_threshold = baseline_delta_metric if threshold is None else threshold
    # keep retains only the claimed subspace, so it passes when it reconstructs the head's
    # behavior better than an arbitrary same-width one (moves the metric less); ablate
    # removes it, so it passes when removing it matters more than removing an arbitrary
    # same-width one (moves the metric more).
    if mode == "keep":
        gated = abs(delta_metric) < gate_threshold
    else:
        gated = abs(delta_metric) > gate_threshold

    return PatchResult(
        head_svd=head_svd,
        retained=retained,
        original_metric=original_metric,
        patched_metric=patched_metric,
        delta_metric=delta_metric,
        baseline_delta_metric=baseline_delta_metric,
        gated=gated,
    )
