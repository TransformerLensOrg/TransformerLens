"""Per-head SVD of attention QK (``W_Q W_K^T``) and OV (``W_V W_O``) maps, plus OV consumers;
kept factored (rank <= ``d_head``), with rotation-ambiguous blocks and null directions flagged."""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Float

from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.tools.analysis._model_state import require_eval_mode

Which = Literal["QK", "OV"]

_VALID_WHICH: Tuple[Which, ...] = ("QK", "OV")

# Adjacent singular values within this relative gap share one rotation-ambiguous block.
_DEFAULT_EPS = 1e-2

# Absolute floor so relative-gap and ratio computations never divide by ~0.
_SIGMA_FLOOR = 1e-12

# Seeds the patch comparison's random baseline so a bare call is reproducible.
_DEFAULT_BASELINE_SEED = 0

# Per-draw control magnitudes are heavy-tailed, so the averaged threshold still varies with the
# seed; pass an rng and raise n_baseline when a verdict sits near it.
_DEFAULT_N_BASELINE = 32


class DegenerateDirectionError(ValueError):
    """Raised when per-direction attribution is requested for a degenerate direction."""


@dataclass
class RankReportRow:
    """One singular direction's summary, aligned with column ``idx`` of ``U``/``V``."""

    idx: int
    sigma: float
    sigma_ratio: float  # sigma / largest sigma
    is_degenerate: bool  # shares a block with a neighbour, or is null
    is_null: bool  # sigma_ratio < null_rtol: an arbitrary null-space vector
    block_id: int


@dataclass
class HeadSVD:
    """Factored SVD of one head map (``"QK"`` or ``"OV"``) with a degeneracy report."""

    which: Which
    layer: int
    head: int
    U: Float[torch.Tensor, "d_model rank"]  # input (read) directions
    S: Float[torch.Tensor, "rank"]  # descending
    V: Float[torch.Tensor, "d_model rank"]  # OV: write directions; QK: key-read directions
    rank_report: List[RankReportRow]
    eps: float
    null_rtol: float
    # Fold state and float64 (sum, sum of squares) per factor at decomposition time; consumers
    # refuse a model that no longer matches. A None fingerprint skips the weight check.
    folded_ln: bool = False
    weight_fingerprint: Optional[Tuple[float, ...]] = None

    def is_degenerate(self, i: int) -> bool:
        """Whether direction ``i`` is refused: rotation-ambiguous inside a block, or null."""
        return self.rank_report[i].is_degenerate

    def block_of(self, i: int) -> List[int]:
        """Every direction index in direction ``i``'s block (``[i]`` when isolated)."""
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
        """Raise :class:`DegenerateDirectionError` unless direction ``i`` is attributable alone."""
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
    """Return value of :func:`decompose_head`; a map that was not requested is ``None``."""

    layer: int
    head: int
    QK: Optional[HeadSVD] = None
    OV: Optional[HeadSVD] = None


def _read_weight(weight: torch.Tensor) -> torch.Tensor:
    """Detach for SVD: fp16/bf16 promote to float32 (unstable SVD); float64 stays so the default
    ``null_rtol`` matches the precision actually decomposed."""
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
    """``(W_Q_h, W_K_h, W_V_h, W_O_h)`` read from one block, not the full-model stacks."""
    attn = model.blocks[layer].attn
    n_kv_heads = attn.W_K.shape[0]
    kv_head = head // (model.cfg.n_heads // n_kv_heads)  # GQA: a query head reads its group's
    W_Q_h = _read_weight(attn.W_Q[head])
    W_K_h = _read_weight(attn.W_K[kv_head])
    W_V_h = _read_weight(attn.W_V[kv_head])
    W_O_h = _read_weight(attn.W_O[head])
    return W_Q_h, W_K_h, W_V_h, W_O_h


def _map_factors(model, layer: int, head: int) -> Dict[Which, Tuple[torch.Tensor, torch.Tensor]]:
    """One head's weights paired by map; QK's ``W_K_h`` is transposed only when decomposed."""
    W_Q_h, W_K_h, W_V_h, W_O_h = _head_weights(model, layer, head)
    return {"QK": (W_Q_h, W_K_h), "OV": (W_V_h, W_O_h)}


def _weight_fingerprint(*weights: torch.Tensor) -> Tuple[float, ...]:
    """Float64 sum and sum of squares of each weight, a cheap check that it is unchanged."""
    stats: List[float] = []
    for weight in weights:
        # Moved to CPU before widening: a combined .to() casts on MPS, which has no float64.
        w = weight.detach().cpu().to(torch.float64)
        stats.extend((float(w.sum()), float(w.square().sum())))
    return tuple(stats)


def _degeneracy_blocks(
    S: Float[torch.Tensor, "rank"], eps: float, null_rtol: float
) -> List[List[int]]:
    """Blocks of descending ``S``: ``i`` joins if ``1 - S[i]/S[i-1] < eps`` or both are null, so a
    block may chain wider than ``eps`` but every boundary is an ``eps`` gap."""
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
    """One :class:`RankReportRow` per direction; ``blocks`` must partition ``range(len(S))``."""
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
    folded_ln: bool = False,
    weight_fingerprint: Optional[Tuple[float, ...]] = None,
) -> HeadSVD:
    """SVD of the factored map ``A @ B`` (OV: ``W_V_h, W_O_h``; QK: ``W_Q_h, W_K_h^T``)."""
    U, S, V = FactoredMatrix(A, B).svd()
    d_model = U.shape[0]
    # torch.linalg.matrix_rank's default tolerance.
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
        folded_ln=folded_ln,
        weight_fingerprint=weight_fingerprint,
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
    """Decompose a head's QK and/or OV maps via SVD; weight-space only, factors detached."""
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

    factors = _map_factors(model, layer, head)
    folded_ln = _folded_ln_state(model)
    svds: Dict[Which, HeadSVD] = {}
    for name, (A, B) in factors.items():
        if name in requested:
            svds[name] = _factored_head_svd(
                A,
                B.transpose(-1, -2) if name == "QK" else B,
                which=name,
                layer=layer,
                head=head,
                eps=eps,
                null_rtol=null_rtol,
                folded_ln=folded_ln,
                weight_fingerprint=_weight_fingerprint(A, B),
            )
    return HeadDecomposition(layer=layer, head=head, QK=svds.get("QK"), OV=svds.get("OV"))


def _folded_ln_state(model) -> bool:
    """Whether the weights carry a folded final LayerNorm; the ``compatibility_mode`` flag does not
    imply one, and ``ProcessWeights`` drops ``fold_ln`` where ``supports_fold_ln`` is False."""
    processed = getattr(model, "_weights_processed", False)
    adapter = getattr(model, "adapter", None)
    fold_ln_requested = getattr(adapter, "_fold_ln_requested", True)
    supports_fold_ln = getattr(adapter, "supports_fold_ln", True)
    return bool(processed and fold_ln_requested and supports_fold_ln)


def _validate_bridge_compatibility(model) -> None:
    """Reject a ``TransformerBridge`` whose ``W_U`` lacks the folded final LayerNorm; hybrid archs
    pass, since a rank-1 ``W_U`` projection needs no block layout."""
    from transformer_lens.model_bridge import TransformerBridge

    if not isinstance(model, TransformerBridge):
        return
    if _folded_ln_state(model):
        return
    requirement = (
        "Projecting an OV direction through W_U on a TransformerBridge requires the "
        "final LayerNorm folded into W_U"
    )
    adapter = getattr(model, "adapter", None)
    if not getattr(adapter, "supports_fold_ln", True):
        raise ValueError(
            f"{requirement}, but {type(adapter).__name__} does not support fold_ln, so no "
            "weight processing can fold it. project_activations and patch_along_directions "
            "need no fold and still work on this model."
        )
    if getattr(model, "_weights_processed", False):
        raise ValueError(
            f"{requirement}, but this bridge's weights were already processed with fold_ln "
            "disabled and cannot be processed again. Boot a fresh bridge and call "
            "`model.enable_compatibility_mode()` with fold_ln enabled, then retry."
        )
    raise ValueError(
        f"{requirement}, which only happens once the bridge has processed its weights with "
        "fold_ln enabled. Call `model.enable_compatibility_mode()` (its default folds "
        "LayerNorm) after loading the bridge, then retry."
    )


def _validate_decomposition_matches_model(model, head_svd: HeadSVD) -> None:
    """Refuse a decomposition whose folded-LayerNorm state or head weights no longer match."""
    current = _folded_ln_state(model)
    if current != head_svd.folded_ln:
        raise ValueError(
            f"This HeadSVD was decomposed with folded_ln={head_svd.folded_ln} but the "
            f"model's weights now have folded_ln={current}; the cached singular vectors "
            f"describe a different OV map. Re-run decompose_head under the current state, "
            f"then retry."
        )
    recorded = head_svd.weight_fingerprint
    if recorded is None:
        return
    # enable_compatibility_mode(fold_ln=False) centres W_O without changing folded_ln.
    live = _weight_fingerprint(*_map_factors(model, head_svd.layer, head_svd.head)[head_svd.which])
    if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9) for a, b in zip(recorded, live)):
        raise ValueError(
            f"The weights of head L{head_svd.layer}H{head_svd.head} changed after this HeadSVD "
            f"was decomposed; the cached singular vectors describe a different "
            f"{head_svd.which} map. Re-run decompose_head under the current state, then retry."
        )


def _require_ov(head_svd: HeadSVD, operation: str) -> None:
    """Refuse a QK decomposition: only OV has write directions to read out, project or patch."""
    if head_svd.which != "OV":
        raise ValueError(f"{operation} requires an OV HeadSVD, got which={head_svd.which!r}")


def vocab_readout(model, head_svd: HeadSVD, *, k: int = 10) -> Float[torch.Tensor, "d_vocab k"]:
    """``W_U.T @ V[:, :k]``: the top-``k`` OV output directions projected into vocab space; not
    gated on degeneracy, as a degenerate direction's readout is still well-defined."""
    _require_ov(head_svd, "vocab_readout")
    _validate_decomposition_matches_model(model, head_svd)
    rank = head_svd.V.shape[1]
    if not 0 < k <= rank:
        raise ValueError(f"k must be in (0, {rank}], got {k!r}")
    _validate_bridge_compatibility(model)
    # Cast to W_U's dtype, not float32: matmul does not promote, so bf16/fp16/fp64 would raise.
    W_U = model.W_U
    return W_U.T @ head_svd.V[:, :k].to(W_U.dtype)


@dataclass
class LogitSignature:
    """Signed logit contribution of one OV direction's rank-1 reconstruction, per token."""

    direction: int
    values: Float[torch.Tensor, "token"]


def logit_signature(
    model,
    head_svd: HeadSVD,
    direction: int,
    tokens: Union[int, Sequence[int], torch.Tensor],
) -> LogitSignature:
    """Logits of ``S[direction] * V[:, direction]`` read at ``tokens``; weight-space only."""
    _require_ov(head_svd, "logit_signature")
    _validate_decomposition_matches_model(model, head_svd)
    # Explicit bounds check: a negative index would silently wrap into V and rank_report.
    rank = head_svd.V.shape[1]
    direction = int(direction)
    if not 0 <= direction < rank:
        raise ValueError(f"direction index {direction} out of range [0, {rank})")
    head_svd.require_isolated(direction)
    _validate_bridge_compatibility(model)
    token_ids = torch.as_tensor(tokens, dtype=torch.long).reshape(-1)
    # matmul does not promote its operands, so match W_U's dtype on bf16, fp16, or fp64 models.
    W_U = model.W_U
    reconstruction = (head_svd.S[direction] * head_svd.V[:, direction]).to(W_U.dtype)
    values = reconstruction @ W_U[:, token_ids]
    return LogitSignature(direction=direction, values=values)


@dataclass
class ActivationProjection:
    """Per-position coefficients of a head's actual output in its OV basis ``V``."""

    head_svd: HeadSVD
    coefficients: Float[torch.Tensor, "pos rank"]  # coefficients @ V.T reconstructs the output
    str_tokens: List[str]


def project_activations(
    model, head_svd: HeadSVD, prompt: Union[str, torch.Tensor]
) -> ActivationProjection:
    """Project a head's per-position ``hook_result`` for one prompt onto its OV directions."""
    _require_ov(head_svd, "project_activations")
    _validate_decomposition_matches_model(model, head_svd)
    # Reject non-single-prompt tensors before the forward; a list of strings shows its batch
    # size only once tokenized, so the post-cache check below still guards that path.
    if isinstance(prompt, torch.Tensor) and (
        prompt.ndim not in (1, 2) or (prompt.ndim == 2 and prompt.shape[0] > 1)
    ):
        raise ValueError(
            "project_activations requires a single prompt, got a token tensor of shape "
            f"{tuple(prompt.shape)}; pass a [pos] or [1, pos] tensor, or a single string."
        )
    require_eval_mode(model, operation="project_activations()")
    # Cache only this hook: an unfiltered cache is ~1 GB vs 19 MB on a 512-token gpt2-small prompt.
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
    """Raise if ``retained`` splits a degenerate block, which would bypass ``require_isolated``."""
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
    """Sorted retained indices from exactly one of ``keep`` or ``ablate`` (complemented)."""
    if (keep is None) == (ablate is None):
        raise ValueError("patch_along_directions requires exactly one of keep or ablate")
    rank = head_svd.V.shape[1]

    def _checked(indices: Sequence[int], name: str) -> List[int]:
        resolved: List[int] = []
        for raw in indices:
            index = int(raw)
            # Explicit, so a negative index cannot wrap and a bad ablate cannot no-op.
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
    """``hook_result`` hook projecting only ``head``'s slice onto ``span(projector)``."""

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        # Clone so the hook never writes into an activation the forward pass is still using.
        activation = activation.clone()
        # Control projectors are drawn on CPU (QR is unimplemented on MPS), so match the activation.
        activation[:, :, head, :] = activation[:, :, head, :] @ projector.to(
            device=activation.device, dtype=activation.dtype
        )
        return activation

    return hook_fn


@dataclass
class PatchResult:
    """A head's output patched onto an OV subspace, compared with same-width in-span controls."""

    head_svd: HeadSVD
    retained: List[int]
    original_metric: float
    patched_metric: float
    delta_metric: float  # patched_metric - original_metric
    baseline_delta_metric: float  # mean |delta| over random controls inside span(V), so >= 0
    gated: bool  # keep: |delta| < threshold; ablate: |delta| > threshold


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
    """Patch a head's output onto an OV subspace and compare the metric change to in-span controls;
    ``threshold`` overrides the control mean."""
    _require_ov(head_svd, "patch_along_directions")
    if n_baseline < 1:
        raise ValueError(f"n_baseline must be at least 1, got {n_baseline}")
    _validate_decomposition_matches_model(model, head_svd)
    retained = _resolve_retained(head_svd, keep, ablate)
    # keep=S and ablate=complement(S) share a retained set, so only the mode fixes gate polarity.
    mode = "keep" if keep is not None else "ablate"
    V = head_svd.V
    rank = V.shape[1]
    if (not retained or len(retained) == rank) and threshold is None:
        # Empty and full-rank retained sets make the kept projector equal every same-width
        # control (0, or V V^T), so the two deltas would tie by construction.
        raise ValueError(
            "keep/ablate retains either no directions or every direction, so the "
            "reconstruction's kept projector coincides with the random control projector "
            "and their deltas tie by construction; pass an explicit threshold to gate this."
        )

    require_eval_mode(model, operation="patch_along_directions()")

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
            # Controls stay inside span(V): a full-stream width-w subspace would keep only
            # w/d_model of a head output that itself spans rank/d_model of the stream. The QR
            # runs on CPU (unimplemented on MPS; a CUDA generator cannot feed a CPU randn).
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
    # Mean of magnitudes, not of signed deltas, so mixed-sign controls cannot shrink it.
    baseline_delta_metric = sum(abs(d) for d in baseline_deltas) / len(baseline_deltas)
    gate_threshold = baseline_delta_metric if threshold is None else threshold
    # keep passes by preserving the metric better than a random subspace; ablate by moving it more.
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
