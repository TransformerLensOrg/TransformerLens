"""Leakage-safe k-sparse probing over activation tensors.

Implements the mean-difference heuristic from Gurnee et al. (2023) with
strict leakage guards: the single stratified train/test split is fixed
before any learned statistic, feature selection and optional
standardisation are computed on the training split only, controls reuse
that split, and all randomness is confined to a local ``torch.Generator``.

Reference defaults intentionally diverge from the paper's sklearn/SAGA
pipeline: centred train-only standardisation, balanced logistic loss with
explicit L2 convention, and deterministic CPU float64 LBFGS. See the
sparse-probing guide for composition with ``run_with_cache``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SparseProbeMetrics:
    """Held-out classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class SparseProbeResult:
    """Result of a single :func:`fit_sparse_probe` call."""

    k: int
    test_fraction: float
    positive_label: int
    preprocess: str
    l2_strength: float
    seed: int

    # Class bookkeeping
    labels: List[int]
    class_counts: Dict[int, int]
    n_train: int
    n_test: int
    n_train_per_class: Dict[int, int]
    n_test_per_class: Dict[int, int]
    train_indices: torch.Tensor
    test_indices: torch.Tensor

    # Selection
    selected_indices: List[int]
    selected_scores: List[float]  # raw signed mean-difference

    # Fit
    coefficients: torch.Tensor  # shape [k] CPU float64
    intercept: float
    objective: float
    grad_norm: float
    n_iter: int
    converged: bool

    # Preprocessing metadata
    preprocess_mean: Optional[torch.Tensor] = None  # shape [k] or None
    preprocess_scale: Optional[torch.Tensor] = None  # shape [k] or None

    # Metrics
    metrics: SparseProbeMetrics = None  # type: ignore[assignment]
    logits_test: Optional[torch.Tensor] = None  # shape [n_test] float64
    preds_test: Optional[torch.Tensor] = None  # shape [n_test] bool/int

    def summary(self) -> str:
        return (
            f"k={self.k} f1={self.metrics.f1:.3f} acc={self.metrics.accuracy:.3f} "
            f"sel={self.selected_indices}"
        )


@dataclass
class SparseSweepResult:
    """Result of :func:`sweep_sparse_probe`."""

    ks: List[int]
    test_fraction: float
    positive_label: int
    preprocess: str
    l2_strength: float
    seed: int
    probes: List[SparseProbeResult] = field(default_factory=list)
    # controls: one entry per k, each is list of metrics or full results
    random_controls: List[List[SparseProbeResult]] = field(default_factory=list)
    label_shuffle_controls: List[List[SparseProbeResult]] = field(default_factory=list)
    # shared split indices (same as in every probe)
    train_indices: Optional[torch.Tensor] = None
    test_indices: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_X_y(
    X: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(X, torch.Tensor):
        raise TypeError(f"X must be a torch.Tensor, got {type(X)}")
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"y must be a torch.Tensor, got {type(y)}")
    if X.ndim != 2:
        raise ValueError(f"X must have shape [n, d], got {tuple(X.shape)}")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1-D, got shape {tuple(y.shape)}")
    if y.ndim == 2:
        if y.shape[1] != 1:
            raise ValueError(f"y must be 1-D, got shape {tuple(y.shape)}")
        y = y.squeeze(1)
    n, d = X.shape
    if y.shape[0] != n:
        raise ValueError(f"X and y must have same n, got {X.shape[0]} vs {y.shape[0]}")
    if X.numel() == 0:
        raise ValueError("X must be non-empty")
    if not X.is_floating_point():
        raise TypeError(f"X must be floating, got {X.dtype}")
    if not torch.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    # y must be Boolean or integer
    if y.dtype == torch.bool:
        pass
    elif y.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        pass
    else:
        raise TypeError(f"y must be Boolean or integer, got {y.dtype}")
    return X, y


def _validate_k(k: int, d: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k must be int, got {type(k)}")
    if not (1 <= k <= d):
        raise ValueError(f"k must satisfy 1 <= k <= d ({d}), got {k}")


def _validate_test_fraction(v: float) -> None:
    if not isinstance(v, (float, int)) or isinstance(v, bool):
        raise TypeError(f"test_fraction must be float, got {type(v)}")
    vf = float(v)
    if not (0 < vf < 1):
        raise ValueError(f"test_fraction must satisfy 0 < test_fraction < 1, got {v}")


def _validate_l2(v: float) -> None:
    if not isinstance(v, (float, int)) or isinstance(v, bool):
        raise TypeError(f"l2_strength must be float, got {type(v)}")
    if float(v) <= 0:
        raise ValueError(f"l2_strength must be positive, got {v}")


def _validate_labels(y: torch.Tensor, positive_label) -> Tuple[List[int], torch.Tensor]:
    # positive_label must be one of the values; y must contain exactly two values
    # Handle bool y: values are 0/1 internally but positive_label may be True/False/0/1
    # Normalise y to int64 for bookkeeping
    if y.dtype == torch.bool:
        y_int = y.to(torch.int64)
        # map positive_label to 0/1 if it's bool
        if isinstance(positive_label, bool):
            pos_int = int(positive_label)
        else:
            # allow 0/1 ints as well
            try:
                pos_int = int(positive_label)
            except Exception as e:
                raise TypeError(f"positive_label {positive_label!r} not in y values") from e
            if pos_int not in (0, 1):
                raise ValueError(f"positive_label {positive_label!r} not in y values (bool y has 0/1)")
        labels = [0, 1]
        # check exactly two values present
        uniq = torch.unique(y_int)
        if uniq.numel() != 2:
            raise ValueError(f"y must contain exactly two values, got {uniq.tolist()}")
        if pos_int not in uniq.tolist():
            raise ValueError(f"positive_label {positive_label!r} not in y values {uniq.tolist()}")
        return labels, y_int

    # integer y
    y_int = y.to(torch.int64)
    uniq = torch.unique(y_int)
    if uniq.numel() != 2:
        raise ValueError(f"y must contain exactly two values, got {uniq.tolist()}")
    # positive_label must be equal to one of them (exact int)
    # allow bool positive_label for int y? coerce
    if isinstance(positive_label, bool):
        # bool is subclass of int, but disambiguate
        pos_candidates = [int(positive_label)]
    else:
        pos_candidates = [positive_label]
    # also try int coercion
    found = False
    pos_int: int = 0
    for c in uniq.tolist():
        if c == positive_label or (isinstance(positive_label, (int,)) and c == int(positive_label)):
            found = True
            pos_int = int(c)
            break
    if not found:
        raise ValueError(f"positive_label {positive_label!r} not in y values {uniq.tolist()}")
    labels = sorted(uniq.tolist())  # deterministic
    return labels, y_int


def _check_class_counts(y_int: torch.Tensor) -> None:
    uniq, counts = torch.unique(y_int, return_counts=True)
    for c, cnt in zip(uniq.tolist(), counts.tolist()):
        if cnt < 2:
            raise ValueError(f"need at least 2 examples per class, got class {c}: {cnt}")


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def _stratified_split(
    y_int: torch.Tensor,
    test_fraction: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], Dict[int, int]]:
    """Return (train_indices, test_indices, n_train_per_class, n_test_per_class)."""
    n = y_int.shape[0]
    uniq = torch.unique(y_int)
    train_parts: List[torch.Tensor] = []
    test_parts: List[torch.Tensor] = []
    n_train_per_class: Dict[int, int] = {}
    n_test_per_class: Dict[int, int] = {}
    for c in uniq.tolist():
        mask = (y_int == c).nonzero(as_tuple=False).squeeze(1)
        n_c = mask.shape[0]
        n_test_c = math.ceil(test_fraction * n_c)
        n_test_c = max(1, min(n_test_c, n_c - 1))
        n_train_c = n_c - n_test_c
        # permute with generator
        perm = torch.randperm(n_c, generator=generator)
        shuffled = mask[perm]
        test_idx = shuffled[:n_test_c]
        train_idx = shuffled[n_test_c:]
        test_parts.append(test_idx)
        train_parts.append(train_idx)
        n_train_per_class[int(c)] = int(n_train_c)
        n_test_per_class[int(c)] = int(n_test_c)
    # shuffle global order? Keep deterministic but mix classes: permute final arrays
    train_indices = torch.cat(train_parts)
    test_indices = torch.cat(test_parts)
    # Shuffle the concatenated order deterministically so class blocks are interleaved
    train_perm = torch.randperm(train_indices.shape[0], generator=generator)
    test_perm = torch.randperm(test_indices.shape[0], generator=generator)
    train_indices = train_indices[train_perm]
    test_indices = test_indices[test_perm]
    return train_indices, test_indices, n_train_per_class, n_test_per_class


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _mean_difference_scores(
    X_train: torch.Tensor,
    y_train_int: torch.Tensor,
    positive_label: int,
) -> torch.Tensor:
    """Return raw mean-difference score per feature, shape [d]."""
    # Ensure at least float32 computation on input device
    if X_train.dtype in (torch.float16, torch.bfloat16):
        Xc = X_train.to(torch.float32)
    else:
        Xc = X_train
    # also ensure float for mean
    pos_mask = y_train_int == positive_label
    neg_mask = ~pos_mask
    # If y has only two values, neg is the other label; pos_mask selects positive
    # Use float32+ for reduction
    pos_mean = Xc[pos_mask].mean(dim=0)
    neg_mean = Xc[neg_mask].mean(dim=0)
    scores = pos_mean - neg_mean  # shape [d], preserves device
    return scores


def _select_top_k(scores: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
    abs_scores = scores.abs()
    d = scores.shape[0]
    # deterministic tie-breaking: stable sort descending by abs, lower index wins on ties
    # torch.argsort is stable for `stable=True` (pytorch 2.0+)
    try:
        order = torch.argsort(abs_scores, descending=True, stable=True)
    except TypeError:
        # fallback: use sort with stable
        order = torch.argsort(abs_scores, descending=True)
        # If not stable, tie-break by index: use lexsort via python
        # We'll do manual stable: sort by (-abs_score, index)
        # Since argsort may not be stable in old torch, we recompute via python
        vals = abs_scores.tolist()
        order_list = sorted(range(d), key=lambda i: (-vals[i], i))
        order = torch.tensor(order_list, dtype=torch.long, device=scores.device)
    top = order[:k]
    selected = top.tolist()
    selected_scores = scores[top].tolist()
    return selected, selected_scores


# ---------------------------------------------------------------------------
# Logistic fitting
# ---------------------------------------------------------------------------


def _fit_logistic_lbfgs(
    X_train_sel: torch.Tensor,
    y_train_bin: torch.Tensor,
    X_test_sel: torch.Tensor,
    l2_strength: float,
    class_weights: Dict[int, float],
    positive_label: int,
    y_train_int: torch.Tensor,
    grad_threshold: float,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, float, float, float, int, bool]:
    """Fit on CPU float64 with LBFGS strong-Wolfe.

    Returns (w, b, objective, grad_norm, n_iter, converged).
    Raises if not finite or grad_norm > threshold.
    """
    device = torch.device("cpu")
    # Move to CPU float64
    Xt = X_train_sel.to(device=device, dtype=torch.float64)
    Xte = X_test_sel.to(device=device, dtype=torch.float64)
    # y_train_bin is 0/1 where 1==positive
    yt = y_train_bin.to(device=device, dtype=torch.float64)
    # weights per example
    n = yt.shape[0]
    # map class -> weight, using original labels counts
    # class_weights already n/(2*n_c)
    w_per_example = torch.empty(n, dtype=torch.float64, device=device)
    for i, lab in enumerate(y_train_int.tolist()):
        # map lab to weight: if lab==positive_label then weight for positive class
        # else weight for negative class; we have two entries in class_weights
        w_per_example[i] = class_weights[int(lab)]

    k = Xt.shape[1]
    w = torch.zeros(k, dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [w, b],
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    # closure
    def closure():
        optimizer.zero_grad()
        logits = Xt @ w + b  # [n]
        # BCE with logits, no reduction
        bce = F.binary_cross_entropy_with_logits(logits, yt, reduction="none")
        weighted = w_per_example * bce
        loss = weighted.mean() + l2_strength * (w.pow(2).sum() / 2.0)
        loss.backward()
        return loss

    # LBFGS step
    try:
        loss_val = optimizer.step(closure)
    except Exception as e:
        raise RuntimeError(f"LBFGS failed: {e}") from e

    # After optimization, compute final objective and grad norm
    # Need to recompute grad
    optimizer.zero_grad()
    logits = Xt @ w + b
    bce = F.binary_cross_entropy_with_logits(logits, yt, reduction="none")
    weighted = w_per_example * bce
    loss = weighted.mean() + l2_strength * (w.pow(2).sum() / 2.0)
    loss.backward()
    # grad inf norm over w and b
    grads = []
    if w.grad is not None:
        grads.append(w.grad.detach().abs().max().item())
    if b.grad is not None:
        grads.append(b.grad.detach().abs().max().item())
    grad_norm = max(grads) if grads else float("inf")
    objective = loss.detach().item()

    if not math.isfinite(objective) or not math.isfinite(grad_norm):
        raise RuntimeError(f"non-finite output: objective={objective} grad_norm={grad_norm}")
    if not torch.isfinite(w).all() or not torch.isfinite(b).all():
        raise RuntimeError("non-finite parameters after LBFGS")

    # Retrieve n_iter from optimizer state? LBFGS stores func_evals; use param group
    n_iter = optimizer.state_dict()["param_groups"][0].get("n_iter", max_iter)  # type: ignore
    # Not reliable across torch versions; estimate from state
    try:
        state_w = optimizer.state[w]
        n_iter_val = int(state_w.get("func_evals", max_iter))  # type: ignore
    except Exception:
        n_iter_val = max_iter

    converged = grad_norm <= grad_threshold
    if not converged:
        raise RuntimeError(
            f"LBFGS did not converge: grad_norm {grad_norm:.3e} > threshold {grad_threshold:.3e} "
            f"(objective {objective:.3e}, n_iter {n_iter_val})"
        )

    # Check w,b finite already
    return w.detach(), float(b.detach().item()), float(objective), float(grad_norm), int(n_iter_val), bool(converged)


def _compute_metrics(
    logits_test: torch.Tensor,
    y_test_bin: torch.Tensor,
) -> SparseProbeMetrics:
    preds = (logits_test >= 0).to(torch.int64)
    y_true = y_test_bin.to(torch.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum().item())
    tn = int(((preds == 0) & (y_true == 0)).sum().item())
    fp = int(((preds == 1) & (y_true == 0)).sum().item())
    fn = int(((preds == 0) & (y_true == 1)).sum().item())
    n = y_true.shape[0]
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return SparseProbeMetrics(accuracy=acc, precision=prec, recall=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _prepare_split(
    X: torch.Tensor,
    y_int: torch.Tensor,
    test_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], Dict[int, int], torch.Generator]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_indices, test_indices, n_train_per_class, n_test_per_class = _stratified_split(
        y_int, test_fraction, gen
    )
    return train_indices, test_indices, n_train_per_class, n_test_per_class, gen


def _validate_preprocess(p: str) -> None:
    if p not in ("none", "standardize"):
        raise ValueError(f'preprocess must be "none" or "standardize", got {p!r}')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_sparse_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    k: int,
    test_fraction: float = 0.3,
    positive_label: int = 1,
    preprocess: str = "none",
    l2_strength: float = 1e-2,
    seed: int = 0,
    grad_threshold: float = 1e-4,
    max_iter: int = 100,
) -> SparseProbeResult:
    """Fit a leakage-safe k-sparse probe.

    Args:
        X: Activation matrix ``[n, d]`` floating and finite.
        y: Label vector ``[n]`` Boolean or integer with exactly two values.
        k: Number of coordinates to select (``1 <= k <= d``).
        test_fraction: Fraction of each class to hold out.
        positive_label: Value in ``y`` that maps to the positive class.
        preprocess: ``"none"`` (default) or ``"standardize"`` (train-only
            centred scaling with population std; zero std maps to scale 1).
        l2_strength: Positive L2 coefficient (``l2 * ||w||^2 / 2``).
        seed: Deterministic seed for the stratified split.
        grad_threshold: Final gradient infinity-norm convergence threshold.
        max_iter: LBFGS max iterations.

    Returns:
        :class:`SparseProbeResult` with selection, fit, and held-out metrics.

    Raises:
        ValueError / TypeError: on invalid inputs.
        RuntimeError: if LBFGS does not converge or produces non-finite output.
    """
    X, y = _validate_X_y(X, y)
    n, d = X.shape
    _validate_k(k, d)
    _validate_test_fraction(test_fraction)
    _validate_l2(l2_strength)
    _validate_preprocess(preprocess)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be int, got {type(seed)}")
    if not isinstance(grad_threshold, (float, int)) or isinstance(grad_threshold, bool):
        raise TypeError(f"grad_threshold must be float, got {type(grad_threshold)}")
    if float(grad_threshold) <= 0:
        raise ValueError(f"grad_threshold must be positive, got {grad_threshold}")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError(f"max_iter must be positive int, got {max_iter}")

    labels, y_int = _validate_labels(y, positive_label)
    _check_class_counts(y_int)
    # Determine positive int value for internal use
    if y.dtype == torch.bool:
        pos_int = int(bool(positive_label)) if isinstance(positive_label, bool) else int(positive_label)
    else:
        # y_int already mapped; find which label equals positive_label
        pos_int = int(positive_label) if not isinstance(positive_label, bool) else int(positive_label)
        # Ensure pos_int is one of the two labels
        if pos_int not in labels:
            # try exact match for original positive_label (e.g., if y has 2/3)
            for lab in labels:
                if lab == positive_label:
                    pos_int = lab
                    break

    train_indices, test_indices, n_train_per_class, n_test_per_class, gen = _prepare_split(
        X, y_int, float(test_fraction), seed
    )

    n_train = int(train_indices.shape[0])
    n_test = int(test_indices.shape[0])
    class_counts = {int(c): int((y_int == c).sum().item()) for c in labels}

    # Slice
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train_int = y_int[train_indices]
    y_test_int = y_int[test_indices]

    # Selection on train only
    scores = _mean_difference_scores(X_train, y_train_int, pos_int)
    selected_indices, selected_scores = _select_top_k(scores, k)

    # Selected matrices
    sel_tensor = torch.tensor(selected_indices, dtype=torch.long, device=X.device)
    X_train_sel = X_train[:, sel_tensor]
    X_test_sel = X_test[:, sel_tensor]

    # Preprocessing
    preprocess_mean = None
    preprocess_scale = None
    if preprocess == "standardize":
        # train-only mean and population std
        mean = X_train_sel.mean(dim=0)  # [k]
        # population std: unbiased=False
        std = X_train_sel.std(dim=0, unbiased=False)
        # map zero std to 1
        scale = torch.where(std == 0, torch.ones_like(std), std)
        # also handle near-zero? spec says zero std -> scale 1 exactly
        preprocess_mean = mean.detach().cpu()
        preprocess_scale = scale.detach().cpu()
        # apply
        X_train_sel = (X_train_sel - mean) / scale
        X_test_sel = (X_test_sel - mean.to(X_test_sel.device)) / scale.to(X_test_sel.device)
    elif preprocess != "none":
        raise ValueError(f"unknown preprocess {preprocess!r}")

    # Binary labels 0/1
    y_train_bin = (y_train_int == pos_int).to(torch.int64)
    y_test_bin = (y_test_int == pos_int).to(torch.int64)

    # Class-balanced weights n/(2 n_c) computed on train counts
    total_n = n_train
    class_weights: Dict[int, float] = {}
    for lab in labels:
        n_c = int((y_train_int == lab).sum().item())
        class_weights[int(lab)] = total_n / (2.0 * n_c)

    # Fit
    # Note: fit uses selected matrices (possibly standardized)
    w, b, objective, grad_norm, n_iter, converged = _fit_logistic_lbfgs(
        X_train_sel,
        y_train_bin,
        X_test_sel,
        float(l2_strength),
        class_weights,
        pos_int,
        y_train_int,
        float(grad_threshold),
        max_iter=max_iter,
    )

    # Evaluate held-out
    # Need to compute logits on test set with fitted w,b (CPU float64 then back)
    X_test_sel_cpu = X_test_sel.to(dtype=torch.float64, device=torch.device("cpu"))
    w_cpu = w.to(dtype=torch.float64)
    logits_test = X_test_sel_cpu @ w_cpu + b
    metrics = _compute_metrics(logits_test, y_test_bin.to(dtype=torch.float64))

    # Build result; keep indices sorted as selected order (ranked)
    result = SparseProbeResult(
        k=k,
        test_fraction=float(test_fraction),
        positive_label=int(pos_int),
        preprocess=preprocess,
        l2_strength=float(l2_strength),
        seed=int(seed),
        labels=labels,
        class_counts=class_counts,
        n_train=n_train,
        n_test=n_test,
        n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class,
        train_indices=train_indices.detach().cpu(),
        test_indices=test_indices.detach().cpu(),
        selected_indices=selected_indices,
        selected_scores=selected_scores,
        coefficients=w.detach().cpu(),
        intercept=float(b),
        objective=float(objective),
        grad_norm=float(grad_norm),
        n_iter=int(n_iter),
        converged=bool(converged),
        preprocess_mean=preprocess_mean,
        preprocess_scale=preprocess_scale,
        metrics=metrics,
        logits_test=logits_test.detach().cpu(),
        preds_test=(logits_test >= 0).to(torch.int64).detach().cpu(),
    )
    return result


def sweep_sparse_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    ks: List[int],
    test_fraction: float = 0.3,
    positive_label: int = 1,
    preprocess: str = "none",
    l2_strength: float = 1e-2,
    n_random_subsets: int = 20,
    n_label_shuffles: int = 20,
    seed: int = 0,
    grad_threshold: float = 1e-4,
    max_iter: int = 100,
) -> SparseSweepResult:
    """Sweep k-sparse probes over a fixed split with controls.

    Reuses the *exact* stratified split across every ``k`` in ``ks``. Control
    distributions are deterministic for a given ``seed`` and use a local
    ``torch.Generator`` without mutating global RNG state.

    Args:
        X, y, test_fraction, positive_label, preprocess, l2_strength, seed:
            As in :func:`fit_sparse_probe`.
        ks: Strictly increasing unique list of ``k`` values.
        n_random_subsets: Number of random-coordinate controls per ``k`` (0 disables).
        n_label_shuffles: Number of label-shuffle controls per ``k`` (0 disables).
        grad_threshold, max_iter: Passed to each probe fit.

    Returns:
        :class:`SparseSweepResult`.

    Raises:
        ValueError: on invalid ``ks``, non-strictly-increasing duplicates, or
            invalid control counts.
    """
    X, y = _validate_X_y(X, y)
    n, d = X.shape
    _validate_test_fraction(test_fraction)
    _validate_l2(l2_strength)
    _validate_preprocess(preprocess)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be int, got {type(seed)}")
    if not isinstance(ks, (list, tuple)):
        raise TypeError(f"ks must be list of ints, got {type(ks)}")
    if len(ks) == 0:
        raise ValueError("ks must be non-empty")
    ks_list = list(ks)
    for v in ks_list:
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"ks entries must be ints, got {v!r}")
        _validate_k(v, d)
    # strictly increasing unique
    if sorted(set(ks_list)) != ks_list:
        raise ValueError(f"ks must be strictly increasing unique, got {ks_list}")
    if len(set(ks_list)) != len(ks_list):
        raise ValueError(f"ks must not contain duplicates, got {ks_list}")
    if not isinstance(n_random_subsets, int) or isinstance(n_random_subsets, bool) or n_random_subsets < 0:
        raise ValueError(f"n_random_subsets must be nonnegative int, got {n_random_subsets}")
    if not isinstance(n_label_shuffles, int) or isinstance(n_label_shuffles, bool) or n_label_shuffles < 0:
        raise ValueError(f"n_label_shuffles must be nonnegative int, got {n_label_shuffles}")

    labels, y_int = _validate_labels(y, positive_label)
    _check_class_counts(y_int)

    # Resolve positive int
    if y.dtype == torch.bool:
        pos_int = int(bool(positive_label)) if isinstance(positive_label, bool) else int(positive_label)
    else:
        pos_int = int(positive_label) if not isinstance(positive_label, bool) else int(positive_label)
        if pos_int not in labels:
            for lab in labels:
                if lab == positive_label:
                    pos_int = lab
                    break

    # Fixed split once
    train_indices, test_indices, n_train_per_class, n_test_per_class, gen = _prepare_split(
        X, y_int, float(test_fraction), seed
    )
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train_int = y_int[train_indices]
    y_test_int = y_int[test_indices]

    # Selection scores once per probe? No, per k selection differs but scores computed once
    # Compute raw scores once (train-only)
    scores = _mean_difference_scores(X_train, y_train_int, pos_int)

    probes: List[SparseProbeResult] = []
    random_controls: List[List[SparseProbeResult]] = []
    label_shuffle_controls: List[List[SparseProbeResult]] = []

    for k in ks_list:
        # Fit main probe for this k (reuse split, recompute selection)
        selected_indices, selected_scores = _select_top_k(scores, k)
        sel_tensor = torch.tensor(selected_indices, dtype=torch.long, device=X.device)
        Xtr_sel = X_train[:, sel_tensor]
        Xte_sel = X_test[:, sel_tensor]

        preprocess_mean = None
        preprocess_scale = None
        if preprocess == "standardize":
            mean = Xtr_sel.mean(dim=0)
            std = Xtr_sel.std(dim=0, unbiased=False)
            scale = torch.where(std == 0, torch.ones_like(std), std)
            preprocess_mean = mean.detach().cpu()
            preprocess_scale = scale.detach().cpu()
            Xtr_sel = (Xtr_sel - mean) / scale
            Xte_sel = (Xte_sel - mean.to(Xte_sel.device)) / scale.to(Xte_sel.device)

        y_train_bin = (y_train_int == pos_int).to(torch.int64)
        y_test_bin = (y_test_int == pos_int).to(torch.int64)
        total_n = int(train_indices.shape[0])
        class_weights = {int(lab): total_n / (2.0 * int((y_train_int == lab).sum().item())) for lab in labels}

        w, b, objective, grad_norm, n_iter, converged = _fit_logistic_lbfgs(
            Xtr_sel,
            y_train_bin,
            Xte_sel,
            float(l2_strength),
            class_weights,
            pos_int,
            y_train_int,
            float(grad_threshold),
            max_iter=max_iter,
        )
        Xte_cpu = Xte_sel.to(dtype=torch.float64, device=torch.device("cpu"))
        logits_test = Xte_cpu @ w.to(dtype=torch.float64) + b
        metrics = _compute_metrics(logits_test, y_test_bin.to(dtype=torch.float64))
        class_counts = {int(c): int((y_int == c).sum().item()) for c in labels}
        probe = SparseProbeResult(
            k=k,
            test_fraction=float(test_fraction),
            positive_label=int(pos_int),
            preprocess=preprocess,
            l2_strength=float(l2_strength),
            seed=int(seed),
            labels=labels,
            class_counts=class_counts,
            n_train=int(train_indices.shape[0]),
            n_test=int(test_indices.shape[0]),
            n_train_per_class=dict(n_train_per_class),
            n_test_per_class=dict(n_test_per_class),
            train_indices=train_indices.detach().cpu(),
            test_indices=test_indices.detach().cpu(),
            selected_indices=selected_indices,
            selected_scores=selected_scores,
            coefficients=w.detach().cpu(),
            intercept=float(b),
            objective=float(objective),
            grad_norm=float(grad_norm),
            n_iter=int(n_iter),
            converged=bool(converged),
            preprocess_mean=preprocess_mean,
            preprocess_scale=preprocess_scale,
            metrics=metrics,
            logits_test=logits_test.detach().cpu(),
            preds_test=(logits_test >= 0).to(torch.int64).detach().cpu(),
        )
        probes.append(probe)

        # Random-coordinate controls
        rand_list: List[SparseProbeResult] = []
        for _ in range(n_random_subsets):
            perm = torch.randperm(d, generator=gen)
            rand_sel = perm[:k].tolist()
            # Need scores for these rand indices? Use zero? But we report selected_scores as raw scores for those indices
            rand_scores = scores[torch.tensor(rand_sel, device=scores.device)].tolist()
            sel_t = torch.tensor(rand_sel, dtype=torch.long, device=X.device)
            Xr_tr = X_train[:, sel_t]
            Xr_te = X_test[:, sel_t]
            rm = None
            rs = None
            if preprocess == "standardize":
                m = Xr_tr.mean(dim=0)
                s = Xr_tr.std(dim=0, unbiased=False)
                sc = torch.where(s == 0, torch.ones_like(s), s)
                rm = m.detach().cpu()
                rs = sc.detach().cpu()
                Xr_tr = (Xr_tr - m) / sc
                Xr_te = (Xr_te - m.to(Xr_te.device)) / sc.to(Xr_te.device)
            w2, b2, obj2, gn2, ni2, cv2 = _fit_logistic_lbfgs(
                Xr_tr, y_train_bin, Xr_te, float(l2_strength), class_weights, pos_int, y_train_int, float(grad_threshold), max_iter=max_iter
            )
            Xte2 = Xr_te.to(dtype=torch.float64, device=torch.device("cpu"))
            logits2 = Xte2 @ w2.to(dtype=torch.float64) + b2
            metrics2 = _compute_metrics(logits2, y_test_bin.to(dtype=torch.float64))
            rand_list.append(
                SparseProbeResult(
                    k=k,
                    test_fraction=float(test_fraction),
                    positive_label=int(pos_int),
                    preprocess=preprocess,
                    l2_strength=float(l2_strength),
                    seed=int(seed),
                    labels=labels,
                    class_counts=class_counts,
                    n_train=int(train_indices.shape[0]),
                    n_test=int(test_indices.shape[0]),
                    n_train_per_class=dict(n_train_per_class),
                    n_test_per_class=dict(n_test_per_class),
                    train_indices=train_indices.detach().cpu(),
                    test_indices=test_indices.detach().cpu(),
                    selected_indices=rand_sel,
                    selected_scores=rand_scores,
                    coefficients=w2.detach().cpu(),
                    intercept=float(b2),
                    objective=float(obj2),
                    grad_norm=float(gn2),
                    n_iter=int(ni2),
                    converged=bool(cv2),
                    preprocess_mean=rm,
                    preprocess_scale=rs,
                    metrics=metrics2,
                    logits_test=logits2.detach().cpu(),
                    preds_test=(logits2 >= 0).to(torch.int64).detach().cpu(),
                )
            )
        random_controls.append(rand_list)

        # Label-shuffle controls: permute training labels, repeat selection/fitting, evaluate against untouched test labels
        shuf_list: List[SparseProbeResult] = []
        for _ in range(n_label_shuffles):
            perm = torch.randperm(y_train_int.shape[0], generator=gen)
            y_shuf = y_train_int[perm]
            # Recompute scores using shuffled labels
            scores_shuf = _mean_difference_scores(X_train, y_shuf, pos_int)
            sel_shuf, scores_shuf_vals = _select_top_k(scores_shuf, k)
            sel_t = torch.tensor(sel_shuf, dtype=torch.long, device=X.device)
            Xs_tr = X_train[:, sel_t]
            Xs_te = X_test[:, sel_t]
            rm = None
            rs = None
            if preprocess == "standardize":
                m = Xs_tr.mean(dim=0)
                s = Xs_tr.std(dim=0, unbiased=False)
                sc = torch.where(s == 0, torch.ones_like(s), s)
                rm = m.detach().cpu()
                rs = sc.detach().cpu()
                Xs_tr = (Xs_tr - m) / sc
                Xs_te = (Xs_te - m.to(Xs_te.device)) / sc.to(Xs_te.device)
            y_shuf_bin = (y_shuf == pos_int).to(torch.int64)
            # class weights based on shuffled train distribution
            cw_shuf = {int(lab): total_n / (2.0 * int((y_shuf == lab).sum().item())) for lab in labels}
            # If a class disappears after shuffle (unlikely with stratification but possible), skip?
            # y_shuf is permuted, so counts preserved; safe.
            w3, b3, obj3, gn3, ni3, cv3 = _fit_logistic_lbfgs(
                Xs_tr, y_shuf_bin, Xs_te, float(l2_strength), cw_shuf, pos_int, y_shuf, float(grad_threshold), max_iter=max_iter
            )
            Xte3 = Xs_te.to(dtype=torch.float64, device=torch.device("cpu"))
            logits3 = Xte3 @ w3.to(dtype=torch.float64) + b3
            # Evaluate against *untouched* test labels
            metrics3 = _compute_metrics(logits3, y_test_bin.to(dtype=torch.float64))
            shuf_list.append(
                SparseProbeResult(
                    k=k,
                    test_fraction=float(test_fraction),
                    positive_label=int(pos_int),
                    preprocess=preprocess,
                    l2_strength=float(l2_strength),
                    seed=int(seed),
                    labels=labels,
                    class_counts=class_counts,
                    n_train=int(train_indices.shape[0]),
                    n_test=int(test_indices.shape[0]),
                    n_train_per_class=dict(n_train_per_class),
                    n_test_per_class=dict(n_test_per_class),
                    train_indices=train_indices.detach().cpu(),
                    test_indices=test_indices.detach().cpu(),
                    selected_indices=sel_shuf,
                    selected_scores=scores_shuf_vals,
                    coefficients=w3.detach().cpu(),
                    intercept=float(b3),
                    objective=float(obj3),
                    grad_norm=float(gn3),
                    n_iter=int(ni3),
                    converged=bool(cv3),
                    preprocess_mean=rm,
                    preprocess_scale=rs,
                    metrics=metrics3,
                    logits_test=logits3.detach().cpu(),
                    preds_test=(logits3 >= 0).to(torch.int64).detach().cpu(),
                )
            )
        label_shuffle_controls.append(shuf_list)

    return SparseSweepResult(
        ks=ks_list,
        test_fraction=float(test_fraction),
        positive_label=int(pos_int),
        preprocess=preprocess,
        l2_strength=float(l2_strength),
        seed=int(seed),
        probes=probes,
        random_controls=random_controls,
        label_shuffle_controls=label_shuffle_controls,
        train_indices=train_indices.detach().cpu(),
        test_indices=test_indices.detach().cpu(),
    )
