"""Model-free statistics over a per-example axis of an analysis result.

Pure torch, no model access: standard errors, a vectorized percentile bootstrap
and a paired sign-flip permutation test, plus the seeded-generator convention
shared with sparse probing so every consumer derives independent, reproducible
streams the same way.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from typing import Optional, Tuple

import torch

_ALTERNATIVES = ("two-sided", "greater", "less")
_STATISTICS = ("mean", "median")


def derive_generator(seed: int, *labels: object) -> torch.Generator:
    """CPU generator keyed on ``seed`` and any labels (metric name, arm, repeat...).

    SHA-256 rather than ``hash`` keeps the derived seed stable across processes;
    the mask keeps it in torch's accepted int64 range.
    """
    key = ":".join([str(seed), *(str(label) for label in labels)]).encode()
    derived = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF
    return torch.Generator(device="cpu").manual_seed(derived)


def _to_last(x: torch.Tensor, dim: int) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("expected a floating-point tensor")
    if x.ndim == 0:
        raise ValueError("expected at least one dimension to resample over")
    return x.movedim(dim, -1)


def standard_error(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Standard error of the mean along ``dim`` (unbiased std over sqrt(n)); needs n >= 2."""
    moved = _to_last(x, dim)
    n = moved.shape[-1]
    if n < 2:
        raise ValueError(f"standard_error needs at least 2 samples along dim {dim}, got {n}")
    return moved.std(dim=-1, unbiased=True) / math.sqrt(n)


def bootstrap_ci(
    x: torch.Tensor,
    *,
    dim: int = -1,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_resamples: int = 1000,
    generator: Optional[torch.Generator] = None,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Percentile bootstrap interval of ``statistic`` along ``dim``, vectorized over the rest.

    Resamples with replacement using a CPU ``generator`` (``derive_generator(0)``
    when omitted), ``chunk_size`` resamples at a time so memory stays at
    ``chunk_size × n`` per leading cell. Returns ``(low, high)`` with ``dim``
    removed.

    A percentile bootstrap cannot express uncertainty a sample does not contain:
    a constant vector resamples to itself and the interval collapses to a point
    whatever ``n`` is, and intervals on 4-8 samples are wide and skewed. For
    binary success rates prefer an exact interval
    (:func:`~transformer_lens.tools.analysis.jacobian_lens_causal_swap_benchmark.success_rate_ci`).
    """
    if statistic not in _STATISTICS:
        raise ValueError(f"statistic must be one of {_STATISTICS}, got {statistic!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1 or chunk_size < 1:
        raise ValueError("n_resamples and chunk_size must be >= 1")
    moved = _to_last(x, dim)
    n = moved.shape[-1]
    if n < 1:
        raise ValueError("cannot bootstrap an empty axis")
    gen = generator if generator is not None else derive_generator(0)
    reduce = torch.mean if statistic == "mean" else lambda t, dim: t.median(dim=dim).values
    stats = []
    for start in range(0, n_resamples, chunk_size):
        count = min(chunk_size, n_resamples - start)
        idx = torch.randint(0, n, (count, n), generator=gen).to(moved.device)
        resampled = moved[..., idx]  # [..., count, n]
        stats.append(reduce(resampled, dim=-1))
    draws = torch.cat(stats, dim=-1).float()  # [..., n_resamples]
    alpha = (1.0 - confidence) / 2.0
    low = torch.quantile(draws, alpha, dim=-1)
    high = torch.quantile(draws, 1.0 - alpha, dim=-1)
    return low.to(x.dtype), high.to(x.dtype)


def sign_flip_permutation_pvalue(
    effects: torch.Tensor,
    *,
    dim: int = -1,
    n_permutations: int = 1000,
    generator: Optional[torch.Generator] = None,
    alternative: str = "two-sided",
    exact_max_n: int = 12,
) -> torch.Tensor:
    """Paired sign-flip permutation p-value that the mean effect along ``dim`` is zero.

    The null is that each per-example effect is symmetric about zero, so every
    pattern of sign flips is equally likely. With ``n <= exact_max_n`` examples
    all ``2**n`` patterns are enumerated and the p-value is exact and
    deterministic; otherwise ``n_permutations`` random patterns are drawn from
    the CPU ``generator`` and the p-value is ``(hits + 1) / (n_permutations + 1)``.
    ``alternative`` is ``"two-sided"`` (``|null| >= |observed|``), ``"greater"``
    or ``"less"``. Returns a tensor with ``dim`` removed.
    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    moved = _to_last(effects, dim).float()
    n = moved.shape[-1]
    if n < 1:
        raise ValueError("cannot permute an empty axis")
    observed = moved.mean(dim=-1, keepdim=True)  # [..., 1]
    if n <= exact_max_n:
        signs = torch.tensor(
            list(itertools.product((1.0, -1.0), repeat=n)), dtype=moved.dtype, device=moved.device
        )  # [2^n, n]
        null = (moved.unsqueeze(-2) * signs).mean(dim=-1)  # [..., 2^n]
        hits = _count_hits(null, observed, alternative)
        return hits / signs.shape[0]
    gen = generator if generator is not None else derive_generator(0)
    hits_total = torch.zeros(moved.shape[:-1], dtype=moved.dtype, device=moved.device)
    chunk = 256
    for start in range(0, n_permutations, chunk):
        count = min(chunk, n_permutations - start)
        signs = (
            torch.randint(0, 2, (count, n), generator=gen).to(moved.device).to(moved.dtype) * 2 - 1
        )
        null = (moved.unsqueeze(-2) * signs).mean(dim=-1)
        hits_total = hits_total + _count_hits(null, observed, alternative)
    return (hits_total + 1.0) / (n_permutations + 1.0)


def _count_hits(null: torch.Tensor, observed: torch.Tensor, alternative: str) -> torch.Tensor:
    if alternative == "two-sided":
        hit = null.abs() >= observed.abs()
    elif alternative == "greater":
        hit = null >= observed
    else:
        hit = null <= observed
    return hit.sum(dim=-1).to(null.dtype)
