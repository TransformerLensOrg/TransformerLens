"""Model-free statistics over a per-example axis of an analysis result.

No model access: standard errors, a vectorized percentile bootstrap, a paired
sign-flip permutation test and an exact Clopper-Pearson interval for success
rates, plus the seeded-generator convention shared with sparse probing so every
consumer derives independent, reproducible streams the same way.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
    binary success rates prefer an exact interval (:func:`success_rate_ci`).
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
    draws = torch.cat(stats, dim=-1).to(torch.promote_types(x.dtype, torch.float32))
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
    moved = _to_last(effects, dim).to(torch.promote_types(effects.dtype, torch.float32))
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
    # Relative to the observed statistic: tolerate reduction-order ties without an absolute floor.
    tolerance = observed.abs() * (100 * torch.finfo(null.dtype).eps)
    if alternative == "two-sided":
        hit = null.abs() >= observed.abs() - tolerance
    elif alternative == "greater":
        hit = null >= observed - tolerance
    else:
        hit = null <= observed + tolerance
    return hit.sum(dim=-1).to(null.dtype)


@dataclass(frozen=True)
class SuccessRateInterval:
    """An exact Clopper-Pearson confidence interval around a success rate.

    ``n_trials`` and ``n_successes`` are the counts the interval was computed from, so a
    reader can see the denominator behind the bounds rather than inferring it.
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    n_trials: int
    n_successes: int
    confidence: float


# Each bisection step halves the bracket, so 200 steps resolve the bound far below the
# precision any reported rate needs.
_CLOPPER_PEARSON_STEPS = 200


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact ``P(X <= k)`` for ``X ~ Binomial(n, p)``.

    Works with each term's logarithm (``log comb(n, i) + i log p + (n - i) log(1 - p)``) and
    sums ``exp(log_term - peak)`` before scaling back by ``exp(peak)``. Forming ``comb(n, i)``
    and ``p ** i`` directly overflows at large ``n``, and starting a pmf recurrence from
    ``(1 - p) ** n`` underflows to ``0.0`` once ``n log(1 - p) < -745``.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n_factorial = math.lgamma(n + 1)
    log_terms = [
        log_n_factorial - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    peak = max(log_terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(term - peak) for term in log_terms))


def _clopper_pearson_bound(n_successes: int, n_trials: int, tail: float, *, upper: bool) -> float:
    """Solves one Clopper-Pearson bound by bisecting the exact binomial CDF.

    The lower bound solves ``P(X >= k) = tail`` and the upper bound solves ``P(X <= k) =
    tail``. ``P(X >= k)`` rises with ``p`` while ``P(X <= k)`` falls, so the two bounds
    bracket from opposite sides; reversing that collapses every interval to ``[0, 1]``.
    """
    if upper and n_successes == n_trials:
        return 1.0
    if not upper and n_successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(_CLOPPER_PEARSON_STEPS):
        midpoint = (low + high) / 2.0
        if upper:
            # P(X <= k) decreases in p, so a probability above the tail puts the bound below
            # the midpoint.
            if _binomial_cdf(n_successes, n_trials, midpoint) > tail:
                low = midpoint
            else:
                high = midpoint
        else:
            # P(X >= k) increases in p, so a probability above the tail puts the bound above
            # the midpoint.
            if 1.0 - _binomial_cdf(n_successes - 1, n_trials, midpoint) > tail:
                high = midpoint
            else:
                low = midpoint
    return (low + high) / 2.0


def success_rate_ci(
    successes: Sequence[bool],
    *,
    confidence: float = 0.95,
) -> SuccessRateInterval:
    """Computes an exact Clopper-Pearson confidence interval for a success rate.

    A percentile bootstrap cannot express uncertainty about an all-failure or all-success
    sample: every resample of an all-zero vector is all zeros, so the interval collapses to
    ``[0, 0]`` and reports the same certainty for one trial as for a thousand. The exact
    interval instead widens as the trial count shrinks, giving ``[0.0, 0.459]`` for zero
    successes out of six.

    Deterministic by construction: there is no resampling and no seed.

    Raises:
        ValueError: If ``successes`` is empty.
    """
    if len(successes) == 0:
        raise ValueError("successes must be a non-empty sequence")
    n_trials = len(successes)
    n_successes = sum(1 for success in successes if success)
    tail = (1.0 - confidence) / 2.0
    return SuccessRateInterval(
        point_estimate=n_successes / n_trials,
        ci_low=_clopper_pearson_bound(n_successes, n_trials, tail, upper=False),
        ci_high=_clopper_pearson_bound(n_successes, n_trials, tail, upper=True),
        n_trials=n_trials,
        n_successes=n_successes,
        confidence=confidence,
    )
