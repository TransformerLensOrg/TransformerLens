"""Relevancy scoring for unsupported architectures.

Computes a composite relevancy score (0-100) for each architecture gap,
combining demand (model count), usage (downloads), benchmarkability
(smallest model size), and authorship (published by a preferred author).

Formula:
    relevancy = 0.95 * (0.45 * demand + 0.35 * usage + 0.20 * benchmarkability)
              + 0.05 * authorship
"""

import math
from typing import Optional

from .authorship import architecture_authors, is_preferred_author

# 5 points is worth roughly one order of magnitude of downloads. On the 2026-11-24
# Hub scan (613 gaps) that left authorship the least influential component while
# still moving 8 architectures into the top 50; from 0.08 up, preferred-author
# architectures with under 10k downloads started displacing well-used community ones.
WEIGHT_AUTHORSHIP = 0.05

# The other three keep their 45:35:20 ratio, so authorship can never reorder two
# architectures that share an authorship status.
WEIGHT_DEMAND = 0.45 * (1 - WEIGHT_AUTHORSHIP)
WEIGHT_USAGE = 0.35 * (1 - WEIGHT_AUTHORSHIP)
WEIGHT_BENCHMARKABILITY = 0.20 * (1 - WEIGHT_AUTHORSHIP)


def _normalize_demand(model_count: int, max_model_count: int) -> float:
    """Normalize model count to 0-100 scale.

    Args:
        model_count: Number of models for this architecture.
        max_model_count: Maximum model count across all architectures.

    Returns:
        Normalized demand score (0-100).
    """
    if max_model_count <= 0:
        return 0.0
    return min(model_count / max_model_count * 100, 100.0)


def _normalize_usage(total_downloads: int, max_downloads: int) -> float:
    """Normalize download count to 0-100 using log scale.

    Log scale prevents mega-popular models from completely dominating.

    Args:
        total_downloads: Total downloads for this architecture.
        max_downloads: Maximum total downloads across all architectures.

    Returns:
        Normalized usage score (0-100).
    """
    if max_downloads <= 0 or total_downloads <= 0:
        return 0.0
    return min(
        math.log10(total_downloads + 1) / math.log10(max_downloads + 1) * 100,
        100.0,
    )


def _score_benchmarkability(min_param_count: Optional[int]) -> float:
    """Score benchmarkability based on smallest available model size.

    Args:
        min_param_count: Parameter count of the smallest model, or None if unknown.

    Returns:
        Benchmarkability score (0-100).
    """
    if min_param_count is None:
        return 0.0
    if min_param_count <= 1_000_000_000:
        return 100.0
    if min_param_count <= 3_000_000_000:
        return 80.0
    if min_param_count <= 7_000_000_000:
        return 60.0
    if min_param_count <= 14_000_000_000:
        return 40.0
    if min_param_count <= 30_000_000_000:
        return 20.0
    return 0.0


def _score_authorship(preferred_author: bool) -> float:
    """Score authorship: 100 when a preferred author published the architecture, else 0."""
    return 100.0 if preferred_author else 0.0


def compute_relevancy_score(
    model_count: int,
    total_downloads: int,
    min_param_count: Optional[int],
    max_model_count: int,
    max_downloads: int,
    preferred_author: bool = False,
) -> float:
    """Compute composite relevancy score for an architecture gap.

    Args:
        model_count: Number of models using this architecture.
        total_downloads: Aggregate downloads across all models of this architecture.
        min_param_count: Parameter count of the smallest model (None if unknown).
        max_model_count: Max model count across all gap architectures (for normalization).
        max_downloads: Max total downloads across all gap architectures (for normalization).
        preferred_author: Whether one of the architecture's authors is a preferred author.

    Returns:
        Relevancy score from 0 to 100.
    """
    demand = _normalize_demand(model_count, max_model_count)
    usage = _normalize_usage(total_downloads, max_downloads)
    benchmarkability = _score_benchmarkability(min_param_count)
    authorship = _score_authorship(preferred_author)

    score = (
        WEIGHT_DEMAND * demand
        + WEIGHT_USAGE * usage
        + WEIGHT_BENCHMARKABILITY * benchmarkability
        + WEIGHT_AUTHORSHIP * authorship
    )

    return round(score, 1)


def compute_scores_for_gaps(gaps: list[dict]) -> list[dict]:
    """Compute relevancy scores for a list of architecture gap dicts.

    Mutates each gap dict in-place by adding 'authors', 'preferred_author' and
    'relevancy_score' fields, then returns the list sorted by score descending.

    Args:
        gaps: List of gap dicts with 'architecture_id', 'total_models',
              'total_downloads', and 'min_param_count' fields. Authors come from
              'publishers', falling back to the first of 'sample_models'.

    Returns:
        The same list, sorted by relevancy_score descending (total_models as tiebreaker).
    """
    max_model_count = max((g.get("total_models", 0) for g in gaps), default=0)
    max_downloads = max((g.get("total_downloads", 0) for g in gaps), default=0)

    for gap in gaps:
        gap["authors"] = architecture_authors(gap)
        gap["preferred_author"] = any(is_preferred_author(a) for a in gap["authors"])
        gap["relevancy_score"] = compute_relevancy_score(
            model_count=gap.get("total_models", 0),
            total_downloads=gap.get("total_downloads", 0),
            min_param_count=gap.get("min_param_count"),
            max_model_count=max_model_count,
            max_downloads=max_downloads,
            preferred_author=gap["preferred_author"],
        )

    gaps.sort(key=lambda g: (-g["relevancy_score"], -g.get("total_models", 0)))
    return gaps
