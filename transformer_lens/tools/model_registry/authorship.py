"""Authorship of unsupported architectures, and organizations that look like preferred authors.

An architecture's publishers are the Hub namespaces hosting its checkpoints. Its authors
are the publisher with the most downloads and the publisher with the oldest repo. The two
usually coincide; when they differ either can be the originator (a popular re-upload can
lead downloads; a mirror, or a repo created empty and filled later, can predate the
official one), so both are credited.

Organization gaps are non-preferred authors whose footprint in the architectures they
author matches the typical preferred author's: at least the median preferred author's
downloads and likes. Downloads alone admit automated pulls; likes alone admit hype.
"""

import statistics
from datetime import date, datetime, timezone
from typing import Iterable, Optional

from . import PREFERRED_AUTHORS

# Per-architecture publisher records kept in architecture_gaps.json.
MAX_PUBLISHERS = 5


def namespace_of(model_id: str) -> Optional[str]:
    """Hub namespace of a repo id, or None for a legacy un-namespaced id."""
    namespace, sep, _ = model_id.partition("/")
    return namespace if sep and namespace else None


def is_preferred_author(author: str, preferred: Optional[Iterable[str]] = None) -> bool:
    """Whether a Hub namespace is a preferred author (case-insensitive)."""
    names = PREFERRED_AUTHORS if preferred is None else preferred
    return author.casefold() in {name.casefold() for name in names}


def record_publisher(
    publishers: dict[str, dict],
    model_id: str,
    downloads: int,
    likes: int,
    created_at: Optional[datetime],
) -> None:
    """Fold one scanned model into its architecture's per-publisher tallies."""
    name = namespace_of(model_id)
    if name is None:
        return
    record = publishers.setdefault(
        name, {"name": name, "models": 0, "downloads": 0, "likes": 0, "first_published": None}
    )
    record["models"] += 1
    record["downloads"] += downloads
    record["likes"] += likes
    if created_at is not None:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        # One fixed UTC format so the strings order chronologically.
        stamp = created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if record["first_published"] is None or stamp < record["first_published"]:
            record["first_published"] = stamp


def _earliest(publishers: Iterable[dict]) -> Optional[dict]:
    dated = [p for p in publishers if p.get("first_published")]
    if not dated:
        return None
    return min(dated, key=lambda p: (p["first_published"], -p["downloads"], p["name"]))


def top_publishers(publishers: Iterable[dict]) -> list[dict]:
    """The MAX_PUBLISHERS most-downloaded publishers, plus the earliest one."""
    ranked = sorted(publishers, key=lambda p: (-p["downloads"], -p["models"], p["name"]))
    kept = ranked[:MAX_PUBLISHERS]
    first = _earliest(ranked)
    if first is not None and first not in kept:
        # Dropping it would lose one of the architecture's two authors.
        kept.append(first)
    return kept


def merge_publishers(old: Iterable[dict], new: Iterable[dict]) -> list[dict]:
    """Combine publisher records from two scrape passes of one architecture."""
    merged: dict[str, dict] = {}
    for record in [*old, *new]:
        into = merged.get(record["name"])
        if into is None:
            merged[record["name"]] = dict(record)
            continue
        for key in ("models", "downloads", "likes"):
            into[key] += record[key]
        stamps = [s for s in (into.get("first_published"), record.get("first_published")) if s]
        into["first_published"] = min(stamps) if stamps else None
    return top_publishers(merged.values())


def architecture_authors(gap: dict) -> list[str]:
    """Namespaces credited with a gap architecture: top publisher, then earliest if different."""
    publishers = gap.get("publishers") or []
    if not publishers:
        # Entries scanned before publishers were tracked: the first sample is the
        # most-downloaded checkpoint.
        samples = gap.get("sample_models") or []
        namespace = namespace_of(samples[0]) if samples else None
        return [namespace] if namespace else []
    top = min(publishers, key=lambda p: (-p["downloads"], -p["models"], p["name"]))
    authors = [top["name"]]
    first = _earliest(publishers)
    if first is not None and first["name"] != top["name"]:
        authors.append(first["name"])
    return authors


def _percentile(value: int, reference: list[int]) -> float:
    """Share of the reference values that `value` matches or exceeds, 0-100."""
    return sum(1 for r in reference if r <= value) / len(reference) * 100


def build_organization_gaps_report(
    gaps: list[dict],
    scan_info: Optional[dict] = None,
    generated_at: Optional[str] = None,
    preferred: Optional[Iterable[str]] = None,
) -> dict:
    """Rank non-preferred authors of gap architectures that measure up to preferred authors."""
    names = list(PREFERRED_AUTHORS if preferred is None else preferred)

    # An author's footprint is only its own checkpoints: community derivatives say
    # nothing about the org, and a re-uploader's popularity lives in other orgs' architectures.
    profiles: dict[str, dict] = {}
    for gap in gaps:
        own_records = {p["name"]: p for p in gap.get("publishers") or []}
        for author in architecture_authors(gap):
            profile = profiles.setdefault(
                author,
                {"architectures": [], "total_models": 0, "total_downloads": 0, "total_likes": 0},
            )
            profile["architectures"].append(gap["architecture_id"])
            own = own_records.get(author)
            if own is not None:
                profile["total_models"] += own["models"]
                profile["total_downloads"] += own["downloads"]
                profile["total_likes"] += own["likes"]

    reference = [p for name, p in profiles.items() if is_preferred_author(name, names)]
    ref_downloads = [p["total_downloads"] for p in reference]
    ref_likes = [p["total_likes"] for p in reference]
    min_downloads = statistics.median(ref_downloads) if reference else None
    min_likes = statistics.median(ref_likes) if reference else None

    entries: list[dict] = []
    for name, profile in profiles.items():
        # No preferred author has a gap architecture: nothing to measure candidates against.
        if min_downloads is None or min_likes is None or is_preferred_author(name, names):
            continue
        if profile["total_downloads"] < min_downloads or profile["total_likes"] < min_likes:
            continue
        downloads_pct = _percentile(profile["total_downloads"], ref_downloads)
        likes_pct = _percentile(profile["total_likes"], ref_likes)
        entries.append(
            {
                "organization": name,
                "relevancy_score": round((downloads_pct + likes_pct) / 2, 1),
                "total_architectures": len(profile["architectures"]),
                "total_models": profile["total_models"],
                "total_downloads": profile["total_downloads"],
                "total_likes": profile["total_likes"],
                "downloads_percentile": round(downloads_pct, 1),
                "likes_percentile": round(likes_pct, 1),
                "architectures": profile["architectures"],
            }
        )
    entries.sort(key=lambda e: (-e["relevancy_score"], -e["total_downloads"], e["organization"]))

    return {
        "generated_at": generated_at or date.today().isoformat(),
        "scan_info": scan_info,
        "criteria": {
            "reference_authors": len(reference),
            "reference_statistic": "median",
            "min_downloads": min_downloads,
            "min_likes": min_likes,
        },
        "total_candidate_organizations": len(entries),
        "gaps": entries,
    }
