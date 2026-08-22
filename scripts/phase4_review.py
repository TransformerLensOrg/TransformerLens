"""Registry-wide Phase-4 review: which verified models' stored scores predate
the profile rework and deserve a re-run.

phase4_score is a mixed-scale column: entries stamped p4_scoring_version=2
were measured with the pinned-judge reference-ratio scoring (pass line 56);
unstamped entries carry the old GPT-2 absolute-perplexity scale (pass line 85)
and are never compared against the new line — they are re-run candidates.
Read-only.
"""

import argparse

from transformer_lens.benchmarks.text_quality_profiles import (
    P4_SCORING_VERSION,
    resolve_profile,
)
from transformer_lens.tools.model_registry.registry_io import load_supported_models_raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--below", type=float, default=None, help="Only scores below this")
    parser.add_argument("--limit", type=int, default=None, help="Max rows per section")
    args = parser.parse_args()

    current: list = []
    stale: list = []
    for entry in load_supported_models_raw().get("models", []):
        if entry.get("status") != 1 or entry.get("phase4_score") is None:
            continue
        score = entry["phase4_score"]
        if args.below is not None and score >= args.below:
            continue
        profile = str(
            resolve_profile(
                entry["model_id"], entry.get("architecture_id"), entry.get("prompt_profile")
            )
        )
        row = (profile != "continuation", score, entry["model_id"], profile)
        if entry.get("p4_scoring_version") == P4_SCORING_VERSION:
            current.append(row)
        else:
            stale.append(row)

    # Profile-changed first, then ascending score: measurement changed most.
    for rows in (current, stale):
        rows.sort(key=lambda r: (not r[0], r[1]))
    if args.limit:
        current = current[: args.limit]
        stale = stale[: args.limit]

    print(
        f"{len(current)} scored on the current scale (v{P4_SCORING_VERSION}); "
        f"{len(stale)} on the old GPT-2 scale (re-run candidates)\n"
    )
    for title, rows in (
        (f"v{P4_SCORING_VERSION} (reference-ratio scale, pass 56)", current),
        ("v1 (GPT-2 scale — scores NOT comparable to the new pass line)", stale),
    ):
        if not rows:
            continue
        changed = sum(1 for r in rows if r[0])
        print(f"== {title}: {len(rows)} models, {changed} non-default profiles")
        print(f"{'score':>6}  {'profile':<28} model")
        for is_changed, score, model_id, profile in rows:
            marker = "*" if is_changed else " "
            print(f"{score:6.1f}{marker} {profile:<28} {model_id}")
        print()


if __name__ == "__main__":
    main()
