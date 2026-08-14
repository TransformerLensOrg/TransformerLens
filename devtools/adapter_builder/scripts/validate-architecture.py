#!/usr/bin/env python3
"""Validate that a HuggingFace architecture class name exists in either the
installed `transformers` package or on the HuggingFace Hub.

Used by agents/launch.sh as a pre-flight check so typo'd architecture names
fail fast (before worktree creation, agent launch, or any wasted work)
rather than grinding through 3 review rounds and hitting the stuck-report
escalation.

The two checks are independent and run in order:

  1. `transformers` package (fast, local, ~1s first import).
     Uses `getattr(transformers, ARCH)` — this exercises the lazy-loading
     machinery and raises AttributeError for unknown names.

  2. HuggingFace Hub scan (slower, network, ~5-10s for --hf-limit=500).
     Pages through the top models by download count, expanding the `config`
     field, and breaks as soon as one model's `config.architectures`
     contains ARCH. We don't need exhaustive coverage — just *any* match.

Exit codes:
  0 — validated (found via transformers OR HF Hub)
  1 — definitively not found (both checks completed, neither found a match)
  2 — could not verify (missing deps or network error); caller should decide
      whether to abort or proceed

Usage:
  python3 scripts/validate-architecture.py CohereForCausalLM
  python3 scripts/validate-architecture.py FakeArch --skip-hf      # tf only
  python3 scripts/validate-architecture.py Qwen3MoeForCausalLM --hf-limit 1000
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional


def check_transformers(arch: str) -> Optional[bool]:
    """Return True if `arch` is importable from the transformers package,
    False if transformers is importable but `arch` is not a known attribute,
    None if transformers itself is unavailable."""
    try:
        import transformers
    except ImportError:
        return None

    # transformers uses _LazyModule: known names trigger lazy import, unknown
    # names raise AttributeError. Both the AttributeError path and a None
    # return count as "not found".
    try:
        cls = getattr(transformers, arch)
    except AttributeError:
        return False
    return cls is not None


def check_huggingface(arch: str, limit: int) -> Optional[bool]:
    """Return True if at least one HuggingFace Hub model has `arch` in its
    `config.architectures`, False if the bounded scan completes without a
    match, None on error (missing dep, network failure, API change)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None

    api = HfApi()
    try:
        for m in api.list_models(
            expand=["config"],
            sort="downloads",
            limit=limit,
        ):
            config = getattr(m, "config", None) or {}
            archs = config.get("architectures") or []
            if arch in archs:
                return True
        return False
    except Exception:
        return None


def main() -> int:
    """Parse args, run the two checks in order, and return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arch", help="HF architecture class, e.g. CohereForCausalLM")
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=500,
        help="Max HF models to scan before giving up (default: 500)",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Only check the transformers package; skip the HF Hub scan",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output on success",
    )
    args = parser.parse_args()

    # --- Check 1: transformers package (fast, local) -----------------------
    tf = check_transformers(args.arch)
    if tf is True:
        if not args.quiet:
            print(f"OK: '{args.arch}' found in transformers package")
        return 0

    if args.skip_hf:
        if tf is False:
            print(
                f"NOT FOUND: '{args.arch}' is not in the installed transformers package "
                f"(HF Hub check skipped).",
                file=sys.stderr,
            )
            return 1
        # tf is None
        print(
            "ERROR: transformers package is not importable in this environment. "
            "Install it or skip this check with --skip-arch-check on the launcher.",
            file=sys.stderr,
        )
        return 2

    # --- Check 2: HuggingFace Hub scan (bounded, network) ------------------
    if tf is False:
        print(
            f"'{args.arch}' not in transformers package; "
            f"scanning HuggingFace Hub (top {args.hf_limit} models)...",
            file=sys.stderr,
        )
    else:
        print(
            f"transformers package not importable; "
            f"scanning HuggingFace Hub (top {args.hf_limit} models) for '{args.arch}'...",
            file=sys.stderr,
        )

    hf = check_huggingface(args.arch, args.hf_limit)
    if hf is True:
        if not args.quiet:
            print(f"OK: '{args.arch}' found on HuggingFace Hub")
        return 0

    # --- Both checks ran; neither found it ---------------------------------
    if tf is False and hf is False:
        print(
            f"NOT FOUND: '{args.arch}' is not in the installed transformers package "
            f"and does not appear in the top {args.hf_limit} HuggingFace Hub models "
            f"by download count. Check for typos (case-sensitive), or if this is a "
            f"genuinely new/rare architecture, launch with --skip-arch-check.",
            file=sys.stderr,
        )
        return 1

    # --- At least one check was inconclusive -------------------------------
    if tf is False:
        print(
            f"INCONCLUSIVE: '{args.arch}' not in transformers, "
            "and HuggingFace Hub scan failed (network error or missing huggingface_hub).",
            file=sys.stderr,
        )
    elif hf is False:
        print(
            "INCONCLUSIVE: transformers package not importable, "
            f"and HuggingFace Hub scan did not find '{args.arch}' "
            f"in the top {args.hf_limit} models.",
            file=sys.stderr,
        )
    else:
        print(
            "INCONCLUSIVE: could not verify against either transformers or HuggingFace Hub. "
            "Check your Python environment and network connectivity.",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
