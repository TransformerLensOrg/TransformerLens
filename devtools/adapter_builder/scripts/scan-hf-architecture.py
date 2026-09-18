#!/usr/bin/env python3
"""Scan HuggingFace for every model of a given architecture class and write
the results to `transformer_lens/tools/model_registry/data/supported_models_<arch_short>.json`.

This produces the authoritative per-architecture model list used by the
adapter-builder workflow. It stays in place as an artifact for human review,
and later gets merged into `supported_models.json` via port-arch-models.py
so that `verify_models` can pick the models up.

Run from inside the TransformerLens worktree:

    uv run python "$TL_ADAPTER_BUILDER_ROOT/scripts/scan-hf-architecture.py" \\
        Qwen3MoeForCausalLM --arch-short qwen3_moe

The `architecture_gaps.json` file is useful for reference metadata but its
`sample_models` list is capped at 10 — this scan is exhaustive, which is
what the adapter workflow needs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def scan(arch_class: str, limit: int) -> tuple[list[dict], int]:
    """Return (matches, scanned_count) from a paginated HF listing."""
    # Deferred import so --help works even when huggingface_hub isn't installed
    # in the current env (the script is meant to run via `uv run` from the
    # worktree, which has the project's deps available).
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "huggingface_hub is not installed. Run this with `uv run` from "
            "the TransformerLens worktree so the project's env is available.",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi()
    matches: list[dict] = []
    scanned = 0

    for m in api.list_models(
        pipeline_tag="text-generation",
        sort="downloads",
        expand=["config", "safetensors", "downloads"],
        limit=limit,
    ):
        scanned += 1
        config = getattr(m, "config", None) or {}
        archs = config.get("architectures") or []
        if arch_class not in archs:
            continue

        safetensors = getattr(m, "safetensors", None)
        total_params = None
        if safetensors and isinstance(safetensors, dict):
            total_params = safetensors.get("total")

        matches.append(
            {
                "architecture_id": arch_class,
                "model_id": m.id,
                "status": 0,
                "verified_date": None,
                "metadata": {
                    "downloads": m.downloads or 0,
                    "total_params": total_params,
                },
                "note": None,
                "phase1_score": None,
                "phase2_score": None,
                "phase3_score": None,
                "phase4_score": None,
                "phase7_score": None,
                "phase8_score": None,
            }
        )

    return matches, scanned


def main() -> int:
    """Parse args, run the scan, and write the result JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("arch_class", help="HF architecture class, e.g. Qwen3MoeForCausalLM")
    parser.add_argument(
        "--arch-short",
        required=True,
        help="Short lowercase name for the output file, e.g. qwen3_moe",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max models to scan from the HF listing (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        default="transformer_lens/tools/model_registry/data",
        help="Output directory (resolved relative to cwd; default matches the registry layout)",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"supported_models_{args.arch_short}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches, scanned = scan(args.arch_class, args.limit)

    output_path.write_text(
        json.dumps(
            {
                "architecture_id": args.arch_class,
                "total_models": len(matches),
                "scanned": scanned,
                "models": matches,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Scanned {scanned} models, found {len(matches)} matches for {args.arch_class}")
    print(f"Wrote: {output_path}")
    print()
    print("Top 20 by downloads:")
    top = sorted(matches, key=lambda x: -(x["metadata"]["downloads"] or 0))[:20]
    for m in top:
        params = m["metadata"]["total_params"]
        params_str = f"{params/1e9:.2f}B" if params else "?"
        downloads = m["metadata"]["downloads"]
        print(f"  {m['model_id']:60s}  downloads={downloads:>10}  params={params_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
