#!/usr/bin/env python3
"""Merge the architecture-specific model list produced by
scan-hf-architecture.py into `supported_models.json` so that `verify_models`
can pick the models up.

Reads:  transformer_lens/tools/model_registry/data/supported_models_<arch_short>.json
Writes: transformer_lens/tools/model_registry/data/supported_models.json

This is an idempotent merge — models already present in the registry are
skipped, not duplicated. The per-arch file stays in place untouched.

Run from inside the TransformerLens worktree:

    uv run python "$TL_ADAPTER_BUILDER_ROOT/scripts/port-arch-models.py" \\
        --arch-short qwen3_moe

Skipping this step is a common failure mode that wastes hours — if the
target architecture's models aren't in `supported_models.json`, verify_models
silently skips them and you get no signal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch-short",
        required=True,
        help="Short lowercase name matching the scan output, e.g. qwen3_moe",
    )
    parser.add_argument(
        "--data-dir",
        default="transformer_lens/tools/model_registry/data",
        help="Registry data directory (resolved relative to cwd)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    arch_file = data_dir / f"supported_models_{args.arch_short}.json"
    registry_file = data_dir / "supported_models.json"

    if not arch_file.exists():
        print(f"ERROR: arch file not found: {arch_file}", file=sys.stderr)
        print("Run scan-hf-architecture.py first.", file=sys.stderr)
        return 1

    if not registry_file.exists():
        print(f"ERROR: registry file not found: {registry_file}", file=sys.stderr)
        return 1

    arch_data = json.loads(arch_file.read_text())
    registry = json.loads(registry_file.read_text())

    existing_ids = {m["model_id"] for m in registry["models"]}
    arch_models = arch_data["models"]
    new_entries = [m for m in arch_models if m["model_id"] not in existing_ids]

    registry["models"].extend(new_entries)
    arch_ids = {m["architecture_id"] for m in registry["models"]}
    registry["total_architectures"] = len(arch_ids)
    registry["total_models"] = len(registry["models"])

    registry_file.write_text(json.dumps(registry, indent=2) + "\n")

    skipped = len(arch_models) - len(new_entries)
    print(f"Ported {len(new_entries)} new models to {registry_file.name}")
    print(f"(Skipped {skipped} already present)")
    print(
        f"Registry now has {registry['total_models']} total models across "
        f"{registry['total_architectures']} architectures"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
