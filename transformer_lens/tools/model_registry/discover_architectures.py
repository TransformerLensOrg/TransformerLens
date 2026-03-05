#!/usr/bin/env python3
"""Discover all architectures on HuggingFace and classify them.

This is a lightweight discovery tool that scans a sample of HuggingFace models
to discover all unique architecture classes and categorize them as supported or
unsupported by TransformerLens. For comprehensive scanning, use hf_scraper.py.

Usage:
    python -m transformer_lens.tools.model_registry.discover_architectures
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Optional, TypedDict

from . import HF_SUPPORTED_ARCHITECTURES


class ArchitectureEntry(TypedDict):
    """Type for architecture entry dictionaries."""

    architecture_id: str
    total_models: int
    example_models: list[str]


def discover_architectures(
    num_models: int = 5000,
    output_dir: Optional[Path] = None,
) -> tuple[dict, dict]:
    """Discover all architectures by scanning HuggingFace models.

    Args:
        num_models: Number of top models to scan
        output_dir: Directory to write output files

    Returns:
        Tuple of (supported_count, unsupported_counts)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("huggingface_hub required: pip install huggingface_hub")

    token = os.environ.get("HF_TOKEN", None)
    api = HfApi(token=token)
    arch_counts: Counter[str] = Counter()
    arch_models: dict[str, list[str]] = {}  # Track example models per architecture
    checked = 0
    errors = 0

    print(f"Scanning top {num_models} text-generation models on HuggingFace...")
    print(f"Using {len(HF_SUPPORTED_ARCHITECTURES)} supported architectures")
    print("This may take several minutes due to API rate limits.\n")

    for model in api.list_models(pipeline_tag="text-generation", sort="downloads"):
        checked += 1
        if checked > num_models:
            break

        try:
            info = api.model_info(model.id)
            if info.config and isinstance(info.config, dict):
                archs = info.config.get("architectures", [])
                for arch in archs or []:
                    arch_counts[arch] += 1
                    if arch not in arch_models:
                        arch_models[arch] = []
                    if len(arch_models[arch]) < 5:
                        arch_models[arch].append(model.id)
        except Exception:
            errors += 1
            continue

        if checked % 500 == 0:
            print(
                f"  Checked {checked}/{num_models} models, found {len(arch_counts)} architectures..."
            )

        time.sleep(0.03)  # Rate limit

    print(f"\nScanned {checked} models ({errors} errors)")
    print(f"Discovered {len(arch_counts)} unique architecture classes\n")

    # Categorize architectures
    supported: dict[str, ArchitectureEntry] = {}
    unsupported: dict[str, ArchitectureEntry] = {}

    for arch, count in arch_counts.most_common():
        entry: ArchitectureEntry = {
            "architecture_id": arch,
            "total_models": count,
            "example_models": arch_models.get(arch, []),
        }
        if arch in HF_SUPPORTED_ARCHITECTURES:
            supported[arch] = entry
        else:
            unsupported[arch] = entry

    # Print summary
    print("=" * 70)
    print("SUPPORTED ARCHITECTURES")
    print("=" * 70)
    total_supported = 0
    for arch in sorted(supported.keys()):
        count = supported[arch]["total_models"]
        total_supported += count
        print(f"  {arch}: {count} models")
    print(f"\nTotal supported: {len(supported)} architectures, {total_supported} models")

    print("\n" + "=" * 70)
    print("UNSUPPORTED ARCHITECTURES (sorted by model count)")
    print("=" * 70)
    total_unsupported = 0
    for arch, data in sorted(unsupported.items(), key=lambda x: -x[1]["total_models"]):
        count = data["total_models"]
        total_unsupported += count
        examples = ", ".join(data["example_models"][:2])
        print(f"  {arch}: {count} models (e.g., {examples})")
    print(f"\nTotal unsupported: {len(unsupported)} architectures, {total_unsupported} models")

    # Write output if directory specified (matching schemas.py format)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write supported models as single file
        supported_models = []
        for arch, arch_data in supported.items():
            for model_id in arch_data["example_models"]:
                supported_models.append(
                    {
                        "architecture_id": arch,
                        "model_id": model_id,
                        "status": 0,
                        "verified_date": None,
                        "metadata": None,
                        "note": None,
                        "phase1_score": None,
                        "phase2_score": None,
                        "phase3_score": None,
                    }
                )

        # Count unique architectures
        arch_ids = set(m["architecture_id"] for m in supported_models)

        report = {
            "generated_at": date.today().isoformat(),
            "scan_info": None,
            "total_architectures": len(arch_ids),
            "total_models": len(supported_models),
            "total_verified": 0,
            "models": supported_models,
        }

        with open(output_dir / "supported_models.json", "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

        # Write gaps
        gaps = [
            {"architecture_id": arch, "total_models": gap_data["total_models"]}
            for arch, gap_data in sorted(unsupported.items(), key=lambda x: -x[1]["total_models"])
        ]

        gaps_report = {
            "generated_at": date.today().isoformat(),
            "total_unsupported": len(unsupported),
            "gaps": gaps,
        }

        with open(output_dir / "architecture_gaps.json", "w") as f:
            json.dump(gaps_report, f, indent=2)

        print(f"\nWrote data to {output_dir}")

    return supported, unsupported


def main():
    parser = argparse.ArgumentParser(
        description="Discover all HuggingFace architectures and classify support status"
    )
    parser.add_argument(
        "-n",
        "--num-models",
        type=int,
        default=3000,
        help="Number of models to scan (default: 3000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for JSON files",
    )

    args = parser.parse_args()
    discover_architectures(num_models=args.num_models, output_dir=args.output)


if __name__ == "__main__":
    main()
