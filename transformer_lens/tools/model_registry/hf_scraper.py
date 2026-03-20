#!/usr/bin/env python3
"""HuggingFace model scraper for discovering compatible models.

This module queries the HuggingFace Hub API to find ALL models and categorize
them by architecture - those supported by TransformerLens and those not yet supported.

The scraper works by:
1. Scanning ALL text-generation models on HuggingFace (paginated)
2. Extracting the architecture class from each model's config
3. Categorizing models into supported vs unsupported based on TransformerLens adapters
4. Building comprehensive lists for both categories

Output format matches the schemas defined in schemas.py exactly, so the data
files can be loaded by api.py without any transformation.

Usage:
    # Full scan of all HuggingFace models (recommended)
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan

    # Quick scan (top N models by downloads)
    python -m transformer_lens.tools.model_registry.hf_scraper --limit 10000

    # Output to custom directory
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan --output data/
"""

import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from . import HF_SUPPORTED_ARCHITECTURES
from .registry_io import is_quantized_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_architecture(model_info) -> Optional[str]:  # type: ignore[no-untyped-def]
    """Extract the primary architecture class from a model's inline config.

    Args:
        model_info: ModelInfo object from list_models(expand=['config'])

    Returns:
        Architecture class name or None if not found
    """
    config = model_info.config
    if config and isinstance(config, dict):
        archs = config.get("architectures", [])
        if archs:
            return archs[0]
    return None


def _load_existing_models(output_dir: Path) -> tuple[set[str], list[dict]]:
    """Load model IDs and data already in supported_models.json.

    Args:
        output_dir: Directory containing the data files

    Returns:
        Tuple of (set of existing model IDs, list of existing model dicts)
    """
    existing_ids: set[str] = set()
    existing_models: list[dict] = []
    supported_path = output_dir / "supported_models.json"

    if supported_path.exists():
        try:
            with open(supported_path) as f:
                data = json.load(f)
            for model in data.get("models", []):
                if "model_id" in model:
                    existing_ids.add(model["model_id"])
                    existing_models.append(model)
            logger.info(f"Loaded {len(existing_ids)} existing models from {supported_path}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load existing models: {e}")

    return existing_ids, existing_models


def _build_model_entry(model_id: str, architecture_id: str) -> dict:
    """Build a model entry dict matching the ModelEntry schema."""
    return {
        "architecture_id": architecture_id,
        "model_id": model_id,
        "status": 0,
        "verified_date": None,
        "metadata": None,
        "note": None,
        "phase1_score": None,
        "phase2_score": None,
        "phase3_score": None,
        "phase4_score": None,
        "phase7_score": None,
        "phase8_score": None,
    }


def scrape_all_models(
    output_dir: Path,
    max_models: Optional[int] = None,
    task: str = "text-generation",
    batch_size: int = 1000,
    checkpoint_interval: int = 5000,
    min_downloads: int = 500,
) -> tuple[dict, dict]:
    """Scrape ALL models from HuggingFace and categorize by architecture.

    This is the comprehensive scraper that:
    1. Loads existing models from supported_models.json to preserve them
    2. Skips models already in the JSON (only scans new models)
    3. Iterates through ALL models for a given task
    4. Fetches the architecture from each model's config
    5. Categorizes into supported vs unsupported
    6. Saves checkpoints periodically for long runs

    Output format matches schemas.py exactly (SupportedModelsReport and
    ArchitectureGapsReport).

    Args:
        output_dir: Directory to write JSON data files
        max_models: Maximum NEW models to scan (None = unlimited/all)
        task: HuggingFace task filter (default: text-generation)
        batch_size: Log progress every N models
        checkpoint_interval: Save checkpoint every N models
        min_downloads: Minimum download count to include a model (default: 1000)

    Returns:
        Tuple of (supported_models_dict, architecture_gaps_dict)
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for scraping. "
            "Install it with: pip install huggingface_hub"
        )

    from transformer_lens.utilities.hf_utils import get_hf_token

    api = HfApi(token=get_hf_token())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing models from supported_models.json
    existing_model_ids, existing_models = _load_existing_models(output_dir)

    # Track all models by architecture (start with existing models)
    supported_models: list[dict] = list(existing_models)  # Preserve existing
    unsupported_arch_counts: dict[str, int] = {}  # arch -> count
    unsupported_arch_samples: dict[str, list[str]] = {}  # arch -> top model IDs
    max_samples = 10  # Keep top N sample models per unsupported architecture

    scanned = 0
    skipped = 0
    new_supported = 0
    errors = 0
    start_time = time.time()

    # Check for existing checkpoint to resume from
    checkpoint_path = output_dir / "scrape_checkpoint.json"
    seen_models: set[str] = set(existing_model_ids)  # Include existing as "seen"

    if checkpoint_path.exists():
        logger.info(f"Found checkpoint at {checkpoint_path}, loading...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        # Merge checkpoint data with existing
        checkpoint_supported = checkpoint.get("supported_models", [])
        for model in checkpoint_supported:
            if model["model_id"] not in existing_model_ids:
                supported_models.append(model)
                existing_model_ids.add(model["model_id"])
        unsupported_arch_counts = checkpoint.get("unsupported_arch_counts", {})
        unsupported_arch_samples = checkpoint.get("unsupported_arch_samples", {})
        seen_models.update(checkpoint.get("seen_models", []))
        scanned = checkpoint.get("scanned", 0)
        skipped = checkpoint.get("skipped", 0)
        logger.info(f"Resumed from checkpoint: {scanned} models already scanned")

    logger.info(f"Starting comprehensive HuggingFace scan for task='{task}'...")
    logger.info(f"Skipping {len(existing_model_ids)} models already in supported_models.json")
    logger.info(f"Supported architectures: {len(HF_SUPPORTED_ARCHITECTURES)}")
    logger.info(f"Minimum downloads threshold: {min_downloads:,}")
    if max_models:
        logger.info(f"Will scan up to {max_models} NEW models")
    else:
        logger.info("Will scan ALL new models (this may take a while)")

    try:
        # Use expand=['config'] to get architecture data inline with the listing,
        # avoiding per-model API calls and rate limits entirely.
        # With ~1000 models per page, a full scan of 200K+ models needs only
        # ~200 paginated requests (well within the 1000 req / 5 min limit).
        list_kwargs: dict = {
            "pipeline_tag": task,
            "sort": "downloads",
            "expand": ["config"],
        }
        if max_models is not None:
            list_kwargs["limit"] = max_models + len(seen_models)

        # Retry loop: if we hit a 429 mid-pagination, save checkpoint, wait,
        # and restart iteration. Already-seen models are skipped automatically.
        max_retries = 10
        for attempt in range(max_retries + 1):
            try:
                for model in api.list_models(**list_kwargs):
                    # Skip if already in our JSON or processed in this run
                    if model.id in seen_models:
                        skipped += 1
                        continue

                    # Filter by minimum download count. Since results are sorted
                    # by downloads descending, once we drop below the threshold
                    # all remaining models will also be below it.
                    downloads = getattr(model, "downloads", None) or 0
                    if downloads < min_downloads:
                        logger.info(
                            f"Reached download threshold ({downloads:,} < "
                            f"{min_downloads:,}) after {scanned} models. "
                            f"Stopping scan."
                        )
                        break

                    scanned += 1
                    seen_models.add(model.id)

                    if max_models and scanned > max_models:
                        break

                    # Skip quantized models (AWQ, GPTQ, GGUF, bnb, FP8, etc.)
                    # TransformerLens requires full-precision weights.
                    if is_quantized_model(model.id):
                        continue

                    # Extract architecture from inline config (no extra API call)
                    arch = _extract_architecture(model)

                    if arch is None:
                        errors += 1
                    elif arch in HF_SUPPORTED_ARCHITECTURES:
                        supported_models.append(_build_model_entry(model.id, arch))
                        new_supported += 1
                    else:
                        unsupported_arch_counts[arch] = unsupported_arch_counts.get(arch, 0) + 1
                        # Track top models per arch (sorted by downloads since list is sorted)
                        samples = unsupported_arch_samples.setdefault(arch, [])
                        if len(samples) < max_samples:
                            samples.append(model.id)

                    # Progress logging
                    if scanned % batch_size == 0:
                        elapsed = time.time() - start_time
                        rate = scanned / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Scanned {scanned} new | "
                            f"Skipped {skipped} existing | "
                            f"New supported: {new_supported} | "
                            f"Total supported: {len(supported_models)} | "
                            f"Unsupported archs: {len(unsupported_arch_counts)} | "
                            f"Errors: {errors} | "
                            f"Rate: {rate:.1f}/s"
                        )

                    # Save checkpoint periodically
                    if scanned % checkpoint_interval == 0:
                        _save_checkpoint(
                            checkpoint_path,
                            supported_models,
                            unsupported_arch_counts,
                            unsupported_arch_samples,
                            list(seen_models),
                            scanned,
                            skipped,
                        )
                        logger.info(f"Saved checkpoint at {scanned} models")

                break  # Iteration completed successfully, exit retry loop

            except Exception as exc:
                if "429" in str(exc) and attempt < max_retries:
                    wait = min(10 * (attempt + 1), 60)
                    logger.warning(
                        f"Rate limited (429). Saving checkpoint and waiting {wait}s "
                        f"before retry ({attempt + 1}/{max_retries})..."
                    )
                    _save_checkpoint(
                        checkpoint_path,
                        supported_models,
                        unsupported_arch_counts,
                        unsupported_arch_samples,
                        list(seen_models),
                        scanned,
                        skipped,
                    )
                    time.sleep(wait)
                    skipped = 0  # Reset skip counter for restart
                else:
                    raise

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving checkpoint...")
        _save_checkpoint(
            checkpoint_path,
            supported_models,
            unsupported_arch_counts,
            unsupported_arch_samples,
            list(seen_models),
            scanned,
            skipped,
        )
        raise
    except Exception as e:
        logger.error(f"Error during scan: {e}")
        _save_checkpoint(
            checkpoint_path,
            supported_models,
            unsupported_arch_counts,
            unsupported_arch_samples,
            list(seen_models),
            scanned,
            skipped,
        )
        raise

    # Build final reports (matching schemas.py exactly)
    elapsed = time.time() - start_time
    logger.info(f"\nScan complete in {elapsed:.1f}s")
    logger.info(f"New models scanned: {scanned}")
    logger.info(f"Existing models skipped: {skipped}")
    logger.info(f"New supported models found: {new_supported}")
    logger.info(f"Total supported models: {len(supported_models)}")
    logger.info(f"Unsupported architectures found: {len(unsupported_arch_counts)}")

    # Count unique supported architectures and verified models
    supported_arch_ids: set[str] = set()
    total_verified = 0
    for model in supported_models:
        supported_arch_ids.add(model["architecture_id"])
        if model.get("status", 0) == 1:
            total_verified += 1

    # Build scan info (shared by both reports)
    scan_info = {
        "total_scanned": scanned,
        "task_filter": task,
        "min_downloads": min_downloads,
        "scan_duration_seconds": round(elapsed, 1),
    }

    # Build supported models report dict (for return value)
    supported_report = {
        "generated_at": date.today().isoformat(),
        "scan_info": scan_info,
        "total_architectures": len(supported_arch_ids),
        "total_models": len(supported_models),
        "total_verified": total_verified,
        "models": supported_models,
    }

    # Write supported models (single file)
    with open(output_dir / "supported_models.json", "w") as f:
        json.dump(supported_report, f, indent=2)
        f.write("\n")
    logger.info(f"Wrote {len(supported_models)} supported models to supported_models.json")

    # Build architecture gaps report (matches ArchitectureGapsReport schema)
    gaps: list[dict] = [
        {
            "architecture_id": arch,
            "total_models": count,
            "sample_models": unsupported_arch_samples.get(arch, []),
        }
        for arch, count in sorted(unsupported_arch_counts.items(), key=lambda x: -x[1])
    ]

    gaps_report = {
        "generated_at": date.today().isoformat(),
        "scan_info": scan_info,
        "total_unsupported_architectures": len(gaps),
        "total_unsupported_models": sum(unsupported_arch_counts.values()),
        "gaps": gaps,
    }

    gaps_path = output_dir / "architecture_gaps.json"
    with open(gaps_path, "w") as f:
        json.dump(gaps_report, f, indent=2)
    logger.info(f"Wrote {len(gaps)} architecture gaps to {gaps_path}")

    # Write verification history placeholder (single file)
    verification_path = output_dir / "verification_history.json"
    if not verification_path.exists():
        with open(verification_path, "w") as f:
            json.dump({"last_updated": None, "records": []}, f, indent=2)
            f.write("\n")

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Removed checkpoint file (scan complete)")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SCAN SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total models scanned: {scanned}")
    logger.info(f"\nSUPPORTED ARCHITECTURES ({len(supported_arch_ids)}):")

    # Count models per supported architecture
    supported_arch_counts: dict[str, int] = {}
    for model in supported_models:
        arch = model["architecture_id"]
        supported_arch_counts[arch] = supported_arch_counts.get(arch, 0) + 1

    for arch, count in sorted(supported_arch_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {arch}: {count} models")

    logger.info(f"\nTOP 20 UNSUPPORTED ARCHITECTURES (of {len(gaps)}):")
    for gap in gaps[:20]:
        logger.info(f"  {gap['architecture_id']}: {gap['total_models']} models")

    if len(gaps) > 20:
        remaining = sum(g["total_models"] for g in gaps[20:])
        logger.info(f"  ... and {len(gaps) - 20} more architectures ({remaining} models)")

    logger.info("=" * 70)

    return supported_report, gaps_report


def _save_checkpoint(
    path: Path,
    supported_models: list,
    unsupported_arch_counts: dict,
    unsupported_arch_samples: dict,
    seen_models: list,
    scanned: int,
    skipped: int = 0,
):
    """Save scraping progress to a checkpoint file."""
    checkpoint = {
        "supported_models": supported_models,
        "unsupported_arch_counts": unsupported_arch_counts,
        "unsupported_arch_samples": unsupported_arch_samples,
        "seen_models": seen_models,
        "scanned": scanned,
        "skipped": skipped,
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape HuggingFace to find all TransformerLens-compatible models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full scan of ALL text-generation models (recommended)
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan

    # Quick scan of top 10,000 models by downloads
    python -m transformer_lens.tools.model_registry.hf_scraper --limit 10000

    # Resume interrupted scan (checkpoints are saved automatically)
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan

    # Output to custom directory
    python -m transformer_lens.tools.model_registry.hf_scraper --full-scan -o ./my_data/
""",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for JSON data files (default: ./data/)",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan ALL models on HuggingFace (may take hours, saves checkpoints)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum models to scan (default: 10000, ignored with --full-scan)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help="HuggingFace task to filter by (default: text-generation)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N models (default: 5000)",
    )
    parser.add_argument(
        "--min-downloads",
        type=int,
        default=500,
        help="Minimum download count to include a model (default: 500)",
    )

    args = parser.parse_args()

    max_models = None if args.full_scan else args.limit

    scrape_all_models(
        output_dir=args.output,
        max_models=max_models,
        task=args.task,
        checkpoint_interval=args.checkpoint_interval,
        min_downloads=args.min_downloads,
    )


if __name__ == "__main__":
    main()
