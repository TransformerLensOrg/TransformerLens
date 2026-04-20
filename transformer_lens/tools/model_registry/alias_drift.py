"""MODEL_ALIASES drift checker.

Compares the legacy MODEL_ALIASES / OFFICIAL_MODEL_NAMES in
transformer_lens/supported_models.py against the model registry in
data/supported_models.json, and reports:

1. Models in MODEL_ALIASES but NOT in the registry
2. Models in the registry (status=1, verified) but NOT in MODEL_ALIASES
3. Summary statistics

Usage:
    python -m transformer_lens.tools.model_registry.alias_drift
    python -m transformer_lens.tools.model_registry.alias_drift --format json
    python -m transformer_lens.tools.model_registry.alias_drift --all-statuses --exit-code
"""

import argparse
import json
import sys
from dataclasses import dataclass, field

from .registry_io import load_supported_models_raw


@dataclass
class DriftReport:
    """Result of comparing MODEL_ALIASES with the model registry."""

    # Models in MODEL_ALIASES / OFFICIAL_MODEL_NAMES but absent from registry
    in_aliases_not_registry: list[str] = field(default_factory=list)

    # Models verified (status=1) in registry but absent from MODEL_ALIASES
    in_registry_not_aliases: list[str] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return bool(self.in_aliases_not_registry or self.in_registry_not_aliases)

    def to_dict(self) -> dict:
        return {
            "in_aliases_not_registry": self.in_aliases_not_registry,
            "in_registry_not_aliases": self.in_registry_not_aliases,
            "has_drift": self.has_drift,
            "summary": {
                "aliases_only": len(self.in_aliases_not_registry),
                "registry_only": len(self.in_registry_not_aliases),
            },
        }


def check_drift(verified_only: bool = True) -> DriftReport:
    """Compare MODEL_ALIASES with the model registry.

    Args:
        verified_only: If True, only consider registry models with status=1
            when checking for models missing from MODEL_ALIASES.

    Returns:
        DriftReport with all discrepancies.
    """
    # Import at call time to avoid circular imports
    from transformer_lens.supported_models import MODEL_ALIASES, OFFICIAL_MODEL_NAMES

    report = DriftReport()

    # Load registry
    data = load_supported_models_raw()
    registry_models: dict[str, dict] = {}
    for entry in data.get("models", []):
        registry_models[entry["model_id"]] = entry

    # Build set of all legacy model IDs
    alias_model_ids = set(MODEL_ALIASES.keys())
    official_model_ids = set(OFFICIAL_MODEL_NAMES)
    legacy_model_ids = alias_model_ids | official_model_ids

    # Build set of registry model IDs (optionally filtered to verified)
    registry_model_ids = set(registry_models.keys())
    if verified_only:
        comparison_registry_ids = {
            mid for mid, entry in registry_models.items() if entry.get("status", 0) == 1
        }
    else:
        comparison_registry_ids = registry_model_ids

    # 1. In legacy but not in registry
    report.in_aliases_not_registry = sorted(legacy_model_ids - registry_model_ids)

    # 2. In registry (verified) but not in legacy
    report.in_registry_not_aliases = sorted(comparison_registry_ids - legacy_model_ids)

    return report


def print_report(report: DriftReport) -> None:
    """Print a human-readable drift report to stdout."""
    print(f"\n{'='*70}")
    print("MODEL_ALIASES <-> Registry Drift Report")
    print(f"{'='*70}")

    if not report.has_drift:
        print("\nNo drift detected. Both systems are in sync.")
        return

    if report.in_aliases_not_registry:
        print(
            f"\n--- Models in MODEL_ALIASES but NOT in registry "
            f"({len(report.in_aliases_not_registry)}) ---"
        )
        for mid in report.in_aliases_not_registry:
            print(f"  {mid}")

    if report.in_registry_not_aliases:
        print(
            f"\n--- Verified models in registry but NOT in MODEL_ALIASES "
            f"({len(report.in_registry_not_aliases)}) ---"
        )
        for mid in report.in_registry_not_aliases:
            print(f"  {mid}")

    print(
        f"\nSummary: "
        f"{len(report.in_aliases_not_registry)} aliases-only, "
        f"{len(report.in_registry_not_aliases)} registry-only"
    )


def main() -> None:
    """CLI entry point for the drift checker."""
    parser = argparse.ArgumentParser(
        description="Check for drift between MODEL_ALIASES and the model registry"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Include unverified registry models in the comparison",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with code 1 if drift is detected (useful for CI)",
    )

    args = parser.parse_args()
    report = check_drift(verified_only=not args.all_statuses)

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)

    if args.exit_code and report.has_drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
