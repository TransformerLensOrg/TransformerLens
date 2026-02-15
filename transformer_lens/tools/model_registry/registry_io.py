"""Shared I/O functions for reading and writing model registry data files.

Consolidates the load-modify-save pattern used by verify_models.py and
main_benchmark.py into a single module that properly uses the
VerificationRecord/VerificationHistory dataclasses.
"""

import json
import logging
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from .verification import VerificationHistory, VerificationRecord

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_SUPPORTED_MODELS_PATH = _DATA_DIR / "supported_models.json"
_VERIFICATION_HISTORY_PATH = _DATA_DIR / "verification_history.json"

# Status codes
STATUS_UNVERIFIED = 0
STATUS_VERIFIED = 1
STATUS_SKIPPED = 2
STATUS_FAILED = 3


def load_supported_models_raw() -> dict:
    """Load supported_models.json as a raw dict."""
    with open(_SUPPORTED_MODELS_PATH) as f:
        return json.load(f)


def save_supported_models_raw(data: dict) -> None:
    """Save raw dict back to supported_models.json."""
    with open(_SUPPORTED_MODELS_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_verification_history() -> VerificationHistory:
    """Load verification_history.json into a VerificationHistory dataclass."""
    if _VERIFICATION_HISTORY_PATH.exists():
        with open(_VERIFICATION_HISTORY_PATH) as f:
            data = json.load(f)
        return VerificationHistory.from_dict(data)
    return VerificationHistory()


def save_verification_history(history: VerificationHistory) -> None:
    """Save VerificationHistory dataclass to verification_history.json."""
    with open(_VERIFICATION_HISTORY_PATH, "w") as f:
        json.dump(history.to_dict(), f, indent=2)
        f.write("\n")


def _get_tl_version() -> Optional[str]:
    """Get the current TransformerLens version, or None."""
    try:
        import transformer_lens

        return getattr(transformer_lens, "__version__", None)
    except Exception:
        return None


def update_model_status(
    model_id: str,
    arch_id: str,
    status: int,
    note: Optional[str] = None,
    phase_scores: Optional[dict[int, Optional[float]]] = None,
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> bool:
    """Update a single model entry in supported_models.json.

    If the model is not found in the registry and status == STATUS_VERIFIED,
    a new entry is appended.

    Args:
        model_id: The model to update
        arch_id: Architecture of the model
        status: New status code (0-3)
        note: Optional note for skip/fail reason
        phase_scores: Phase score dict {1: float, 2: float, 3: float}
        sanitize_fn: Optional callable to sanitize note strings

    Returns:
        True if entry was found/created and updated
    """
    if phase_scores is None:
        phase_scores = {}

    if sanitize_fn and note:
        note = sanitize_fn(note)

    data = load_supported_models_raw()
    updated = False

    for entry in data.get("models", []):
        if entry["model_id"] == model_id and entry["architecture_id"] == arch_id:
            entry["status"] = status
            entry["verified_date"] = (
                date.today().isoformat() if status != STATUS_UNVERIFIED else None
            )
            entry["note"] = note
            entry["phase1_score"] = phase_scores.get(1)
            entry["phase2_score"] = phase_scores.get(2)
            entry["phase3_score"] = phase_scores.get(3)
            updated = True
            break

    if not updated and status == STATUS_VERIFIED:
        # Model not in registry -- add it
        data.get("models", []).append(
            {
                "model_id": model_id,
                "architecture_id": arch_id,
                "status": status,
                "verified_date": date.today().isoformat(),
                "metadata": None,
                "note": note,
                "phase1_score": phase_scores.get(1),
                "phase2_score": phase_scores.get(2),
                "phase3_score": phase_scores.get(3),
            }
        )
        updated = True

    if updated:
        models = data.get("models", [])
        data["total_verified"] = sum(1 for m in models if m.get("status", 0) == STATUS_VERIFIED)
        data["total_models"] = len(models)
        data["total_architectures"] = len(set(m["architecture_id"] for m in models))
        save_supported_models_raw(data)

    return updated


def add_verification_record(
    model_id: str,
    arch_id: str,
    notes: Optional[str] = None,
    verified_by: str = "verify_models",
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> None:
    """Append a VerificationRecord to verification_history.json.

    Uses the VerificationRecord dataclass properly instead of raw dict
    manipulation.

    Args:
        model_id: The verified model
        arch_id: Architecture type
        notes: Optional verification notes
        verified_by: Who/what performed the verification
        sanitize_fn: Optional callable to sanitize note strings
    """
    if sanitize_fn and notes:
        notes = sanitize_fn(notes)

    record = VerificationRecord(
        model_id=model_id,
        architecture_id=arch_id,
        verified_date=date.today(),
        verified_by=verified_by,
        transformerlens_version=_get_tl_version(),
        notes=notes,
    )

    history = load_verification_history()
    history.add_record(record)
    save_verification_history(history)
