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

# Patterns in model IDs that indicate quantized models.  TransformerLens
# requires full-precision weights for mechanistic interpretability research,
# so quantized variants are fundamentally incompatible.
_QUANTIZED_PATTERNS = [
    "-awq",
    "_awq",
    "-AWQ",
    "_AWQ",
    "-gptq",
    "_gptq",
    "-GPTQ",
    "_GPTQ",
    "GPTQ",
    "-gguf",
    "_gguf",
    "-GGUF",
    "_GGUF",
    "-bnb-",
    "_bnb_",
    "bnb-4bit",
    "bnb-8bit",
    "-4bit",
    "_4bit",
    "-5bit",
    "-6bit",
    "-8bit",
    "_8bit",
    "-fp8",
    "_fp8",
    "-FP8",
    "_FP8",
    "-nvfp4",
    "_nvfp4",
    "-NVFP4",
    "_NVFP4",
    "-mxfp4",
    "_mxfp4",
    "-MXFP4",
    "_MXFP4",
    "-int4",
    "_int4",
    "-int8",
    "_int8",
    "-w4a16",
    "-w8a8",
    "-W4A16",
    "-W8A8",
    ".w4a16",
    ".W4A16",
    "-3bit",
    "_3bit",
    "-2bit",
    "_2bit",
    "-oQ",
    "_oQ",
    "-quantized.",
    "_Quantized",
    "-Quantized",
    "mlx-community/",
    "-MLX-",
]
QUANTIZED_NOTE = "TransformerLens does not support quantized models at this time"


def is_quantized_model(model_id: str) -> bool:
    """Check if a model ID indicates a quantized model variant.

    Detects AWQ, GPTQ, GGUF, BitsAndBytes (bnb), FP8, INT4/INT8,
    MLX quantized, and other common quantization suffixes.
    """
    return any(pat in model_id for pat in _QUANTIZED_PATTERNS)


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
    status: Optional[int] = None,
    note: Optional[str] = None,
    phase_scores: Optional[dict[int, Optional[float]]] = None,
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
) -> bool:
    """Update a single model entry in supported_models.json.

    If the model is not found in the registry and status == STATUS_VERIFIED,
    a new entry is appended.

    When status is None (partial-phase update), only the provided phase_scores
    are updated — status, note, and other scores are preserved.

    Args:
        model_id: The model to update
        arch_id: Architecture of the model
        status: New status code (0-3), or None for score-only updates
        note: Optional note for skip/fail reason
        phase_scores: Phase score dict {1: float, 2: float, 3: float, 4: float}
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
            if status is not None:
                entry["status"] = status
                entry["verified_date"] = (
                    date.today().isoformat() if status != STATUS_UNVERIFIED else None
                )
                entry["note"] = note
            elif note is not None:
                # Score-only update with an explicit note — overwrite stale notes
                entry["note"] = note
            elif phase_scores and "exceeds" in (entry.get("note") or "").lower():
                # Writing real scores clears a stale memory-skip note
                entry["note"] = None
            for phase_num in (1, 2, 3, 4, 7, 8):
                key = f"phase{phase_num}_score"
                if phase_num in phase_scores:
                    entry[key] = phase_scores[phase_num]
                elif key not in entry:
                    entry[key] = None
            # Reorder keys so phase scores are always in numerical order
            _KEY_ORDER = [
                "architecture_id",
                "model_id",
                "status",
                "verified_date",
                "metadata",
                "note",
                "phase1_score",
                "phase2_score",
                "phase3_score",
                "phase4_score",
                "phase7_score",
                "phase8_score",
            ]
            reordered = {k: entry[k] for k in _KEY_ORDER if k in entry}
            for k in entry:
                if k not in reordered:
                    reordered[k] = entry[k]
            entry.clear()
            entry.update(reordered)
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
                "phase4_score": phase_scores.get(4),
                "phase7_score": phase_scores.get(7),
                "phase8_score": phase_scores.get(8),
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
