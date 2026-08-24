"""Shared I/O functions for reading and writing model registry data files.

Consolidates the load-modify-save pattern used by verify_models.py and
main_benchmark.py into a single module that properly uses the
VerificationRecord/VerificationHistory dataclasses.
"""

import json
import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from .verification import VerificationHistory, VerificationRecord

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_SUPPORTED_MODELS_PATH = _DATA_DIR / "supported_models.json"
_VERIFICATION_HISTORY_PATH = _DATA_DIR / "verification_history.json"
_MODEL_ALIASES_PATH = _DATA_DIR / "model_aliases.json"

# Status codes
STATUS_UNVERIFIED = 0
STATUS_VERIFIED = 1
STATUS_SKIPPED = 2
STATUS_FAILED = 3
# Structural-only pass (--no-hf-reference): Phase 1 ran without an HF reference,
# so the forward was never numerically compared to HuggingFace. Recorded as a
# real result but deliberately NOT counted as verified.
STATUS_PROVISIONAL = 4

# Human-readable labels for docs display. STATUS_SKIPPED deliberately renders as
# "Unverified": a skip (memory/tooling) is not a verification outcome.
STATUS_LABELS: dict[int, str] = {
    STATUS_UNVERIFIED: "Unverified",
    STATUS_VERIFIED: "Verified",
    STATUS_SKIPPED: "Unverified",
    STATUS_FAILED: "Failed",
    STATUS_PROVISIONAL: "Provisional",
}

# Registry phase-score columns: text 1-4, multimodal 7, audio 8, vision 9.
# PHASES is derived so the full column set can never drift from the two groups.
TEXT_PHASES: tuple[int, ...] = (1, 2, 3, 4)
MODALITY_PHASES: tuple[int, ...] = (7, 8, 9)
PHASES: tuple[int, ...] = TEXT_PHASES + MODALITY_PHASES

# HF-loadable quantization formats. Admitted to the registry; verification gates
# on `required_quant_library_for_model()` at run time.
_HF_LOADABLE_QUANT_PATTERNS = [
    "-awq",
    "_awq",
    "-AWQ",
    "_AWQ",
    "-gptq",
    "_gptq",
    "-GPTQ",
    "_GPTQ",
    "GPTQ",
    "-bnb-",
    "_bnb_",
    "bnb-4bit",
    "bnb-8bit",
    "-4bit",
    "_4bit",
    "-8bit",
    "_8bit",
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
    "-hqq",
    "_hqq",
    "-HQQ",
    "_HQQ",
    "-3bit",
    "_3bit",
    "-2bit",
    "_2bit",
    "-5bit",
    "-6bit",
    "-oQ",
    "_oQ",
    "-quantized.",
    "_Quantized",
    "-Quantized",
]

# Formats that need a non-HF loader (GGUF→llama.cpp, MLX→Apple, FP4/FP8→NVIDIA).
_INCOMPATIBLE_QUANT_PATTERNS = [
    "-gguf",
    "_gguf",
    "-GGUF",
    "_GGUF",
    "mlx-community/",
    "-mlx",
    "-MLX",
    "_mlx",
    "_MLX",
    ".mlx",
    ".MLX",
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
]

# Values are Python import names, not PyPI package names. Order matters: explicit
# format markers must precede generic bit-width markers (HQQ-4bit IDs match both).
_QUANT_LIBRARY_BY_PATTERN: list[tuple[tuple[str, ...], str]] = [
    (("-hqq", "_hqq", "-HQQ", "_HQQ"), "hqq"),
    (("-gptq", "_gptq", "-GPTQ", "_GPTQ", "GPTQ"), "auto_gptq"),
    (("-awq", "_awq", "-AWQ", "_AWQ"), "awq"),
    (("-w4a16", "-w8a8", "-W4A16", "-W8A8", ".w4a16", ".W4A16"), "auto_gptq"),
    (("-bnb-", "_bnb_", "bnb-4bit", "bnb-8bit"), "bitsandbytes"),
    (("-4bit", "_4bit", "-8bit", "_8bit", "-int4", "_int4", "-int8", "_int8"), "bitsandbytes"),
]

QUANTIZED_NOTE = "Quantized format not loadable by HF transformers"


def is_incompatible_quantized(model_id: str) -> bool:
    """True for quantization formats the bridge can't ingest (GGUF, MLX, FP4/FP8)."""
    return any(pat in model_id for pat in _INCOMPATIBLE_QUANT_PATTERNS)


def is_hf_loadable_quantized(model_id: str) -> bool:
    """True for quantizations loadable by HF transformers + a quant library."""
    return any(pat in model_id for pat in _HF_LOADABLE_QUANT_PATTERNS)


def required_quant_library_for_model(model_id: str) -> Optional[str]:
    """Return the Python import name needed to load this model, or None if unquantized."""
    for patterns, library in _QUANT_LIBRARY_BY_PATTERN:
        if any(pat in model_id for pat in patterns):
            return library
    return None


def is_quantized_model(model_id: str) -> bool:
    """Alias for ``is_incompatible_quantized`` — kept for back-compat with existing call sites."""
    return is_incompatible_quantized(model_id)


@lru_cache(maxsize=1)
def load_model_aliases() -> dict[str, list[str]]:
    """Load the canonical alias table: official HF model name -> deprecated short aliases."""
    with open(_MODEL_ALIASES_PATH) as f:
        return json.load(f)["aliases"]


def resolve_model_alias(model_name: str) -> Optional[str]:
    """Return the official HF name if ``model_name`` is a deprecated alias, else None."""
    for official_name, aliases in load_model_aliases().items():
        if model_name in aliases:
            return official_name
    return None


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

        version = getattr(transformer_lens, "__version__", None)
        if version:
            return str(version)
        # The package exports no __version__; installed-distribution
        # metadata is the fallback (dev installs record 0.0.0).
        from importlib.metadata import version as dist_version

        return dist_version("transformer-lens")
    except Exception:
        return None


def pass_status(use_hf_reference: bool) -> int:
    """Status for a passing run: VERIFIED with an HF reference, else PROVISIONAL
    (a --no-hf-reference structural-only pass is recorded but not counted verified)."""
    return STATUS_VERIFIED if use_hf_reference else STATUS_PROVISIONAL


def extract_phase_scores(results: list) -> dict[int, Optional[float]]:
    """Extract phase scores from benchmark results.

    Shared home for both registry-writing paths (verify_models and
    main_benchmark.update_model_registry) so they cannot drift.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Dict mapping phase number to score (0-100) or None
    """
    from transformer_lens.benchmarks.utils import BenchmarkSeverity

    phase_results: dict[int, list[bool]] = {phase: [] for phase in PHASES}
    for result in results:
        if result.phase in phase_results and result.severity != BenchmarkSeverity.SKIPPED:
            phase_results[result.phase].append(result.passed)

    scores: dict[int, Optional[float]] = {}
    for phase, passed_list in phase_results.items():
        if passed_list:
            scores[phase] = round(sum(passed_list) / len(passed_list) * 100, 1)
        # Omit phases with no results — they weren't run, so their
        # existing registry scores should be preserved.

    # Phase 4 (text quality): store the actual 0-100 quality score from the
    # benchmark details instead of a binary pass/fail percentage.
    if 4 in scores:
        for result in results:
            if result.phase == 4 and result.details and "score" in result.details:
                scores[4] = round(result.details["score"], 1)
                break

    return scores


def recompute_registry_totals(models: list[dict]) -> dict:
    """Header totals for supported_models.json, recomputed from the models list.

    Shared by both writers (``update_model_status`` here and hf_scraper's report
    builder) so the counting rules cannot drift.
    """
    return {
        "total_architectures": len({m["architecture_id"] for m in models}),
        "total_models": len(models),
        "total_verified": sum(1 for m in models if m.get("status", 0) == STATUS_VERIFIED),
        "total_provisional": sum(1 for m in models if m.get("status", 0) == STATUS_PROVISIONAL),
    }


def update_model_status(
    model_id: str,
    arch_id: str,
    status: Optional[int] = None,
    note: Optional[str] = None,
    phase_scores: Optional[dict[int, Optional[float]]] = None,
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
    prompt_profile: Optional[str] = None,
) -> bool:
    """Update a single model entry in supported_models.json.

    If the model is not found in the registry and status is STATUS_VERIFIED or
    STATUS_PROVISIONAL, a new entry is appended.

    When status is None (partial-phase update), only the provided phase_scores
    are updated — status, note, and other scores are preserved.

    Args:
        model_id: The model to update
        arch_id: Architecture of the model
        status: New status code (0-4), or None for score-only updates
        note: Optional note for skip/fail reason
        phase_scores: Phase score dict {1: float, 2: float, 3: float, 4: float}
        sanitize_fn: Optional callable to sanitize note strings
        prompt_profile: Phase-4 prompt profile actually used (e.g.
            "task:translation@en-de"). Sparse: the default "continuation"
            removes the key (clearing a stale non-default value), None (no
            Phase-4 result) leaves it untouched.

    Returns:
        True if entry was found/created and updated
    """
    # Deferred: benchmarks imports model_bridge, and this module loads during
    # `import transformer_lens` (via supported_models) — importing it at module
    # scope closes an import cycle back through the bridge's HF source.
    from transformer_lens.benchmarks.text_quality_profiles import (
        P4_SCORING_VERSION,
        is_default_profile,
    )

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
            for phase_num in PHASES:
                key = f"phase{phase_num}_score"
                if phase_num in phase_scores:
                    entry[key] = phase_scores[phase_num]
                elif key not in entry:
                    entry[key] = None
            if prompt_profile is not None and is_default_profile(prompt_profile):
                entry.pop("prompt_profile", None)
            elif prompt_profile is not None:
                entry["prompt_profile"] = prompt_profile
            if 4 in phase_scores:
                entry["p4_scoring_version"] = P4_SCORING_VERSION
            # Reorder keys so phase scores are always in numerical order
            _KEY_ORDER = [
                "architecture_id",
                "model_id",
                "status",
                "verified_date",
                "metadata",
                "note",
                "prompt_profile",
                "p4_scoring_version",
                *[f"phase{p}_score" for p in PHASES],
            ]
            reordered = {k: entry[k] for k in _KEY_ORDER if k in entry}
            for k in entry:
                if k not in reordered:
                    reordered[k] = entry[k]
            entry.clear()
            entry.update(reordered)
            updated = True
            break

    if not updated and status in (STATUS_VERIFIED, STATUS_PROVISIONAL):
        # Model not in registry -- add it. A structural-only (provisional) pass
        # is a real result worth recording; skipped/failed on a missing model
        # are not, so they still fall through.
        data.get("models", []).append(
            {
                "model_id": model_id,
                "architecture_id": arch_id,
                "status": status,
                "verified_date": date.today().isoformat(),
                "metadata": None,
                "note": note,
                **{f"phase{p}_score": phase_scores.get(p) for p in PHASES},
            }
        )
        new_entry = data["models"][-1]
        extras: list[tuple[str, object]] = []
        if prompt_profile is not None and not is_default_profile(prompt_profile):
            extras.append(("prompt_profile", prompt_profile))
        if phase_scores.get(4) is not None:
            extras.append(("p4_scoring_version", P4_SCORING_VERSION))
        if extras:
            # Keep key position consistent with _KEY_ORDER (after "note").
            items = list(new_entry.items())
            idx = [k for k, _ in items].index("note") + 1
            for offset, pair in enumerate(extras):
                items.insert(idx + offset, pair)
            new_entry.clear()
            new_entry.update(items)
        updated = True

    if updated:
        data.update(recompute_registry_totals(data.get("models", [])))
        save_supported_models_raw(data)

    return updated


def registry_prompt_profile(model_id: str) -> Optional[str]:
    """Stored prompt_profile for a model, or None. Uncached read: the sweep
    rewrites the registry between models."""
    try:
        data = load_supported_models_raw()
    except Exception:
        return None
    for entry in data.get("models", []):
        if entry.get("model_id") == model_id:
            profile = entry.get("prompt_profile")
            return profile if isinstance(profile, str) else None
    return None


def add_verification_record(
    model_id: str,
    arch_id: str,
    notes: Optional[str] = None,
    verified_by: str = "verify_models",
    sanitize_fn: Optional[Callable[[Optional[str]], Optional[str]]] = None,
    prompt_profile: Optional[str] = None,
    p4_scoring_version: Optional[int] = None,
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
        prompt_profile=prompt_profile,
        p4_scoring_version=p4_scoring_version,
    )

    history = load_verification_history()
    history.add_record(record)
    save_verification_history(history)
