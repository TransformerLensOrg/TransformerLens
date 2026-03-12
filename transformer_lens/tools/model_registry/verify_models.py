"""Batch model verification tool for the TransformerLens model registry.

Iterates through supported models, estimates memory requirements, runs benchmarks
phase-by-phase, and updates the registry with status, phase scores, and notes.

Usage:
    python -m transformer_lens.tools.model_registry.verify_models [options]

Examples:
    # Dry run to see what would be tested
    python -m transformer_lens.tools.model_registry.verify_models --dry-run

    # Verify top 10 models per architecture on CPU
    python -m transformer_lens.tools.model_registry.verify_models --device cpu

    # Verify only GPT2 models, limit to 3
    python -m transformer_lens.tools.model_registry.verify_models --architectures GPT2LMHeadModel --limit 3

    # Resume from a previous interrupted run
    python -m transformer_lens.tools.model_registry.verify_models --resume

    # Re-verify already-tested models for a specific architecture
    python -m transformer_lens.tools.model_registry.verify_models --reverify --architectures Olmo2ForCausalLM
"""

import argparse
import gc
import json
import logging
import os
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Exit code used for graceful interrupts (Ctrl+C).  The wrapper script
# recognises this and stops without marking the in-flight model as failed.
_EXIT_GRACEFUL_INTERRUPT = 42

# Module-level flag set by the SIGINT handler so the main loop can stop
# between models without corrupting state.
_interrupt_requested = False

from .registry_io import (
    QUANTIZED_NOTE,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    add_verification_record,
    is_quantized_model,
    load_supported_models_raw,
    update_model_status,
)

logger = logging.getLogger(__name__)

# Data directory for registry files
_DATA_DIR = Path(__file__).parent / "data"
_CHECKPOINT_PATH = _DATA_DIR / "verification_checkpoint.json"


def _handle_sigint(signum, frame):  # noqa: ARG001
    """Handle Ctrl+C by setting a flag instead of raising immediately.

    The main verification loop checks this flag between models so it can
    save the checkpoint cleanly and exit without marking the current model
    as failed.
    """
    global _interrupt_requested  # noqa: PLW0603
    if _interrupt_requested:
        # Second Ctrl+C — force exit immediately
        print("\nForce quit.")
        raise SystemExit(1)
    _interrupt_requested = True
    print("\n\nInterrupt received — finishing current model before stopping.")
    print("(Press Ctrl+C again to force quit immediately.)\n")


# Pattern matching HuggingFace API tokens (hf_ followed by 20+ alphanumeric chars)
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _sanitize_note(note: Optional[str]) -> Optional[str]:
    """Sanitize a note string to remove sensitive information.

    Strips HuggingFace tokens and replaces verbose gated-repo error messages
    with a concise summary.
    """
    if not note:
        return note
    # Replace any HF tokens that leaked into the message
    note = _HF_TOKEN_RE.sub("HF_TOKEN", note)
    # Replace verbose gated-repo 401 errors with a clean summary
    if "gated repo" in note:
        url_match = re.search(r"https://huggingface\.co/([^\s.]+)", note)
        model_ref = url_match.group(1) if url_match else "unknown"
        return f"Config unavailable: Gated repo ({model_ref})"
    return note


def _get_current_model_status(model_id: str, arch_id: str) -> int:
    """Look up a model's current status in the registry.

    Returns STATUS_UNVERIFIED (0) if the model is not found.
    """
    data = load_supported_models_raw()
    for entry in data.get("models", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("model_id") == model_id and entry.get("architecture_id") == arch_id:
            return entry.get("status", STATUS_UNVERIFIED)
    return STATUS_UNVERIFIED


@dataclass
class ModelCandidate:
    """A model selected for verification."""

    model_id: str
    architecture_id: str
    estimated_params: Optional[int] = None
    estimated_memory_gb: Optional[float] = None


@dataclass
class VerificationProgress:
    """Tracks progress across a verification run."""

    tested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    start_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tested": self.tested,
            "skipped": self.skipped,
            "failed": self.failed,
            "verified": self.verified,
            "start_time": self.start_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VerificationProgress":
        return cls(
            tested=data.get("tested", []),
            skipped=data.get("skipped", []),
            failed=data.get("failed", []),
            verified=data.get("verified", []),
            start_time=data.get("start_time"),
        )


def estimate_model_params(model_id: str) -> int:
    """Estimate parameter count using AutoConfig (lightweight, no model download).

    Fetches only the config JSON (~KB) and computes n_params from dimensions
    using the same formula as HookedTransformerConfig.__post_init__.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Estimated number of parameters

    Raises:
        Exception: If config cannot be fetched or parsed
    """
    from transformers import AutoConfig

    from transformer_lens.loading_from_pretrained import NEED_REMOTE_CODE_MODELS

    trust_remote_code = any(model_id.startswith(prefix) for prefix in NEED_REMOTE_CODE_MODELS)
    _token = os.environ.get("HF_TOKEN", "") or None
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=_token)

    # For multimodal models (LLaVA, Gemma3 multimodal), the language model config
    # is nested under text_config. Fall through to the top-level config otherwise.
    lang_config = getattr(config, "text_config", config)

    # Extract dimensions from config (different models use different attribute names)
    d_model = (
        getattr(lang_config, "hidden_size", None)
        or getattr(lang_config, "d_model", None)
        or getattr(lang_config, "model_dim", None)  # OpenELM
        or 0
    )
    n_heads_raw = (
        getattr(lang_config, "num_attention_heads", None)
        or getattr(lang_config, "n_head", None)
        or getattr(lang_config, "num_query_heads", None)  # OpenELM (may be per-layer list)
        or 0
    )
    # OpenELM uses per-layer lists for heads; take the max for estimation
    n_heads = max(n_heads_raw) if isinstance(n_heads_raw, (list, tuple)) else n_heads_raw
    n_layers = (
        getattr(lang_config, "num_hidden_layers", None)
        or getattr(lang_config, "n_layer", None)
        or getattr(lang_config, "num_transformer_layers", None)  # OpenELM
        or 0
    )
    d_mlp = (
        getattr(lang_config, "intermediate_size", None)
        or getattr(lang_config, "d_inner", None)
        or getattr(lang_config, "n_inner", None)
        or getattr(lang_config, "ffn_dim", None)  # OPT
        or getattr(lang_config, "d_ff", None)  # T5
    )
    # OpenELM uses per-layer ffn_multipliers instead of a fixed intermediate_size
    if not d_mlp and d_model:
        ffn_multipliers = getattr(lang_config, "ffn_multipliers", None)
        if isinstance(ffn_multipliers, (list, tuple)):
            d_mlp = int(max(ffn_multipliers) * d_model)
        else:
            # Many architectures (GPT-2, Bloom, GPT-Neo, GPT-J) leave d_mlp/n_inner
            # as None and default to 4 * hidden_size internally.
            d_mlp = 4 * d_model
    d_vocab = getattr(lang_config, "vocab_size", None) or 0

    if d_model == 0 or n_heads == 0 or n_layers == 0:
        raise ValueError(f"Could not extract model dimensions from config for {model_id}")

    d_head = getattr(lang_config, "head_dim", None) or (d_model // n_heads)

    # Attention parameters: W_Q, W_K, W_V, W_O per layer
    n_params = n_layers * (d_model * d_head * n_heads * 4)

    # MLP parameters (if present)
    if d_mlp is not None and d_mlp > 0:
        # Check for gated MLP (LLaMA, Gemma, Mistral, Qwen, T5 gated-gelu, etc.)
        has_gate = getattr(lang_config, "is_gated_act", False) or (
            hasattr(lang_config, "intermediate_size")
            and (
                getattr(lang_config, "hidden_act", None) in ("silu", "gelu", "swiglu")
                or getattr(lang_config, "model_type", None)
                in (
                    "llama",
                    "gemma",
                    "gemma2",
                    "gemma3",
                    "mistral",
                    "mixtral",
                    "qwen2",
                    "qwen3",
                    "phi3",
                    "stablelm",
                )
            )
        )
        mlp_multiplier = 3 if has_gate else 2
        n_params += n_layers * (d_model * d_mlp * mlp_multiplier)

        # MoE expert scaling
        num_experts = getattr(lang_config, "num_local_experts", None) or getattr(
            lang_config, "num_experts", None
        )
        if num_experts and num_experts > 1:
            # For MoE, MLP params are multiplied by num_experts + gate params
            mlp_per_layer = d_model * d_mlp * mlp_multiplier
            moe_per_layer = (mlp_per_layer + d_model) * num_experts
            # Replace the non-MoE MLP contribution
            n_params -= n_layers * (d_model * d_mlp * mlp_multiplier)
            n_params += n_layers * moe_per_layer

    # Embedding parameters (not in HookedTransformerConfig formula but relevant for memory)
    n_params += d_vocab * d_model

    return n_params


def estimate_benchmark_memory_gb(
    n_params: int,
    dtype: str = "float32",
    phases: Optional[list[int]] = None,
    conserve_memory: bool = False,
) -> float:
    """Estimate peak memory needed for benchmark suite.

    Phases run sequentially, so peak memory is the maximum of any single phase,
    not the sum. The multiplier represents how many model copies exist at peak:

    Phase 1 (conserve_memory=True):  Bridge only (uses bridge.original_model
        as reference) → 1.0x model + overhead
    Phase 1 (conserve_memory=False): Briefly loads HF ref + Bridge → 2.0x peak
    Phase 2: Bridge + HookedTransformer (separate copy) → 2.0x model + overhead
    Phase 3: Same as Phase 2 (processed versions) → 2.0x model + overhead
    Phase 4: Bridge + GPT-2 scorer (~500MB) → ~1.0x model + 0.5 GB

    Args:
        n_params: Number of model parameters
        dtype: Data type for memory calculation
        phases: Which phases will be run (None = all phases)
        conserve_memory: Whether --conserve-memory mode is enabled

    Returns:
        Estimated peak memory in GB
    """
    bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2}
    bpp = bytes_per_param.get(dtype, 4)
    model_size_gb = n_params * bpp / (1024**3)

    # GPT-2 scorer overhead (loaded during Phase 4)
    gpt2_overhead_gb = 0.5

    # Activation/framework overhead as a fraction of model size
    overhead_fraction = 0.2

    # Determine peak memory across all requested phases
    phase_peaks = []

    if phases is None:
        phases = [1, 2, 3, 4]

    for p in phases:
        if p == 1:
            copies = 1.0 if conserve_memory else 2.0
            phase_peaks.append(model_size_gb * copies * (1 + overhead_fraction))
        elif p in (2, 3):
            # Bridge + HookedTransformer = 2 full model copies
            copies = 2.0
            phase_peaks.append(model_size_gb * copies * (1 + overhead_fraction))
        elif p == 4:
            # Bridge + GPT-2 scorer
            phase_peaks.append(model_size_gb * (1 + overhead_fraction) + gpt2_overhead_gb)

    return max(phase_peaks) if phase_peaks else model_size_gb


def get_available_memory_gb(device: str) -> float:
    """Detect available memory on the target device.

    Args:
        device: "cpu" or "cuda"

    Returns:
        Available memory in GB
    """
    if device.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                device_idx = 0
                if ":" in device:
                    device_idx = int(device.split(":")[1])
                props = torch.cuda.get_device_properties(device_idx)
                return props.total_memory / (1024**3)
        except Exception:
            pass
        return 8.0  # Conservative default for GPU

    # CPU: use psutil if available, else conservative default
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        return 16.0  # Conservative default for CPU


def select_models_for_verification(
    per_arch: int = 10,
    architectures: Optional[list[str]] = None,
    limit: Optional[int] = None,
    resume_progress: Optional[VerificationProgress] = None,
    retry_failed: bool = False,
    reverify: bool = False,
) -> list[ModelCandidate]:
    """Select models for verification from the registry.

    Loads supported_models.json (already sorted by downloads).
    Takes the top N unverified models per architecture.

    Args:
        per_arch: Maximum models to verify per architecture
        architectures: Filter to specific architectures (None = all)
        limit: Total model cap (None = no cap)
        resume_progress: If resuming, skip already-tested models
        retry_failed: If True, include previously failed models for re-testing
        reverify: If True, ignore previous status and re-test all matching models

    Returns:
        List of ModelCandidate objects to verify
    """
    already_tested: set[str] = set()
    if resume_progress and not reverify:
        already_tested = set(resume_progress.tested)
        if retry_failed:
            # Remove failed models from already_tested so they get re-selected
            failed_set = set(resume_progress.failed)
            already_tested -= failed_set

    data = load_supported_models_raw()
    models = data.get("models", [])

    # Group by architecture
    by_arch: dict[str, list[dict]] = {}
    for model in models:
        arch = model["architecture_id"]
        by_arch.setdefault(arch, []).append(model)

    # Determine which architectures to scan
    if architectures:
        arch_ids = architectures
    else:
        arch_ids = sorted(by_arch.keys())

    candidates: list[ModelCandidate] = []

    for arch in arch_ids:
        arch_models = by_arch.get(arch, [])
        count = 0

        for model in arch_models:
            model_id = model["model_id"]

            # Skip already-verified or already-tested models
            if not reverify:
                model_status = model.get("status", 0)
                if model_status == STATUS_VERIFIED or model_status == STATUS_SKIPPED:
                    continue
                if model_status == STATUS_FAILED and not retry_failed:
                    continue
            if model_id in already_tested:
                continue

            # Check per-arch limit
            if count >= per_arch:
                break

            count += 1
            candidates.append(ModelCandidate(model_id=model_id, architecture_id=arch))

            # Check total limit
            if limit and len(candidates) >= limit:
                return candidates

    return candidates


def _extract_phase_scores(results: list) -> dict[int, Optional[float]]:
    """Extract phase scores from benchmark results.

    Mirrors the logic in update_model_registry() from main_benchmark.py.

    Args:
        results: List of BenchmarkResult objects

    Returns:
        Dict mapping phase number to score (0-100) or None
    """
    from transformer_lens.benchmarks.utils import BenchmarkSeverity

    phase_results: dict[int, list[bool]] = {1: [], 2: [], 3: [], 4: [], 7: []}
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


# Per-phase minimum score thresholds (0-100).
# Phase 1: Core correctness (bridge vs HF) — must pass everything.
# Phase 2: Hook/cache/gradient tests — most should pass.
# Phase 3: Weight processing tests — most should pass.
# Phase 4: Text quality — inherently fuzzy, keep lenient.
_MIN_PHASE_SCORES: dict[int, float] = {
    1: 100.0,
    2: 75.0,
    3: 75.0,
    4: 50.0,
    7: 75.0,
}
_DEFAULT_MIN_PHASE_SCORE = 50.0

# Architectures that include a vision encoder and require Phase 7 (multimodal
# benchmarks) as part of core verification.
_MULTIMODAL_ARCHITECTURES = {
    "LlavaForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
}

# Tests that MUST pass for a phase to be considered passing, regardless of
# the overall percentage score.  If any required test fails, the phase fails
# even if the score is above the minimum threshold.
_REQUIRED_PHASE_TESTS: dict[int, list[str]] = {
    2: ["logits_equivalence", "loss_equivalence"],
    3: ["logits_equivalence", "loss_equivalence"],
    7: ["multimodal_forward"],
}


def _check_phase_scores(
    phase_scores: dict[int, Optional[float]],
    all_results: list,
) -> Optional[str]:
    """Check phase scores against per-phase minimum thresholds and required tests.

    A phase fails if:
      1. Its overall score is below the minimum threshold, OR
      2. Any of its required tests (per _REQUIRED_PHASE_TESTS) failed.

    Phase 4 (text quality) is excluded — it is a quality metric, not a
    correctness check.  Low text quality is surfaced in the verification
    note via _build_verified_note() but never causes a model to fail.

    Returns an error message if any phase fails, or None if all phases pass.
    The message includes the names of failed tests.
    """
    from transformer_lens.benchmarks.utils import BenchmarkSeverity

    failing_phases: list[str] = []
    for phase, score in sorted(phase_scores.items()):
        if score is None:
            continue

        # Phase 4 is a quality metric, not a pass/fail check — skip it here.
        # Low text quality is reported in the note by _build_verified_note().
        if phase == 4:
            continue

        # Check 1: overall score threshold
        threshold = _MIN_PHASE_SCORES.get(phase, _DEFAULT_MIN_PHASE_SCORE)
        if score < threshold:
            failed_tests = [
                r.name
                for r in all_results
                if r.phase == phase and not r.passed and r.severity != BenchmarkSeverity.SKIPPED
            ]
            tests_str = ", ".join(failed_tests) if failed_tests else "unknown"
            failing_phases.append(f"P{phase}={score}% < {threshold}% (failed: {tests_str})")
            continue  # Already failing; no need to also check required tests

        # Check 2: required tests must pass
        required_tests = _REQUIRED_PHASE_TESTS.get(phase, [])
        if required_tests:
            failed_required = [
                r.name
                for r in all_results
                if r.phase == phase
                and r.name in required_tests
                and not r.passed
                and r.severity != BenchmarkSeverity.SKIPPED
            ]
            if failed_required:
                tests_str = ", ".join(failed_required)
                failing_phases.append(f"P{phase}={score}% but required tests failed: {tests_str}")

    if failing_phases:
        return f"Below threshold: {'; '.join(failing_phases)}"
    return None


def _build_verified_note(
    phase_scores: dict[int, Optional[float]],
    all_results: list,
) -> str:
    """Build a verification note summarizing phase scores.

    Phase 4 (text quality) is excluded from the score summary since it's a
    quality metric, not a pass/fail comparison. It only contributes a "low
    text quality" flag when below threshold.
    """
    from transformer_lens.benchmarks.utils import BenchmarkSeverity

    issue_parts: list[str] = []
    low_text_quality = False

    for phase in sorted(phase_scores):
        score = phase_scores[phase]
        if score is None:
            continue
        # Phase 4 is a quality score, not a pass/fail comparison — don't
        # include it in the normal score summary.
        if phase == 4:
            threshold = _MIN_PHASE_SCORES.get(4, _DEFAULT_MIN_PHASE_SCORE)
            if score < threshold:
                low_text_quality = True
            continue

        if score < 100.0:
            failed_tests = [
                r.name
                for r in all_results
                if r.phase == phase and not r.passed and r.severity != BenchmarkSeverity.SKIPPED
            ]
            if failed_tests:
                issue_parts.append(f"P{phase}={score}% (failed: {', '.join(failed_tests)})")
            else:
                issue_parts.append(f"P{phase}={score}%")

    if issue_parts and low_text_quality:
        return (
            f"Full verification completed with issues, low text quality: {'; '.join(issue_parts)}"
        )
    if issue_parts:
        return f"Full verification completed with issues: {'; '.join(issue_parts)}"
    if low_text_quality:
        return "Full verification completed with issues, low text quality"
    return "Full verification completed"


def _clear_hf_cache(quiet: bool = False) -> None:
    """Remove downloaded model weights from the HuggingFace cache to free disk."""
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return

    freed = 0
    for blobs_dir in cache_dir.glob("models--*/blobs"):
        for blob in blobs_dir.iterdir():
            try:
                size = blob.stat().st_size
                blob.unlink()
                freed += size
            except OSError:
                pass

    if not quiet and freed > 0:
        print(f"  Cleared {freed / (1024**3):.1f} GB from HuggingFace cache")


def _save_checkpoint(progress: VerificationProgress) -> None:
    """Save verification progress to checkpoint file."""
    with open(_CHECKPOINT_PATH, "w") as f:
        json.dump(progress.to_dict(), f, indent=2)
        f.write("\n")


def _load_checkpoint() -> Optional[VerificationProgress]:
    """Load verification progress from checkpoint file."""
    if not _CHECKPOINT_PATH.exists():
        return None
    try:
        with open(_CHECKPOINT_PATH) as f:
            data = json.load(f)
        return VerificationProgress.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def verify_models(
    candidates: list[ModelCandidate],
    device: str = "cpu",
    max_memory_gb: Optional[float] = None,
    dtype: str = "float32",
    use_hf_reference: bool = True,
    use_ht_reference: bool = True,
    phases: Optional[list[int]] = None,
    quiet: bool = False,
    progress: Optional[VerificationProgress] = None,
    conserve_memory: bool = False,
) -> VerificationProgress:
    """Run verification benchmarks on a list of model candidates.

    Args:
        candidates: Models to verify
        device: Device for benchmarks
        max_memory_gb: Memory limit (auto-detected if None)
        dtype: Dtype for memory estimation
        use_hf_reference: Whether to compare against HuggingFace model
        use_ht_reference: Whether to compare against HookedTransformer
        phases: Which benchmark phases to run (default: [1, 2, 3, 4])
        quiet: Suppress verbose output
        progress: Existing progress for resume
        conserve_memory: Reduce Phase 1 peak memory by using bridge.original_model

    Returns:
        VerificationProgress with results
    """
    from transformer_lens.benchmarks.main_benchmark import run_benchmark_suite

    if progress is None:
        progress = VerificationProgress(start_time=datetime.now().isoformat())

    if max_memory_gb is None:
        max_memory_gb = get_available_memory_gb(device)
        if not quiet:
            print(f"Auto-detected available memory: {max_memory_gb:.1f} GB")

    if phases is None:
        phases = [1, 2, 3, 4]

    # Pre-load the GPT-2 scoring model for Phase 4 so it persists across all
    # models in the batch instead of being loaded and destroyed for each one.
    _scoring_model = None
    _scoring_tokenizer = None
    if 4 in phases:
        try:
            from transformer_lens.benchmarks.text_quality import _load_scoring_model

            _scoring_model, _scoring_tokenizer = _load_scoring_model("gpt2", device)
            if not quiet:
                print("Pre-loaded GPT-2 scoring model for Phase 4")
        except Exception as e:
            if not quiet:
                print(f"Warning: Could not pre-load GPT-2 scorer: {e}")
                print("  Phase 4 will load its own scorer per model.")

    total = len(candidates)
    for i, candidate in enumerate(candidates, 1):
        # Check for graceful interrupt between models
        if _interrupt_requested:
            if not quiet:
                print(f"\nStopping gracefully. Progress saved ({len(progress.verified)} verified).")
            _save_checkpoint(progress)
            raise SystemExit(_EXIT_GRACEFUL_INTERRUPT)

        model_id = candidate.model_id
        arch = candidate.architecture_id

        if not quiet:
            print(f"\n{'='*70}")
            print(f"[{i}/{total}] {model_id} ({arch})")
            print(f"{'='*70}")

        progress.tested.append(model_id)

        # Step 0: Check for quantized models (fundamentally incompatible)
        if is_quantized_model(model_id):
            if not quiet:
                print(f"  SKIP: {QUANTIZED_NOTE}")
            current_status = _get_current_model_status(model_id, arch)
            if current_status != STATUS_VERIFIED:
                update_model_status(model_id, arch, STATUS_SKIPPED, note=QUANTIZED_NOTE)
            elif not quiet:
                print(f"  (preserving existing verified status)")
            progress.skipped.append(model_id)
            _save_checkpoint(progress)
            continue

        # Step 1: Estimate parameters
        try:
            n_params = estimate_model_params(model_id)
            candidate.estimated_params = n_params
            if not quiet:
                print(f"  Estimated parameters: {n_params:,}")
        except Exception as e:
            note = f"Config unavailable: {str(e)[:200]}"
            if not quiet:
                print(f"  SKIP: {note}")
            # Don't downgrade previously verified models to SKIPPED
            # If a model is verified, we assume it still runs even though
            # it is below the memory limit of the current run
            current_status = _get_current_model_status(model_id, arch)
            if current_status != STATUS_VERIFIED:
                update_model_status(
                    model_id, arch, STATUS_SKIPPED, note=note, sanitize_fn=_sanitize_note
                )
            elif not quiet:
                print(f"  (preserving existing verified status)")
            progress.skipped.append(model_id)
            _save_checkpoint(progress)
            continue

        # Step 2: Check memory
        estimated_mem = estimate_benchmark_memory_gb(
            n_params, dtype, phases=phases, conserve_memory=conserve_memory
        )
        candidate.estimated_memory_gb = estimated_mem
        if not quiet:
            print(
                f"  Estimated benchmark memory: {estimated_mem:.1f} GB (limit: {max_memory_gb:.1f} GB)"
            )

        if estimated_mem > max_memory_gb:
            note = f"Estimated {estimated_mem:.1f} GB exceeds {max_memory_gb:.1f} GB limit"
            if not quiet:
                print(f"  SKIP: {note}")
            # Don't downgrade previously verified models to SKIPPED
            # If a model is verified, we assume it still runs even though
            # it is below the memory limit of the current run
            current_status = _get_current_model_status(model_id, arch)
            if current_status != STATUS_VERIFIED:
                update_model_status(
                    model_id, arch, STATUS_SKIPPED, note=note, sanitize_fn=_sanitize_note
                )
            elif not quiet:
                print(f"  (preserving existing verified status)")
            progress.skipped.append(model_id)
            _save_checkpoint(progress)
            continue

        # Step 3: Run benchmarks (all phases in a single call to share models)
        all_results: list = []
        error_msg: Optional[str] = None

        from transformer_lens.loading_from_pretrained import NEED_REMOTE_CODE_MODELS

        needs_remote_code = any(model_id.startswith(prefix) for prefix in NEED_REMOTE_CODE_MODELS)

        # Convert string dtype to torch.dtype for benchmark suite
        import torch

        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = _dtype_map[dtype]

        # Multimodal models always use conserve_memory to avoid loading two
        # large models simultaneously (causes MPS memory-pressure divergence).
        effective_conserve_memory = conserve_memory or arch in _MULTIMODAL_ARCHITECTURES

        if not quiet:
            print(f"  Running phases {phases} in a single benchmark call...")
        try:
            all_results = run_benchmark_suite(
                model_id,
                device=device,
                dtype=torch_dtype,
                use_hf_reference=use_hf_reference,
                use_ht_reference=use_ht_reference,
                verbose=not quiet,
                phases=phases,
                conserve_memory=effective_conserve_memory,
                trust_remote_code=needs_remote_code,
                scoring_model=_scoring_model,
                scoring_tokenizer=_scoring_tokenizer,
            )
        except Exception as e:
            error_msg = str(e)
            if not quiet:
                print(f"  Benchmark failed: {error_msg[:200]}")

        phase_scores = _extract_phase_scores(all_results)

        if not error_msg:
            score_error = _check_phase_scores(phase_scores, all_results)
            if score_error:
                error_msg = score_error

        if error_msg:
            is_oom = "out of memory" in error_msg.lower() or "oom" in error_msg.lower()
            if is_oom:
                note = "OOM during benchmark"
            else:
                # Include the specific error from failed results (e.g., tokenizer
                # errors, load failures) so the note explains WHY it failed.
                root_errors = [r.message for r in all_results if not r.passed and r.message]
                if root_errors:
                    # Deduplicate and use first unique error as the detail
                    unique_errors = list(dict.fromkeys(root_errors))
                    detail = unique_errors[0][:150]
                    note = f"{error_msg[:100]} — {detail}"
                else:
                    note = error_msg[:200]
            final_status = STATUS_FAILED
        else:
            note = _build_verified_note(phase_scores, all_results)
            final_status = STATUS_VERIFIED

        # When running a partial phase set (e.g., --phases 4 for backfill),
        # only update the phase scores that were run.  Don't change the
        # model's overall status or note — those reflect the full
        # verification and should only be set by a complete run.
        is_multimodal = arch in _MULTIMODAL_ARCHITECTURES
        # For multimodal models, Phase 7 is part of core verification.
        # A full run is {1,2,3,4,7} for multimodal, {1,2,3,4} for text-only.
        full_phases = {1, 2, 3, 4, 7} if is_multimodal else {1, 2, 3, 4}
        core_required = {1, 4, 7} if is_multimodal else {1, 4}
        is_partial_run = set(phases) != full_phases

        if is_partial_run and phase_scores:
            # Only write scores for phases that were actually requested.
            # Bridge load failures can produce Phase 1-tagged error results
            # even during Phase 4-only runs — don't let those corrupt
            # existing scores for unrequested phases.
            filtered_scores = {p: s for p, s in phase_scores.items() if p in phases}
            if filtered_scores:
                if not quiet:
                    score_parts = [f"P{p}={s}%" for p, s in sorted(filtered_scores.items())]
                    print(f"  Partial phase update: {', '.join(score_parts)}")

                # Core verification: P1+P4 for text-only, P1+P4+P7 for multimodal.
                is_core_verification = set(phases) >= core_required
                partial_status = None
                partial_note = None

                if is_core_verification:
                    p1 = filtered_scores.get(1)
                    p4 = filtered_scores.get(4)
                    p1_pass = p1 is not None and p1 >= _MIN_PHASE_SCORES.get(
                        1, _DEFAULT_MIN_PHASE_SCORE
                    )
                    p4_pass = p4 is not None and p4 >= _MIN_PHASE_SCORES.get(
                        4, _DEFAULT_MIN_PHASE_SCORE
                    )

                    # For multimodal, also require Phase 7 to pass.
                    # If P7 was requested but all tests were skipped (e.g., no
                    # processor for a community model), treat it as a soft pass
                    # rather than a failure.
                    p7_pass = True
                    p7_skipped = False
                    if is_multimodal:
                        p7 = filtered_scores.get(7)
                        if p7 is not None:
                            p7_pass = p7 >= _MIN_PHASE_SCORES.get(7, _DEFAULT_MIN_PHASE_SCORE)
                        elif 7 in phases:
                            # Phase 7 requested but no score — all tests skipped
                            p7_skipped = True
                        else:
                            # Phase 7 not even requested
                            p7_pass = False

                    if p1_pass and p4_pass and p7_pass and not p7_skipped:
                        partial_status = STATUS_VERIFIED
                        partial_note = "Core verification completed"
                    elif p1_pass and p4_pass and p7_skipped:
                        partial_status = STATUS_VERIFIED
                        partial_note = (
                            "Core verification completed (multimodal tests skipped — no processor)"
                        )
                    elif p1_pass and p4_pass and not p7_pass:
                        partial_status = STATUS_VERIFIED
                        partial_note = (
                            "Core verification passed, but multimodal tests failed. Needs review"
                        )
                    elif p1_pass:
                        partial_status = STATUS_VERIFIED
                        partial_note = (
                            "Core verification passed, but text quality poor. Needs review"
                        )
                    else:
                        # P1 failed — build a descriptive failure note
                        partial_status = STATUS_FAILED
                        if error_msg:
                            partial_note = f"CORE FAILED: {error_msg[:200]}"
                        else:
                            # Score-based failure — include details
                            from transformer_lens.benchmarks.utils import (
                                BenchmarkSeverity,
                            )

                            failed_tests = [
                                r.name
                                for r in all_results
                                if r.phase == 1
                                and not r.passed
                                and r.severity != BenchmarkSeverity.SKIPPED
                            ]
                            tests_str = ", ".join(failed_tests) if failed_tests else "unknown"
                            partial_note = f"CORE FAILED: P1={p1}% (failed: {tests_str})"

                    if not quiet:
                        print(f"  {partial_note}")

                update_model_status(
                    model_id,
                    arch,
                    status=partial_status,
                    phase_scores=filtered_scores,
                    note=partial_note,
                )
                if partial_status == STATUS_FAILED:
                    progress.failed.append(model_id)
                else:
                    progress.verified.append(model_id)
            else:
                if not quiet:
                    print(f"  No results for requested phases {phases} — skipping update")
                progress.skipped.append(model_id)
        elif final_status == STATUS_VERIFIED:
            if not quiet:
                print(
                    f"  VERIFIED: P1={phase_scores.get(1)}%, "
                    f"P2={phase_scores.get(2)}%, P3={phase_scores.get(3)}%, "
                    f"P4={phase_scores.get(4)}%, P7={phase_scores.get(7)}%"
                )
            update_model_status(
                model_id,
                arch,
                STATUS_VERIFIED,
                phase_scores=phase_scores,
                note=note,
            )
            add_verification_record(
                model_id,
                arch,
                notes=note,
            )
            progress.verified.append(model_id)
        else:
            if not quiet:
                print(f"  FAILED: {note}")
                if any(v is not None for v in phase_scores.values()):
                    print(
                        f"  Partial scores saved: P1={phase_scores.get(1)}%, "
                        f"P2={phase_scores.get(2)}%, P3={phase_scores.get(3)}%, "
                        f"P4={phase_scores.get(4)}%"
                    )
            update_model_status(
                model_id,
                arch,
                STATUS_FAILED,
                note=note,
                phase_scores=phase_scores,
                sanitize_fn=_sanitize_note,
            )
            add_verification_record(
                model_id,
                arch,
                notes=note,
                sanitize_fn=_sanitize_note,
            )
            progress.failed.append(model_id)

        # Post-model cleanup
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.synchronize()
                torch.mps.empty_cache()

            # Log MPS memory state for debugging long runs
            if device == "mps" and not quiet and hasattr(torch.mps, "current_allocated_memory"):
                alloc_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                driver_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
                print(f"  MPS memory: {alloc_mb:.0f} MB allocated, " f"{driver_mb:.0f} MB driver")
        except ImportError:
            pass

        # Brief pause to let the OS and MPS reclaim memory between models
        if device in ("mps", "cuda"):
            time.sleep(3)

        # Periodically clear the HuggingFace cache to prevent disk exhaustion
        if i % 50 == 0:
            _clear_hf_cache(quiet)

        _save_checkpoint(progress)

    # Clean up pre-loaded scoring model
    if _scoring_model is not None:
        del _scoring_model
        del _scoring_tokenizer
        gc.collect()

    return progress


def _print_dry_run(
    candidates: list[ModelCandidate],
    dtype: str,
    max_memory_gb: float,
    phases: Optional[list[int]] = None,
    conserve_memory: bool = False,
) -> None:
    """Print what would be tested in a dry run."""
    print(f"\nDry run: {len(candidates)} models would be tested")
    print(f"Memory limit: {max_memory_gb:.1f} GB | Dtype: {dtype}")
    print()

    # Group by architecture
    by_arch: dict[str, list[ModelCandidate]] = {}
    for c in candidates:
        by_arch.setdefault(c.architecture_id, []).append(c)

    skippable = 0
    testable = 0

    for arch in sorted(by_arch.keys()):
        models = by_arch[arch]
        print(f"  {arch} ({len(models)} models):")
        for c in models:
            try:
                n_params = estimate_model_params(c.model_id)
                mem = estimate_benchmark_memory_gb(
                    n_params, dtype, phases=phases, conserve_memory=conserve_memory
                )
                status = "OK" if mem <= max_memory_gb else "SKIP (too large)"
                if mem > max_memory_gb:
                    skippable += 1
                else:
                    testable += 1
                print(f"    {c.model_id}: ~{n_params/1e6:.0f}M params, ~{mem:.1f} GB [{status}]")
            except Exception as e:
                skippable += 1
                print(f"    {c.model_id}: config error ({e})")
        print()

    print(f"Summary: {testable} testable, {skippable} would be skipped")


def _print_summary(progress: VerificationProgress) -> None:
    """Print a summary of the verification run."""
    total = len(progress.tested)
    print(f"\n{'='*70}")
    print("Verification Summary")
    print(f"{'='*70}")
    print(f"  Total tested:  {total}")
    print(f"  Verified:      {len(progress.verified)}")
    print(f"  Skipped:       {len(progress.skipped)}")
    print(f"  Failed:        {len(progress.failed)}")

    if progress.verified:
        print(f"\n  Verified models:")
        for m in progress.verified:
            print(f"    - {m}")

    if progress.failed:
        print(f"\n  Failed models:")
        for m in progress.failed:
            print(f"    - {m}")

    if progress.skipped:
        print(f"\n  Skipped models:")
        for m in progress.skipped[:20]:
            print(f"    - {m}")
        if len(progress.skipped) > 20:
            print(f"    ... and {len(progress.skipped) - 20} more")


def main() -> None:
    """CLI entry point for batch model verification."""
    parser = argparse.ArgumentParser(
        description="Batch verify models in the TransformerLens registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                    Show what would be tested
  %(prog)s --limit 3                    Test 3 models total
  %(prog)s --architectures GPT2LMHeadModel --per-arch 5
  %(prog)s --device cuda --max-memory 24
  %(prog)s --resume                     Resume from checkpoint
  %(prog)s --reverify --architectures Olmo2ForCausalLM   Re-verify already-tested models
  %(prog)s --model google/gemma-2b      Verify a single model by ID
        """,
    )
    parser.add_argument(
        "--per-arch",
        type=int,
        default=10,
        help="Max models to verify per architecture (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for benchmarks (default: cpu)",
    )
    parser.add_argument(
        "--max-memory",
        type=float,
        default=None,
        help="Memory limit in GB (default: auto-detect)",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=None,
        help="Filter to specific architectures",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Total model cap",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be tested without running benchmarks",
    )
    parser.add_argument(
        "--no-hf-reference",
        action="store_true",
        help="Skip HuggingFace reference comparison",
    )
    parser.add_argument(
        "--no-ht-reference",
        action="store_true",
        help="Skip HookedTransformer reference comparison",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=None,
        help="Which benchmark phases to run (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype for memory estimation (default: float32)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run previously failed models instead of skipping them",
    )
    parser.add_argument(
        "--conserve-memory",
        action="store_true",
        help="Reduce Phase 1 peak memory from 2.0x to 1.0x by using "
        "bridge.original_model instead of a separate HF model",
    )
    parser.add_argument(
        "--reverify",
        action="store_true",
        help="Re-run verification for already-verified/skipped/failed models. "
        "Ignores previous status and re-tests matching models from scratch.",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=None,
        help="Verify one or more models by HuggingFace model ID. "
        "Looks up architecture from the registry automatically.",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Auto-detect memory
    max_memory_gb = args.max_memory
    if max_memory_gb is None:
        max_memory_gb = get_available_memory_gb(args.device)

    # Load checkpoint if resuming
    progress = None
    if args.resume:
        progress = _load_checkpoint()
        if progress:
            print(f"Resuming from checkpoint: {len(progress.tested)} models already tested")
        else:
            print("No checkpoint found, starting fresh")

    # If retrying failed, clean them from checkpoint and reset status in registry
    if args.retry_failed and progress and not args.dry_run:
        failed_set = set(progress.failed)
        if failed_set:
            # Reset status in supported_models.json
            registry_data = load_supported_models_raw()
            for entry in registry_data.get("models", []):
                if entry["model_id"] in failed_set and entry.get("status") == STATUS_FAILED:
                    update_model_status(
                        entry["model_id"],
                        entry["architecture_id"],
                        STATUS_UNVERIFIED,
                    )
            # Clean checkpoint
            progress.tested = [m for m in progress.tested if m not in failed_set]
            progress.failed = []
            _save_checkpoint(progress)
            print(f"  Cleared {len(failed_set)} failed models for retry")

    # Select models — either --model list or the normal batch selection
    if args.model:
        # Look up architecture for each model from the registry
        registry_data = load_supported_models_raw()
        candidates = []
        for model_id in args.model:
            arch_id = None
            for entry in registry_data.get("models", []):
                if entry["model_id"] == model_id:
                    arch_id = entry["architecture_id"]
                    break
            if arch_id is None:
                print(f"Model '{model_id}' not found in supported_models.json, skipping")
                continue
            candidates.append(ModelCandidate(model_id=model_id, architecture_id=arch_id))
        if not candidates:
            print("No valid models found in registry")
            return
        print(f"Model list mode: {len(candidates)} model(s)")
    else:
        candidates = select_models_for_verification(
            per_arch=args.per_arch,
            architectures=args.architectures,
            limit=args.limit,
            resume_progress=progress,
            retry_failed=args.retry_failed,
            reverify=args.reverify,
        )

    if not candidates:
        print("No models to verify (all matching models already tested)")
        return

    print(f"Selected {len(candidates)} models for verification")

    # Dry run
    if args.dry_run:
        _print_dry_run(
            candidates,
            args.dtype,
            max_memory_gb,
            phases=args.phases,
            conserve_memory=args.conserve_memory,
        )
        return

    # Install graceful interrupt handler (Ctrl+C stops between models)
    signal.signal(signal.SIGINT, _handle_sigint)

    # Run verification
    start = time.time()
    progress = verify_models(
        candidates,
        device=args.device,
        max_memory_gb=max_memory_gb,
        dtype=args.dtype,
        use_hf_reference=not args.no_hf_reference,
        use_ht_reference=not args.no_ht_reference,
        phases=args.phases,
        quiet=args.quiet,
        progress=progress,
        conserve_memory=args.conserve_memory,
    )
    elapsed = time.time() - start

    _print_summary(progress)
    print(f"\nTotal time: {elapsed:.1f}s")

    # Clean up checkpoint on successful completion
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        print("Checkpoint cleared (run complete)")


if __name__ == "__main__":
    main()
