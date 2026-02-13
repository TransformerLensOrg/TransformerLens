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
"""

import argparse
import gc
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Data directory for registry files
_DATA_DIR = Path(__file__).parent / "data"
_CHECKPOINT_PATH = _DATA_DIR / "verification_checkpoint.json"
_SUPPORTED_MODELS_PATH = _DATA_DIR / "supported_models.json"
_VERIFICATION_HISTORY_PATH = _DATA_DIR / "verification_history.json"

# Status codes matching ModelEntry schema
STATUS_UNVERIFIED = 0
STATUS_VERIFIED = 1
STATUS_SKIPPED = 2
STATUS_FAILED = 3

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

    config = AutoConfig.from_pretrained(model_id)

    # Extract dimensions from config (different models use different attribute names)
    d_model = getattr(config, "hidden_size", None) or getattr(config, "d_model", None) or 0
    n_heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None) or 0
    n_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None) or 0
    d_mlp = (
        getattr(config, "intermediate_size", None)
        or getattr(config, "d_inner", None)
        or getattr(config, "n_inner", None)
    )
    d_vocab = getattr(config, "vocab_size", None) or 0

    if d_model == 0 or n_heads == 0 or n_layers == 0:
        raise ValueError(f"Could not extract model dimensions from config for {model_id}")

    d_head = d_model // n_heads

    # Attention parameters: W_Q, W_K, W_V, W_O per layer
    n_params = n_layers * (d_model * d_head * n_heads * 4)

    # MLP parameters (if present)
    if d_mlp is not None and d_mlp > 0:
        # Check for gated MLP (LLaMA, Gemma, Mistral, Qwen, etc.)
        has_gate = hasattr(config, "intermediate_size") and (
            getattr(config, "hidden_act", None) in ("silu", "gelu", "swiglu")
            or getattr(config, "model_type", None)
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
        mlp_multiplier = 3 if has_gate else 2
        n_params += n_layers * (d_model * d_mlp * mlp_multiplier)

        # MoE expert scaling
        num_experts = getattr(config, "num_local_experts", None) or getattr(
            config, "num_experts", None
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


def estimate_benchmark_memory_gb(n_params: int, dtype: str = "float32") -> float:
    """Estimate peak memory needed for benchmark suite.

    During benchmarks, up to 3 model copies may be in memory simultaneously
    (HF reference + Bridge + HookedTransformer), plus overhead for activations.

    Args:
        n_params: Number of model parameters
        dtype: Data type for memory calculation

    Returns:
        Estimated peak memory in GB
    """
    bytes_per_param = {"float32": 4, "float16": 2, "bfloat16": 2}
    bpp = bytes_per_param.get(dtype, 4)
    # 3.5x multiplier: 3 model copies + activation/gradient overhead
    return n_params * bpp * 3.5 / (1024**3)


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


def _load_supported_models() -> dict:
    """Load the supported_models.json file.

    Returns:
        Parsed JSON data as a dictionary
    """
    with open(_SUPPORTED_MODELS_PATH) as f:
        return json.load(f)


def _save_supported_models(data: dict) -> None:
    """Save data back to supported_models.json.

    Args:
        data: The full report dict to write
    """
    with open(_SUPPORTED_MODELS_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def select_models_for_verification(
    per_arch: int = 10,
    architectures: Optional[list[str]] = None,
    limit: Optional[int] = None,
    resume_progress: Optional[VerificationProgress] = None,
) -> list[ModelCandidate]:
    """Select models for verification from the registry.

    Loads supported_models.json (already sorted by downloads).
    Takes the top N unverified models per architecture.

    Args:
        per_arch: Maximum models to verify per architecture
        architectures: Filter to specific architectures (None = all)
        limit: Total model cap (None = no cap)
        resume_progress: If resuming, skip already-tested models

    Returns:
        List of ModelCandidate objects to verify
    """
    already_tested: set[str] = set()
    if resume_progress:
        already_tested = set(resume_progress.tested)

    data = _load_supported_models()
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
            if model.get("status", 0) != STATUS_UNVERIFIED:
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

    phase_results: dict[int, list[bool]] = {1: [], 2: [], 3: []}
    for result in results:
        if result.phase in phase_results and result.severity != BenchmarkSeverity.SKIPPED:
            phase_results[result.phase].append(result.passed)

    scores: dict[int, Optional[float]] = {}
    for phase, passed_list in phase_results.items():
        if passed_list:
            scores[phase] = round(sum(passed_list) / len(passed_list) * 100, 1)
        else:
            scores[phase] = None

    return scores


def _update_registry_entry(
    model_id: str,
    arch_id: str,
    status: int,
    phase_scores: dict[int, Optional[float]],
    note: Optional[str] = None,
) -> bool:
    """Update a single model entry in supported_models.json.

    Args:
        model_id: The model to update
        arch_id: Architecture of the model
        status: New status code (0-3)
        phase_scores: Phase score dict {1: float, 2: float, 3: float}
        note: Optional note for skip/fail reason

    Returns:
        True if entry was found and updated
    """
    data = _load_supported_models()
    updated = False

    for entry in data.get("models", []):
        if entry["model_id"] == model_id and entry["architecture_id"] == arch_id:
            entry["status"] = status
            entry["verified_date"] = (
                date.today().isoformat() if status != STATUS_UNVERIFIED else None
            )
            entry["note"] = _sanitize_note(note)
            entry["phase1_score"] = phase_scores.get(1)
            entry["phase2_score"] = phase_scores.get(2)
            entry["phase3_score"] = phase_scores.get(3)
            updated = True
            break

    if updated:
        # Update total_verified count
        data["total_verified"] = sum(1 for m in data.get("models", []) if m.get("status", 0) == 1)
        _save_supported_models(data)

    return updated


def _update_verification_history(
    model_id: str,
    architecture_id: str,
    notes: Optional[str] = None,
) -> None:
    """Append a VerificationRecord to the verification history file.

    Args:
        model_id: The verified model
        architecture_id: Architecture type
        notes: Optional verification notes
    """
    # Get TransformerLens version
    tl_version = None
    try:
        import transformer_lens

        tl_version = getattr(transformer_lens, "__version__", None)
    except Exception:
        pass

    record = {
        "model_id": model_id,
        "architecture_id": architecture_id,
        "verified_date": date.today().isoformat(),
        "verified_by": "verify_models",
        "transformerlens_version": tl_version,
        "notes": _sanitize_note(notes),
        "invalidated": False,
        "invalidation_reason": None,
    }

    # Load existing history
    if _VERIFICATION_HISTORY_PATH.exists():
        with open(_VERIFICATION_HISTORY_PATH) as f:
            history = json.load(f)
    else:
        history = {"last_updated": None, "records": []}

    history["records"].append(record)
    history["last_updated"] = datetime.now().isoformat()

    with open(_VERIFICATION_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
        f.write("\n")


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
) -> VerificationProgress:
    """Run verification benchmarks on a list of model candidates.

    Args:
        candidates: Models to verify
        device: Device for benchmarks
        max_memory_gb: Memory limit (auto-detected if None)
        dtype: Dtype for memory estimation
        use_hf_reference: Whether to compare against HuggingFace model
        use_ht_reference: Whether to compare against HookedTransformer
        phases: Which benchmark phases to run (default: [1, 2, 3])
        quiet: Suppress verbose output
        progress: Existing progress for resume

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
        phases = [1, 2, 3]

    total = len(candidates)
    for i, candidate in enumerate(candidates, 1):
        model_id = candidate.model_id
        arch = candidate.architecture_id

        if not quiet:
            print(f"\n{'='*70}")
            print(f"[{i}/{total}] {model_id} ({arch})")
            print(f"{'='*70}")

        progress.tested.append(model_id)

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
            _update_registry_entry(model_id, arch, STATUS_SKIPPED, {}, note=note)
            progress.skipped.append(model_id)
            _save_checkpoint(progress)
            continue

        # Step 2: Check memory
        estimated_mem = estimate_benchmark_memory_gb(n_params, dtype)
        candidate.estimated_memory_gb = estimated_mem
        if not quiet:
            print(
                f"  Estimated benchmark memory: {estimated_mem:.1f} GB (limit: {max_memory_gb:.1f} GB)"
            )

        if estimated_mem > max_memory_gb:
            note = f"Estimated {estimated_mem:.1f} GB exceeds {max_memory_gb:.1f} GB limit"
            if not quiet:
                print(f"  SKIP: {note}")
            _update_registry_entry(model_id, arch, STATUS_SKIPPED, {}, note=note)
            progress.skipped.append(model_id)
            _save_checkpoint(progress)
            continue

        # Step 3: Run benchmarks phase by phase for partial result capture
        all_results: list = []
        failed_phase: Optional[int] = None
        error_msg: Optional[str] = None

        for phase_num in phases:
            if not quiet:
                print(f"  Running phase {phase_num}...")
            try:
                phase_results = run_benchmark_suite(
                    model_id,
                    device=device,
                    use_hf_reference=use_hf_reference,
                    use_ht_reference=use_ht_reference,
                    verbose=not quiet,
                    phases=[phase_num],
                )
                all_results.extend(phase_results)
            except Exception as e:
                failed_phase = phase_num
                error_msg = str(e)
                if not quiet:
                    print(f"  Phase {phase_num} failed: {error_msg[:200]}")
                break

        # Extract phase scores from whatever results we have
        phase_scores = _extract_phase_scores(all_results)

        if error_msg:
            is_oom = "out of memory" in error_msg.lower() or "oom" in error_msg.lower()
            if is_oom:
                note = f"OOM during phase {failed_phase}"
            else:
                note = f"Error during phase {failed_phase}: {error_msg[:200]}"
            if not quiet:
                print(f"  FAILED: {note}")
                if any(v is not None for v in phase_scores.values()):
                    print(
                        f"  Partial scores saved: P1={phase_scores.get(1)}%, P2={phase_scores.get(2)}%, P3={phase_scores.get(3)}%"
                    )
            _update_registry_entry(model_id, arch, STATUS_FAILED, phase_scores, note=note)
            _update_verification_history(model_id, arch, notes=note)
            progress.failed.append(model_id)
        else:
            if not quiet:
                print(
                    f"  VERIFIED: P1={phase_scores.get(1)}%, P2={phase_scores.get(2)}%, P3={phase_scores.get(3)}%"
                )
            _update_registry_entry(model_id, arch, STATUS_VERIFIED, phase_scores)
            _update_verification_history(model_id, arch, notes="Benchmark passed")
            progress.verified.append(model_id)

        # Cleanup between models
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        _save_checkpoint(progress)

    return progress


def _print_dry_run(candidates: list[ModelCandidate], dtype: str, max_memory_gb: float) -> None:
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
                mem = estimate_benchmark_memory_gb(n_params, dtype)
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
        help="Which benchmark phases to run (default: 1 2 3)",
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

    # Select models
    candidates = select_models_for_verification(
        per_arch=args.per_arch,
        architectures=args.architectures,
        limit=args.limit,
        resume_progress=progress,
    )

    if not candidates:
        print("No models to verify (all matching models already tested)")
        return

    print(f"Selected {len(candidates)} models for verification")

    # Dry run
    if args.dry_run:
        _print_dry_run(candidates, args.dtype, max_memory_gb)
        return

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
