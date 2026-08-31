"""Re-run Phase 4 across the models `verify_models` has already verified.

`phase4_score` is a mixed-scale column: entries stamped `p4_scoring_version=2`
were measured on the pinned-judge reference-ratio scale, unstamped ones on the
old GPT-2 absolute-perplexity scale. This driver walks the stale population and
re-measures it, one model per subprocess:

    uv run python scripts/phase4_rerun.py --dry-run
    uv run python scripts/phase4_rerun.py --device cuda --max-memory 16

Each model runs as its own `verify_models --model <id> --phases 4` process, so a
crash, OOM or hang costs one model rather than the run. A phase-4-only run writes
`phase4_score` and the scoring-version stamp and leaves `status` untouched — this
never promotes or demotes a model.

Between models the HF hub cache is garbage-collected (repos this run downloaded
are deleted once their model is scored) so a 1000-model sweep does not fill the
disk, and every non-clean outcome raises a warning: crashes, models the verifier
skipped, scores under the pass line, regressions against a score measured on the
same scale, and models that have gone quiet for `--stall-after` seconds. Nothing
is killed on a clock -- a slow model and a hung one look alike from outside, and
only the watcher can tell them apart. Warnings stream to `warnings.log` in the
state dir as they happen; `--abort-after` stops the run on a streak of hard
failures.

Read `scripts/phase4_review.py` first for the shape of the backlog.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Run against the tree this script lives in. In a worktree the editable install
# still resolves transformer_lens from the main checkout, and the scores would
# then be written into that checkout's registry instead of this branch's.
sys.path.insert(0, str(_REPO_ROOT))

from transformer_lens.benchmarks.text_quality import JUDGE_MODEL_ID  # noqa: E402
from transformer_lens.benchmarks.text_quality_profiles import (  # noqa: E402
    P4_SCORING_VERSION,
    p4_pass_threshold,
)
from transformer_lens.tools.model_registry.registry_io import (  # noqa: E402
    STATUS_PROVISIONAL,
    STATUS_VERIFIED,
    is_incompatible_quantized,
    load_supported_models_raw,
)
from transformer_lens.tools.model_registry.verify_models import (  # noqa: E402
    _full_and_core_phases,
)
from transformer_lens.utilities.architectures import classify_architecture  # noqa: E402

import transformer_lens  # noqa: E402  isort:skip

if not str(Path(transformer_lens.__file__).resolve()).startswith(str(_REPO_ROOT)):
    raise SystemExit(
        f"transformer_lens resolves to {transformer_lens.__file__}, outside {_REPO_ROOT} — "
        f"refusing to write another tree's registry"
    )

_DEFAULT_STATE_DIR = _REPO_ROOT / ".claude" / "reports" / "phase4_rerun"
_VERIFY_CHECKPOINT = (
    _REPO_ROOT
    / "transformer_lens"
    / "tools"
    / "model_registry"
    / "data"
    / "verification_checkpoint.json"
)
_EXIT_GRACEFUL_INTERRUPT = 42

# Outcomes that mean the measurement did not happen — these gate the exit code
# and feed the consecutive-failure abort.
_HARD_OUTCOMES = {"crash", "timeout", "no_score"}
# Outcomes that produced a score worth flagging to a human.
_SOFT_OUTCOMES = {"below_pass", "regressed"}

# P4 cannot describe these models at all, so an unscored run is the right result
# rather than a failure: counting them as failures inflates the defect tally and
# trips the consecutive-failure abort part-way through a healthy sweep.
_NOT_APPLICABLE_MARKERS = (
    "masked-LM is not representable causally",
    "is the pinned judge",
    "No phase produced a score",
)
# Distinct from the above: P4 would apply, but no prompts exist for this model's
# profile yet. Actionable — the benchmark itself asks for an issue to add coverage.
_MISSING_PROFILE_MARKER = "no prompts for profile"

_interrupted = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    """Ctrl+C stops between models; the child gets the same signal and exits 42."""
    global _interrupted  # noqa: PLW0603
    if _interrupted:
        print("\nForce quit.")
        raise SystemExit(1)
    _interrupted = True
    print("\n\nInterrupt received — stopping after the current model.\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- selection


@dataclass
class Target:
    model_id: str
    architecture_id: str
    old_score: Optional[float]
    old_version: Optional[int]
    params: Optional[int] = None

    @property
    def size_label(self) -> str:
        return "?" if self.params is None else f"{self.params / 1e9:.2f}B"


def size_index(model_ids: list[str], cache_path: Path) -> dict[str, Optional[int]]:
    """Parameter counts from the hub's safetensors metadata, cached on disk.

    One metadata call per model, no download. This is what makes a size-ordered
    sweep possible: small models first, so a batch is not five 6B models deep
    before it reports anything.
    """
    index: dict[str, Optional[int]] = {}
    if cache_path.exists():
        index = json.loads(cache_path.read_text())
    # Re-fetch nulls: they predate the weight-file fallback below.
    missing = [m for m in model_ids if index.get(m) is None]
    if not missing:
        return index

    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    api = HfApi()

    _WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt")

    def from_weight_files(model_id: str) -> Optional[int]:
        """Params implied by the weight files, for repos publishing no metadata.

        Two bytes per parameter: an fp32 repo over-estimates, which keeps a big
        model out of a small batch rather than letting it in. Getting this wrong
        in the other direction costs a multi-GB download and an hour of runtime.
        """
        try:
            info = api.model_info(model_id, files_metadata=True)
        except Exception:
            return None
        total = sum(
            sibling.size or 0
            for sibling in (info.siblings or [])
            if sibling.rfilename.endswith(_WEIGHT_SUFFIXES)
        )
        return int(total / 2) if total else None

    def fetch(model_id: str) -> tuple[str, Optional[int]]:
        try:
            info = api.model_info(model_id, expand=["safetensors"])
            if info.safetensors and info.safetensors.total:
                return model_id, info.safetensors.total
        except Exception:
            return model_id, None
        return model_id, from_weight_files(model_id)

    print(f"Fetching parameter counts for {len(missing)} models...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for model_id, total in pool.map(fetch, missing):
            index[model_id] = total
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(index))
    return index


def _p4_applicable(arch: str) -> bool:
    """Audio and vision architectures have no text tower, and a masked LM cannot be
    generated from causally — phase 4 never produces a score for any of them."""
    if classify_architecture(arch) == "masked_lm":
        return False
    return 4 in _full_and_core_phases(arch)[0]


def select_targets(args: argparse.Namespace) -> list[Target]:
    entries = load_supported_models_raw().get("models", [])
    wanted_status = {STATUS_VERIFIED}
    if args.include_provisional:
        wanted_status.add(STATUS_PROVISIONAL)
    explicit = set(args.model or [])
    archs = set(args.architectures or [])

    targets: list[Target] = []
    for entry in entries:
        model_id = entry["model_id"]
        arch = entry.get("architecture_id") or ""
        if explicit:
            if model_id not in explicit:
                continue
        else:
            if entry.get("status") not in wanted_status:
                continue
            if archs and arch not in archs:
                continue
            if not _p4_applicable(arch):
                continue
            if is_incompatible_quantized(model_id):
                continue
            if model_id.lower() == JUDGE_MODEL_ID.lower():
                # P4 refuses to let the pinned judge score itself, so this one
                # never yields a score however many times it is selected.
                continue
            if not args.all and entry.get("p4_scoring_version") == P4_SCORING_VERSION:
                continue
            score = entry.get("phase4_score")
            if args.below is not None and (score is None or score >= args.below):
                continue
            if args.only_scored and score is None:
                continue
        targets.append(
            Target(model_id, arch, entry.get("phase4_score"), entry.get("p4_scoring_version"))
        )

    missing = explicit - {t.model_id for t in targets}
    for model_id in sorted(missing):
        print(f"  {model_id} is not in supported_models.json — skipping")

    needs_size = args.order == "size" or args.max_params_b or args.min_params_b
    if needs_size:
        index = size_index([t.model_id for t in targets], args.state_dir / "model_sizes.json")
        for target in targets:
            target.params = index.get(target.model_id)
        if args.max_params_b or args.min_params_b:
            low = (args.min_params_b or 0) * 1e9
            high = (args.max_params_b or float("inf")) * 1e9

            def in_band(target: Target) -> bool:
                # An unknown size cannot be bounded. Older repos publish no
                # safetensors metadata, so dropping them silently strands a
                # large slice of the backlog -- --include-unknown-size takes
                # them anyway, with --max-memory as the real guard rail.
                if target.params is None:
                    return bool(args.include_unknown_size)
                return low <= target.params <= high

            targets = [t for t in targets if in_band(t)]

    if args.order == "size":
        targets.sort(key=lambda t: (t.params is None, t.params or 0, t.model_id))
    elif args.order == "score":
        # Worst-scoring first: the models whose stale score is most suspect.
        targets.sort(key=lambda t: (t.old_score is None, t.old_score))
    elif args.order == "arch":
        targets.sort(key=lambda t: (t.architecture_id, t.model_id))
    if args.limit:
        targets = targets[: args.limit]
    return targets


# ---------------------------------------------------------------- hf cache


class CacheManager:
    """Deletes the bytes this run pulled in, once their model has been scored.

    Eligibility is measured in bytes, not repo identity. Most registry models are
    already in the cache as a config-only stub from an earlier verification run,
    so "is this repo new?" answers no for exactly the repos whose weights this run
    just downloaded -- and the cache grows unbounded.

    What was on disk beforehand stays: this is the user's cache, and a sweep of a
    thousand models must not silently reclaim hundreds of GB that were already
    there. `--reclaim-preexisting` opts into that, oldest-accessed first, and only
    while the cache is over `--cache-limit-gb`.
    """

    def __init__(self, args: argparse.Namespace, warn) -> None:
        self.policy = args.cache_policy
        self.limit_bytes = args.cache_limit_gb * 1e9
        self.min_free_bytes = args.min_free_gb * 1e9
        self.reclaim_preexisting = args.reclaim_preexisting
        self.dry_run = args.cache_dry_run
        self.protected = {JUDGE_MODEL_ID, *(args.keep or [])}
        self.warn = warn
        self.freed_bytes = 0
        # repo_id -> bytes on disk when the run started; absent means brand new.
        self.baseline: dict[str, int] = {}
        if self.policy != "off":
            self.baseline = {r.repo_id: r.size_on_disk for r in self._repos()}

    def _grew_this_run(self, repo) -> bool:
        """True when this run added bytes to the repo (or created it)."""
        # 1 MB of slack: metadata refreshes are not a weight download.
        return repo.size_on_disk > self.baseline.get(repo.repo_id, 0) + 1_000_000

    @staticmethod
    def _repos():
        from huggingface_hub import scan_cache_dir

        try:
            return [r for r in scan_cache_dir().repos if r.repo_type == "model"]
        except Exception as exc:  # a corrupt cache must not stop verification
            print(f"  [cache] scan failed: {exc}")
            return []

    @property
    def cache_dir(self) -> Path:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)

    def free_bytes(self) -> float:
        try:
            return float(shutil.disk_usage(self.cache_dir).free)
        except OSError:
            return float("inf")

    def _delete(self, repos, reason: str) -> None:
        from huggingface_hub import scan_cache_dir

        hashes = [rev.commit_hash for repo in repos for rev in repo.revisions]
        if not hashes:
            return
        size = sum(r.size_on_disk for r in repos)
        names = ", ".join(sorted(r.repo_id for r in repos)[:5])
        more = f" (+{len(repos) - 5} more)" if len(repos) > 5 else ""
        prefix = "[cache] would free" if self.dry_run else "[cache] freed"
        print(f"  {prefix} {size / 1e9:.1f} GB — {reason}: {names}{more}")
        if self.dry_run:
            return
        try:
            scan_cache_dir().delete_revisions(*hashes).execute()
            self.freed_bytes += size
        except Exception as exc:
            self.warn("cache", "-", f"failed to delete {names}: {exc}")

    def prune_xet(self) -> None:
        """Drop the Xet chunk cache, which scan_cache_dir neither sees nor deletes.

        hub 1.x stages downloads through content-addressed chunks that survive
        deleting the revision they belong to. Regenerable, so only worth the
        re-download when the disk is actually tight.
        """
        try:
            from huggingface_hub.constants import HF_XET_CACHE

            xet = Path(HF_XET_CACHE)
        except ImportError:  # pragma: no cover - hub without Xet support
            return
        if not xet.exists():
            return
        size = sum(f.stat().st_size for f in xet.rglob("*") if f.is_file())
        if size < 1e9:
            return
        print(
            f"  [cache] {'would free' if self.dry_run else 'freed'} {size / 1e9:.1f} GB — Xet chunks"
        )
        if self.dry_run:
            return
        shutil.rmtree(xet, ignore_errors=True)
        self.freed_bytes += size

    def collect(self, just_finished: Optional[str] = None) -> None:
        """Run after each model, with the id of the model that just finished."""
        if self.policy == "off":
            return
        repos = self._repos()
        total = sum(r.size_on_disk for r in repos)
        over_limit = total > self.limit_bytes
        low_disk = self.free_bytes() < self.min_free_bytes

        # Bytes this run pulled in are ours to drop — including the model that just
        # finished, which is why GC runs after the subprocess exits, never during.
        downloaded = [
            r for r in repos if r.repo_id not in self.protected and self._grew_this_run(r)
        ]
        if self.policy == "purge-each" or over_limit or low_disk:
            if downloaded:
                self._delete(downloaded, "downloaded by this run")
                # Deleted repos must not look "grown" again on the next pass.
                for repo in downloaded:
                    self.baseline.pop(repo.repo_id, None)

        if low_disk:
            self.prune_xet()
            low_disk = self.free_bytes() < self.min_free_bytes

        if not (self.reclaim_preexisting and (over_limit or low_disk)):
            if low_disk:
                self.warn(
                    "cache",
                    just_finished or "-",
                    f"only {self.free_bytes() / 1e9:.1f} GB free on {self.cache_dir}; "
                    f"pass --reclaim-preexisting to let the pre-existing cache be trimmed",
                )
            return

        # Opt-in reclaim: oldest-accessed pre-existing repos until back under the limit.
        remaining = [
            r
            for r in self._repos()
            if r.repo_id not in self.protected and r.repo_id != just_finished
        ]
        remaining.sort(key=lambda r: r.last_accessed)
        total = sum(r.size_on_disk for r in self._repos())
        victims = []
        for repo in remaining:
            if total <= self.limit_bytes and self.free_bytes() >= self.min_free_bytes:
                break
            victims.append(repo)
            total -= repo.size_on_disk
        if victims:
            self._delete(victims, "reclaiming pre-existing cache")


# ---------------------------------------------------------------- the run


@dataclass
class Runner:
    args: argparse.Namespace
    state_dir: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    consecutive_hard: int = 0

    # -- warnings ---------------------------------------------------------

    def warn(self, category: str, model_id: str, detail: str) -> None:
        entry = {"time": _now(), "category": category, "model_id": model_id, "detail": detail}
        self.warnings.append(entry)
        line = f"WARNING [{category}] {model_id}: {detail}"
        print(f"\n  {'!' * 70}\n  {line}\n  {'!' * 70}\n", flush=True)
        with (self.state_dir / "warnings.log").open("a") as fh:
            fh.write(f"{entry['time']}  {line}\n")

    # -- state ------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.json"

    def load_state(self) -> None:
        if self.args.restart or not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        self.records = data.get("records", {})
        self.warnings = data.get("warnings", [])
        failed = [m for m, r in self.records.items() if r["outcome"] in _HARD_OUTCOMES]
        verb = "retried" if self.args.retry_failed else "skipped (--retry-failed re-runs them)"
        print(
            f"Resuming: {len(self.records) - len(failed)} models already re-scored, "
            f"{len(failed)} hard failures will be {verb}"
        )

    def save_state(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {"updated": _now(), "records": self.records, "warnings": self.warnings}, indent=1
            )
        )

    # -- one model --------------------------------------------------------

    def command(self, model_id: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "transformer_lens.tools.model_registry.verify_models",
            "--model",
            model_id,
            "--phases",
            "4",
            "--device",
            self.args.device,
            "--dtype",
            self.args.dtype,
        ]
        if self.args.max_memory is not None:
            cmd += ["--max-memory", str(self.args.max_memory)]
        # Never --quiet the child: the log is the post-mortem, and the skip reason
        # only appears in its verbose output. --quiet here only stops the mirroring.
        return cmd

    def run_one(self, target: Target, index: int, total: int) -> dict[str, Any]:
        model_id = target.model_id
        print(f"\n{'=' * 70}\n[{index}/{total}] {model_id} ({target.architecture_id})")
        old = "unscored" if target.old_score is None else f"{target.old_score}"
        print(
            f"  {target.size_label} params, stored P4: {old} (v{target.old_version or 1})\n"
            f"{'=' * 70}",
            flush=True,
        )

        log_path = self.state_dir / "logs" / f"{model_id.replace('/', '__')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        tail: list[str] = []
        with log_path.open("w") as log:
            env = dict(os.environ)
            # Same pinning as the parent: the child must import this tree.
            env["PYTHONPATH"] = os.pathsep.join(
                [str(_REPO_ROOT), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
            )
            proc = subprocess.Popen(
                self.command(model_id),
                cwd=_REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            last_output = [time.time()]
            watching = [True]

            def _watch_for_stall() -> None:
                """Report a quiet model; never kill it.

                A big model legitimately runs for an hour -- Nemotron-3-Nano-4B
                needed 67 minutes and was killed seconds after it had produced a
                valid score. Silence is the signal worth surfacing, and the human
                (or agent) watching decides whether it is a hang.
                """
                warned_at = 0.0
                while watching[0]:
                    time.sleep(min(30.0, self.args.stall_after))
                    if not watching[0]:
                        return
                    quiet = time.time() - last_output[0]
                    if (
                        quiet >= self.args.stall_after
                        and quiet - warned_at >= self.args.stall_after
                    ):
                        warned_at = quiet
                        self.warn(
                            "stall",
                            model_id,
                            f"no output for {quiet / 60:.0f} min "
                            f"({(time.time() - start) / 60:.0f} min total) — check whether it hung",
                        )

            stall_watch = threading.Thread(target=_watch_for_stall, daemon=True)
            if self.args.stall_after:
                stall_watch.start()
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    last_output[0] = time.time()
                    log.write(line)
                    tail.append(line.rstrip())
                    del tail[:-40]
                    if not self.args.quiet:
                        print(f"  | {line.rstrip()}", flush=True)
                returncode = proc.wait()
            finally:
                watching[0] = False
        elapsed = time.time() - start

        record = self.classify(target, returncode, tail, elapsed)
        record["log"] = os.path.relpath(log_path, _REPO_ROOT)
        return record

    def classify(
        self,
        target: Target,
        returncode: int,
        tail: list[str],
        elapsed: float,
    ) -> dict[str, Any]:
        model_id = target.model_id
        new_score, new_version = _current_p4(model_id)
        record: dict[str, Any] = {
            "model_id": model_id,
            "architecture_id": target.architecture_id,
            "old_score": target.old_score,
            "old_version": target.old_version,
            "new_score": new_score,
            "new_version": new_version,
            "returncode": returncode,
            "elapsed_s": round(elapsed, 1),
            "time": _now(),
        }

        scored = new_score is not None and new_version == P4_SCORING_VERSION
        # Score first, exit code second. A model killed moments after writing a
        # valid score has been measured; recording it as a failure would skip a
        # model that is already done (Nemotron-3-Nano-4B, killed at 67 min with
        # 84.1 already in the registry).
        if not scored:
            if returncode == _EXIT_GRACEFUL_INTERRUPT:
                record["outcome"] = "interrupted"
                return record
            if returncode != 0:
                record["outcome"] = "crash"
                self.warn("crash", model_id, f"exit {returncode}: {_reason(tail)}")
                return record
        elif returncode != 0:
            self.warn(
                "scored_but_exited",
                model_id,
                f"exit {returncode} after writing P4={new_score} — score kept",
            )
        if not scored:
            joined = " ".join(tail)
            # Order matters: a missing profile also reports "No phase produced a
            # score", and it is the more specific — and actionable — cause.
            if _MISSING_PROFILE_MARKER in joined:
                record["outcome"] = "missing_profile"
                self.warn("missing_profile", model_id, _reason(tail))
                return record
            if any(marker in joined for marker in _NOT_APPLICABLE_MARKERS):
                record["outcome"] = "not_applicable"
                record["detail"] = _reason(tail)
                return record
            # Exit 0 with no fresh score: the verifier skipped the model (memory cap,
            # gated repo, missing quant library) or the bridge failed to load.
            record["outcome"] = "no_score"
            self.warn(
                "no_score", model_id, f"no v{P4_SCORING_VERSION} score written: {_reason(tail)}"
            )
            return record

        assert new_score is not None  # `scored` is exactly this check
        threshold = p4_pass_threshold()
        record["outcome"] = "ok"
        if new_score < threshold:
            record["outcome"] = "below_pass"
            self.warn(
                "below_pass",
                model_id,
                f"P4={new_score} is under the v{P4_SCORING_VERSION} pass line {threshold}",
            )
        elif (
            target.old_version == P4_SCORING_VERSION
            and target.old_score is not None
            and new_score < target.old_score - self.args.regression_delta
        ):
            record["outcome"] = "regressed"
            self.warn(
                "regressed",
                model_id,
                f"P4 dropped {target.old_score} -> {new_score} on the same scale",
            )
        else:
            print(
                f"  P4={new_score} (was {target.old_score}, v{target.old_version or 1}) "
                f"in {elapsed:.0f}s"
            )
        return record

    # -- loop -------------------------------------------------------------

    def run(self, targets: list[Target]) -> int:
        cache = CacheManager(self.args, self.warn)
        pending = [t for t in targets if not self._already_done(t.model_id)]
        print(
            f"\n{len(pending)} models to re-score "
            f"({len(targets) - len(pending)} already done in this state file)"
        )
        if cache.policy != "off":
            print(
                f"Cache GC: {cache.policy}, limit {self.args.cache_limit_gb} GB, "
                f"{cache.free_bytes() / 1e9:.0f} GB free, protecting {sorted(cache.protected)}"
            )

        total = len(pending)
        started = time.time()
        aborted = False
        for i, target in enumerate(pending, 1):
            if _interrupted:
                print("\nStopped by interrupt — re-run with the same command to resume.")
                break
            record = self.run_one(target, i, total)
            self.records[target.model_id] = record
            self.save_state()

            if record["outcome"] == "interrupted":
                print("\nChild stopped gracefully — re-run to resume.")
                break
            if record["outcome"] in _HARD_OUTCOMES:
                self.consecutive_hard += 1
                if self.args.abort_after and self.consecutive_hard >= self.args.abort_after:
                    self.warn(
                        "abort",
                        target.model_id,
                        f"{self.consecutive_hard} hard failures in a row — stopping. "
                        f"Check {self.state_dir / 'warnings.log'}",
                    )
                    aborted = True
                    break
            else:
                self.consecutive_hard = 0

            cache.collect(just_finished=target.model_id)

            done = i
            rate = (time.time() - started) / done
            print(f"  [{done}/{total} done, ~{rate * (total - done) / 3600:.1f} h left]")

        cache.collect()
        self.save_state()
        return self.summarize(cache, aborted)

    def _already_done(self, model_id: str) -> bool:
        """Hard failures stay done unless --retry-failed: a model that skips on every
        attempt would otherwise eat a slot in every batch and stall the sweep."""
        record = self.records.get(model_id)
        if record is None or record["outcome"] == "interrupted":
            return False
        if record["outcome"] in _HARD_OUTCOMES:
            return not self.args.retry_failed
        return True

    def summarize(self, cache: CacheManager, aborted: bool) -> int:
        counts: dict[str, int] = {}
        for record in self.records.values():
            counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
        print(f"\n{'=' * 70}\nPhase-4 re-run summary\n{'=' * 70}")
        for outcome in sorted(counts):
            print(f"  {outcome:<12} {counts[outcome]}")
        if cache.freed_bytes:
            print(f"  cache freed  {cache.freed_bytes / 1e9:.1f} GB")

        if self.warnings:
            print(f"\n{len(self.warnings)} warnings:")
            for entry in self.warnings[-30:]:
                print(f"  [{entry['category']}] {entry['model_id']}: {entry['detail']}")
            if len(self.warnings) > 30:
                print(f"  ... {len(self.warnings) - 30} earlier — see warnings.log")

        report = self.state_dir / "report.json"
        report.write_text(
            json.dumps(
                {
                    "generated": _now(),
                    "scoring_version": P4_SCORING_VERSION,
                    "pass_threshold": p4_pass_threshold(),
                    "counts": counts,
                    "records": self.records,
                    "warnings": self.warnings,
                },
                indent=1,
            )
        )
        print(f"\nReport: {report}\nState:  {self.state_path}")

        hard = sum(counts.get(o, 0) for o in _HARD_OUTCOMES)
        if aborted:
            return 2
        return 1 if hard else 0


def _current_p4(model_id: str) -> tuple[Optional[float], Optional[int]]:
    """Read back what the subprocess wrote — the registry is the source of truth."""
    for entry in load_supported_models_raw().get("models", []):
        if entry["model_id"] == model_id:
            return entry.get("phase4_score"), entry.get("p4_scoring_version")
    return None, None


_REASON_KEYS = (
    "out of memory",
    "OutOfMemoryError",
    "SKIP",
    "Skipping",
    "skipping",
    "Killed",
    "trust_remote_code",
    "gated",
    "401",
    "403",
    "Benchmark failed",
    "No results for requested phases",
    "Error",
    "Traceback",
)
# Run boilerplate that is never the reason for anything.
_REASON_NOISE = (
    "Checkpoint cleared",
    "Total time",
    "Verification Summary",
    "Total tested",
    "Verified:",
    "Provisional:",
    "Skipped:",
    "Failed:",
    "Skipped models",
    "=====",
    "-----",
)


def _reason(tail: list[str]) -> str:
    """Best-effort one-liner for why a model produced nothing, from its log tail."""
    for line in reversed(tail):
        if any(key in line for key in _REASON_KEYS):
            return line.strip()[:200]
    for line in reversed(tail):
        stripped = line.strip()
        if stripped and not any(noise in stripped for noise in _REASON_NOISE):
            return stripped[:200]
    return "no output — see the log"


def _load_env_file(path: Optional[Path]) -> None:
    """Source HF_TOKEN the way the repo's other hub-hitting entry points do."""
    if path is None or not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    print(f"Loaded env from {path}")


def _default_env_file() -> Optional[Path]:
    """This tree's .env, else the main checkout's — a worktree never has one."""
    local = _REPO_ROOT / ".env"
    if local.exists():
        return local
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None
    candidate = Path(common).parent / ".env"
    return candidate if candidate.exists() else None


def _backup_verify_checkpoint(state_dir: Path) -> None:
    """`--model` runs clear the shared checkpoint; keep a copy of anyone else's."""
    if _VERIFY_CHECKPOINT.exists():
        backup = state_dir / "verification_checkpoint.backup.json"
        shutil.copy2(_VERIFY_CHECKPOINT, backup)
        print(
            f"Note: a verify_models checkpoint exists and per-model runs clear it.\n"
            f"      Backed up to {backup}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sel = parser.add_argument_group("selection")
    sel.add_argument("--model", nargs="+", help="Explicit model ids (ignores every other filter)")
    sel.add_argument("--architectures", nargs="+", help="Restrict to these architecture ids")
    sel.add_argument("--limit", type=int, help="Cap the number of models")
    sel.add_argument("--below", type=float, help="Only models whose stored P4 is below this")
    sel.add_argument(
        "--all",
        action="store_true",
        help=f"Include models already scored on v{P4_SCORING_VERSION} (default: stale only)",
    )
    sel.add_argument("--only-scored", action="store_true", help="Skip models with no stored P4")
    sel.add_argument(
        "--include-provisional", action="store_true", help="Also re-score status=4 models"
    )
    sel.add_argument(
        "--order",
        choices=["size", "registry", "score", "arch"],
        default="size",
        help="ascending parameter count (default), registry order, worst stored score "
        "first, or grouped by architecture",
    )
    sel.add_argument(
        "--max-params-b", type=float, help="Only models with at most this many billion params"
    )
    sel.add_argument(
        "--min-params-b", type=float, help="Only models with at least this many billion params"
    )
    sel.add_argument(
        "--include-unknown-size",
        action="store_true",
        help="Keep models the hub publishes no size for (older repos) inside a param band",
    )

    run = parser.add_argument_group("run")
    run.add_argument("--device", default="cpu", help="Device passed to verify_models")
    run.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Dtype passed to verify_models (fp32 is the verification default)",
    )
    run.add_argument("--max-memory", type=float, help="GB cap passed to verify_models")
    run.add_argument(
        "--stall-after",
        type=float,
        default=1800,
        help="Warn when a model has produced no output for this many seconds (0 disables). "
        "Nothing is ever killed: a download of unknown size is not a hang, so the warning "
        "goes to whoever is watching and they decide whether to cancel.",
    )
    run.add_argument("--dry-run", action="store_true", help="List the plan and exit")
    run.add_argument("--restart", action="store_true", help="Ignore existing state and start over")
    run.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run models this state file recorded as crashed/timed-out/unscored",
    )
    run.add_argument("--quiet", action="store_true", help="Don't mirror child output to stdout")
    run.add_argument(
        "--state-dir", type=Path, default=_DEFAULT_STATE_DIR, help="Where state/logs/report go"
    )
    run.add_argument(
        "--env-file",
        type=Path,
        help="Env file to source (default: this tree's .env, else the main checkout's)",
    )

    warn = parser.add_argument_group("failure warnings")
    warn.add_argument(
        "--abort-after",
        type=int,
        default=5,
        help="Stop after this many consecutive hard failures (0 disables)",
    )
    warn.add_argument(
        "--regression-delta",
        type=float,
        default=5.0,
        help="Warn when a same-scale score drops by more than this many points",
    )

    cache = parser.add_argument_group("hf cache")
    cache.add_argument(
        "--cache-policy",
        choices=["purge-each", "watermark", "off"],
        default="purge-each",
        help="purge-each: drop each downloaded repo once scored (default); "
        "watermark: only when over --cache-limit-gb or under --min-free-gb; off: never",
    )
    cache.add_argument("--cache-limit-gb", type=float, default=200.0, help="Cache high-water mark")
    cache.add_argument("--min-free-gb", type=float, default=50.0, help="Free-disk floor")
    cache.add_argument(
        "--reclaim-preexisting",
        action="store_true",
        help="Let GC also delete repos that predate this run (oldest-accessed first)",
    )
    cache.add_argument("--keep", nargs="+", help="Repo ids GC must never delete")
    cache.add_argument(
        "--cache-dry-run", action="store_true", help="Report deletions, don't do them"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stall_after == 0:
        args.stall_after = None
    _load_env_file(args.env_file or _default_env_file())

    targets = select_targets(args)
    if not targets:
        print("Nothing to re-score.")
        return 0

    threshold = p4_pass_threshold()
    stale = sum(1 for t in targets if t.old_version != P4_SCORING_VERSION)
    print(
        f"{len(targets)} models selected ({stale} on the old scale); "
        f"v{P4_SCORING_VERSION} pass line is {threshold}"
    )

    if args.dry_run:
        print(f"\nWould run: {' '.join(Runner(args, args.state_dir).command('<model>'))}\n")
        for target in targets:
            old = "unscored" if target.old_score is None else f"P4={target.old_score}"
            print(
                f"  {target.model_id:<58} {target.architecture_id:<30} "
                f"{target.size_label:>7}  {old} (v{target.old_version or 1})"
            )
        return 0

    args.state_dir.mkdir(parents=True, exist_ok=True)
    _backup_verify_checkpoint(args.state_dir)
    signal.signal(signal.SIGINT, _handle_sigint)

    runner = Runner(args, args.state_dir)
    runner.load_state()
    return runner.run(targets)


if __name__ == "__main__":
    raise SystemExit(main())
