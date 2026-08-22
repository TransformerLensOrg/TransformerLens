"""Regression tests for main_benchmark.update_model_registry.

This path was a drifted mirror of verify_models' registry-writing logic: its
phase dict stopped at phase 3 and it wrote STATUS_VERIFIED unconditionally,
bypassing the provisional gate for --no-hf-reference runs. It now shares
registry_io's extract_phase_scores / pass_status, so these tests pin the
registry outcomes, not the internals.
"""
import json
from types import SimpleNamespace

import pytest

from transformer_lens.benchmarks.main_benchmark import update_model_registry
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.tools.model_registry.registry_io import (
    STATUS_FAILED,
    STATUS_PROVISIONAL,
    STATUS_VERIFIED,
)

ARCH = "GPT2LMHeadModel"


def _result(phase, passed, name="forward_pass", severity=None, details=None):
    if severity is None:
        severity = BenchmarkSeverity.INFO if passed else BenchmarkSeverity.DANGER
    return BenchmarkResult(
        name=name,
        severity=severity,
        message="ok" if passed else "mismatch",
        details=details,
        passed=passed,
        phase=phase,
    )


@pytest.fixture
def registry_paths(tmp_path, monkeypatch):
    """Point registry_io at temp files and stub the AutoConfig network call."""
    from transformer_lens.tools.model_registry import registry_io

    supported = {
        "total_architectures": 1,
        "total_models": 1,
        "total_verified": 0,
        "models": [
            {
                "architecture_id": ARCH,
                "model_id": "seeded/model",
                "status": 0,
                "verified_date": None,
                "metadata": None,
                "note": None,
            },
        ],
    }
    supported_path = tmp_path / "supported_models.json"
    supported_path.write_text(json.dumps(supported, indent=2))
    history_path = tmp_path / "verification_history.json"

    monkeypatch.setattr(registry_io, "_SUPPORTED_MODELS_PATH", supported_path)
    monkeypatch.setattr(registry_io, "_VERIFICATION_HISTORY_PATH", history_path)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(architectures=[ARCH]),
    )
    return supported_path, history_path


def _entry(supported_path, model_id):
    data = json.loads(supported_path.read_text())
    return next(m for m in data["models"] if m["model_id"] == model_id), data


class TestProvisionalGate:
    def test_no_hf_reference_writes_provisional(self, registry_paths):
        supported_path, history_path = registry_paths
        results = [_result(1, True)]

        assert update_model_registry("new/model", results, use_hf_reference=False)

        entry, data = _entry(supported_path, "new/model")
        assert entry["status"] == STATUS_PROVISIONAL
        assert entry["note"].startswith("Structural only (no HF reference)")
        assert data["total_verified"] == 0
        assert data["total_provisional"] == 1
        # No history record: VerificationHistory.is_verified() treats any
        # record as verified — the second "counts as verified" path.
        assert not history_path.exists()

    def test_hf_reference_writes_verified(self, registry_paths):
        supported_path, history_path = registry_paths
        results = [_result(1, True)]

        assert update_model_registry("new/model", results, use_hf_reference=True)

        entry, data = _entry(supported_path, "new/model")
        assert entry["status"] == STATUS_VERIFIED
        assert data["total_verified"] == 1
        history = json.loads(history_path.read_text())
        assert history["records"][-1]["model_id"] == "new/model"
        assert history["records"][-1]["verified_by"] == "main_benchmark"

    def test_default_is_conservative_provisional(self, registry_paths):
        supported_path, _ = registry_paths
        update_model_registry("new/model", [_result(1, True)])
        entry, _ = _entry(supported_path, "new/model")
        assert entry["status"] == STATUS_PROVISIONAL


class TestPhaseCoverage:
    def test_phase9_contributes_phase9_score(self, registry_paths):
        # The drifted mirror's {1: [], 2: [], 3: []} dict silently dropped P9.
        supported_path, _ = registry_paths
        results = [_result(1, True), _result(9, True, name="vision_forward")]

        update_model_registry("new/model", results, use_hf_reference=True)

        entry, _ = _entry(supported_path, "new/model")
        assert entry["phase9_score"] == 100.0
        assert entry["status"] == STATUS_VERIFIED

    def test_unrun_phases_preserve_existing_scores(self, registry_paths):
        # The old path wrote None for unrun phases, clobbering prior scores.
        supported_path, _ = registry_paths
        update_model_registry("seeded/model", [_result(2, True)], use_hf_reference=True)
        entry, _ = _entry(supported_path, "seeded/model")
        assert entry["phase2_score"] == 100.0

        update_model_registry("seeded/model", [_result(1, True)], use_hf_reference=True)
        entry, _ = _entry(supported_path, "seeded/model")
        assert entry["phase1_score"] == 100.0
        assert entry["phase2_score"] == 100.0


class TestThresholdGate:
    def test_failing_scores_write_failed_not_verified(self, registry_paths):
        # The drifted mirror wrote VERIFIED even for all-fail runs.
        supported_path, _ = registry_paths
        results = [_result(1, False, name="logits_equivalence")]

        update_model_registry("seeded/model", results, use_hf_reference=True)

        entry, data = _entry(supported_path, "seeded/model")
        assert entry["status"] == STATUS_FAILED
        assert "Below threshold" in entry["note"]
        assert data["total_verified"] == 0


class TestPromptProfileWriteback:
    """The Phase-4 profile actually used must land in the registry sparsely:
    non-default profiles are recorded, the default writes no key at all (the
    registry JSON is served to the docs site; 15k default keys are dead weight)."""

    def _p4(self, profile):
        return _result(
            4,
            True,
            name="text_quality",
            details={"score": 91.0, "prompt_profile": profile},
        )

    def test_prompt_profile_written_from_p4_details(self, registry_paths):
        supported_path, _ = registry_paths
        update_model_registry(
            "seeded/model",
            [_result(1, True), self._p4("task:translation@en-de")],
            use_hf_reference=True,
        )
        entry, _ = _entry(supported_path, "seeded/model")
        assert entry["prompt_profile"] == "task:translation@en-de"
        # Key order: sparse key sits right after note, before phase scores.
        keys = list(entry)
        assert keys.index("prompt_profile") == keys.index("note") + 1

    def test_default_profile_not_written(self, registry_paths):
        supported_path, _ = registry_paths
        update_model_registry(
            "seeded/model",
            [_result(1, True), self._p4("continuation")],
            use_hf_reference=True,
        )
        entry, _ = _entry(supported_path, "seeded/model")
        assert "prompt_profile" not in entry

    def test_existing_profile_survives_profileless_rerun(self, registry_paths):
        """A later run without a P4 result must not clobber the stored profile."""
        supported_path, _ = registry_paths
        update_model_registry(
            "seeded/model",
            [_result(1, True), self._p4("chat@fr")],
            use_hf_reference=True,
        )
        update_model_registry("seeded/model", [_result(1, True)], use_hf_reference=True)
        entry, _ = _entry(supported_path, "seeded/model")
        assert entry["prompt_profile"] == "chat@fr"

    def test_default_profile_clears_stale_nondefault(self, registry_paths):
        """A model re-resolved to the default must lose its old sparse key —
        otherwise a stale 'chat@fr' misdescribes how the score was produced."""
        supported_path, _ = registry_paths
        update_model_registry(
            "seeded/model",
            [_result(1, True), self._p4("chat@fr")],
            use_hf_reference=True,
        )
        update_model_registry(
            "seeded/model",
            [_result(1, True), self._p4("continuation")],
            use_hf_reference=True,
        )
        entry, _ = _entry(supported_path, "seeded/model")
        assert "prompt_profile" not in entry

    def test_new_entry_append_carries_profile(self, registry_paths):
        """The append (model-not-in-registry) branch must also write the sparse
        key, positioned after note."""
        supported_path, _ = registry_paths
        update_model_registry(
            "unseeded/model",
            [_result(1, True), self._p4("task:summarization")],
            use_hf_reference=True,
        )
        entry, _ = _entry(supported_path, "unseeded/model")
        assert entry["prompt_profile"] == "task:summarization"
        keys = list(entry)
        assert keys.index("prompt_profile") == keys.index("note") + 1


class TestP4ScoringVersionStamp:
    """phase4_score is a mixed-scale column (old GPT-2 scale vs pinned-judge
    ratio scale); every P4-bearing write must stamp the scale it measured on,
    and writes without a P4 result must not touch an existing stamp."""

    def test_p4_write_stamps_current_version(self, registry_paths):
        from transformer_lens.benchmarks.text_quality_profiles import P4_SCORING_VERSION
        from transformer_lens.tools.model_registry import registry_io

        supported_path, _ = registry_paths
        registry_io.update_model_status(
            "seeded/model",
            "GPT2LMHeadModel",
            registry_io.STATUS_VERIFIED,
            phase_scores={1: 100.0, 4: 91.0},
        )
        entry, _ = _entry(supported_path, "seeded/model")
        assert entry["p4_scoring_version"] == P4_SCORING_VERSION

    def test_no_p4_write_preserves_existing_stamp(self, registry_paths):
        from transformer_lens.tools.model_registry import registry_io

        supported_path, _ = registry_paths
        registry_io.update_model_status(
            "seeded/model",
            "GPT2LMHeadModel",
            registry_io.STATUS_VERIFIED,
            phase_scores={1: 100.0, 4: 91.0},
        )
        registry_io.update_model_status(
            "seeded/model",
            "GPT2LMHeadModel",
            registry_io.STATUS_VERIFIED,
            phase_scores={1: 100.0},
        )
        entry, _ = _entry(supported_path, "seeded/model")
        assert entry["p4_scoring_version"] == 2
        assert entry["phase4_score"] == 91.0

    def test_old_scale_entry_has_no_stamp(self, registry_paths):
        supported_path, _ = registry_paths
        entry, _ = _entry(supported_path, "seeded/model")
        assert "p4_scoring_version" not in entry

    def test_new_entry_with_p4_is_stamped(self, registry_paths):
        from transformer_lens.tools.model_registry import registry_io

        supported_path, _ = registry_paths
        registry_io.update_model_status(
            "brand/new-model",
            "GPT2LMHeadModel",
            registry_io.STATUS_VERIFIED,
            phase_scores={1: 100.0, 4: 77.0},
        )
        entry, _ = _entry(supported_path, "brand/new-model")
        assert entry["p4_scoring_version"] == 2


class TestPreservedIssueSuffix:
    """A phases-1-4 pass must not overwrite tracked residue from phases it
    did not re-run (gemma-2-2b-it's P3=95.5 unembed_centering note was
    clobbered by a bare 'Core verification completed')."""

    def test_sub100_score_from_unrun_phase_is_retained(self, registry_paths):
        from transformer_lens.tools.model_registry import registry_io
        from transformer_lens.tools.model_registry.verify_models import (
            _preserved_issue_suffix,
        )

        registry_io.update_model_status(
            "seeded/model",
            "GPT2LMHeadModel",
            registry_io.STATUS_VERIFIED,
            phase_scores={1: 100.0, 3: 95.5},
        )
        assert _preserved_issue_suffix("seeded/model", [1, 4]) == (
            " (prior issues retained: P3=95.5%)"
        )
        # Re-running the phase drops it from the suffix (the fresh score speaks).
        assert _preserved_issue_suffix("seeded/model", [1, 3, 4]) == ""

    def test_clean_entry_has_no_suffix(self, registry_paths):
        from transformer_lens.tools.model_registry.verify_models import (
            _preserved_issue_suffix,
        )

        assert _preserved_issue_suffix("seeded/model", [1, 4]) == ""


def test_judge_overhead_not_charged_to_accelerator():
    """The judge is CPU-pinned; charging its 2.5 GB to a cuda budget caused
    spurious VRAM skips."""
    from transformer_lens.tools.model_registry.verify_models import (
        estimate_benchmark_memory_gb,
    )

    # Small model so the phase-4 peak (model + judge) is the max across phases.
    cpu = estimate_benchmark_memory_gb(int(1e6), phases=[1, 4], device="cpu")
    cuda = estimate_benchmark_memory_gb(int(1e6), phases=[1, 4], device="cuda")
    assert cpu > 2.5
    assert cuda < 0.1
