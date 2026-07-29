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
