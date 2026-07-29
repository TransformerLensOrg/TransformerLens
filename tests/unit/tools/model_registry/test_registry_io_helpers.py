"""Unit tests for the shared registry-writing helpers in registry_io.

``extract_phase_scores`` and ``pass_status`` are the single home for phase-score
extraction and pass-status logic, used by both verify_models and
main_benchmark.update_model_registry (moved here from verify_models).
"""
from types import SimpleNamespace

from transformer_lens.benchmarks.utils import BenchmarkSeverity
from transformer_lens.tools.model_registry.registry_io import (
    STATUS_PROVISIONAL,
    STATUS_VERIFIED,
    extract_phase_scores,
    pass_status,
)


def _result(phase, passed, severity, details=None, name="test"):
    return SimpleNamespace(
        phase=phase, passed=passed, severity=severity, details=details, name=name
    )


class TestExtractPhaseScores:
    def test_skipped_results_excluded_from_score(self):
        # A SKIPPED centering test (passed=False) must NOT drag the phase down — this is
        # exactly the SSM P3=90 bug. One real PASS + one SKIPPED → 100.
        results = [
            _result(3, True, BenchmarkSeverity.INFO),
            _result(3, False, BenchmarkSeverity.SKIPPED),
        ]
        assert extract_phase_scores(results)[3] == 100.0

    def test_mixed_pass_fail_averaged(self):
        results = [
            _result(1, True, BenchmarkSeverity.INFO),
            _result(1, False, BenchmarkSeverity.DANGER),
        ]
        assert extract_phase_scores(results)[1] == 50.0

    def test_all_skipped_phase_omitted(self):
        results = [_result(2, False, BenchmarkSeverity.SKIPPED)]
        assert 2 not in extract_phase_scores(results)

    def test_phase4_uses_quality_score_from_details(self):
        results = [_result(4, True, BenchmarkSeverity.INFO, details={"score": 87.5})]
        assert extract_phase_scores(results)[4] == 87.5

    def test_phase9_results_scored(self):
        results = [
            _result(9, True, BenchmarkSeverity.INFO),
            _result(9, False, BenchmarkSeverity.DANGER),
        ]
        assert extract_phase_scores(results)[9] == 50.0


class TestPassStatus:
    """A passing run is VERIFIED only when numerically compared to HF; a
    --no-hf-reference (structural-only) pass is PROVISIONAL, not verified."""

    def test_hf_reference_passes_verify(self):
        assert pass_status(use_hf_reference=True) == STATUS_VERIFIED

    def test_no_hf_reference_is_provisional(self):
        assert pass_status(use_hf_reference=False) == STATUS_PROVISIONAL
