"""Unit tests for the shared registry-writing helpers in registry_io.

``extract_phase_scores`` and ``pass_status`` are the single home for phase-score
extraction and pass-status logic, used by both verify_models and
main_benchmark.update_model_registry (moved here from verify_models).
"""
from types import SimpleNamespace

from transformer_lens.benchmarks.utils import BenchmarkSeverity
from transformer_lens.tools.model_registry.registry_io import (
    MODALITY_PHASES,
    PHASES,
    STATUS_PROVISIONAL,
    STATUS_SKIPPED,
    STATUS_VERIFIED,
    TEXT_PHASES,
    extract_phase_scores,
    load_supported_models_raw,
    pass_status,
    recompute_registry_totals,
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


class TestPhaseGroups:
    """TEXT_PHASES gate applicable_phases; MODALITY_PHASES bypass that gate
    (verify_models._phases_to_run / main_benchmark._phase_enabled)."""

    def test_phases_is_text_plus_modality(self):
        assert PHASES == TEXT_PHASES + MODALITY_PHASES
        assert set(TEXT_PHASES).isdisjoint(MODALITY_PHASES)

    def test_registry_score_columns_are_phase_members(self):
        # Columns appear lazily (a phase's column lands on first write), so the
        # data-file schema check is subset of PHASES, not equality.
        allowed = {f"phase{p}_score" for p in PHASES}
        data = load_supported_models_raw()
        seen = {k for m in data["models"] for k in m if k.startswith("phase")}
        assert seen  # the registry does carry phase columns
        assert seen <= allowed


class TestRecomputeRegistryTotals:
    """Both supported_models.json writers (update_model_status and hf_scraper)
    share this recomputation — pin key names and counting rules."""

    def test_counting_rules_and_key_names(self):
        models = [
            {"architecture_id": "ArchA", "model_id": "a1", "status": STATUS_VERIFIED},
            {"architecture_id": "ArchA", "model_id": "a2", "status": STATUS_PROVISIONAL},
            {"architecture_id": "ArchB", "model_id": "b1", "status": STATUS_SKIPPED},
            {"architecture_id": "ArchB", "model_id": "b2"},  # missing status -> unverified
        ]
        totals = recompute_registry_totals(models)
        assert totals == {
            "total_architectures": 2,
            "total_models": 4,
            "total_verified": 1,
            "total_provisional": 1,
        }
        # Key order feeds hf_scraper's JSON dump directly — pin it for
        # byte-identical report output.
        assert list(totals) == [
            "total_architectures",
            "total_models",
            "total_verified",
            "total_provisional",
        ]

    def test_matches_checked_in_registry_header(self):
        # The checked-in header was written under the same rules; recomputing it
        # from the models list must reproduce it exactly.
        data = load_supported_models_raw()
        totals = recompute_registry_totals(data["models"])
        assert totals == {key: data[key] for key in totals}
