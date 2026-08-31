"""Outcome classification for the phase-4 re-run driver.

A model P4 cannot describe (masked LM, the pinned judge, an architecture whose
adapter excludes phase 4) is not a failure. Recording it as one inflates the
defect tally and trips the consecutive-failure abort mid-sweep.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "phase4_rerun.py"


@pytest.fixture(scope="module")
def driver():
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("phase4_rerun", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase4_rerun"] = module
    spec.loader.exec_module(module)
    return module


def _classify(driver, tail_line):
    """Run the marker logic over one line of child output."""
    joined = " ".join([tail_line])
    if driver._MISSING_PROFILE_MARKER in joined:
        return "missing_profile"
    if any(marker in joined for marker in driver._NOT_APPLICABLE_MARKERS):
        return "not_applicable"
    return "no_score"


class TestOutcomeMarkers:
    def test_masked_lm_is_not_a_failure(self, driver):
        assert (
            _classify(
                driver,
                "Skipping HookedTransformer reference: masked-LM is not representable causally.",
            )
            == "not_applicable"
        )

    def test_pinned_judge_is_not_a_failure(self, driver):
        assert (
            _classify(
                driver, "P4 skipped: Qwen/Qwen2.5-0.5B is the pinned judge — cannot self-score"
            )
            == "not_applicable"
        )

    def test_adapter_excluded_phase_is_not_a_failure(self, driver):
        assert (
            _classify(driver, "No phase produced a score (requested [4]) — status left unchanged")
            == "not_applicable"
        )

    def test_missing_profile_beats_the_generic_marker(self, driver):
        """A profile gap also reports "No phase produced a score", but it is the
        specific — and fixable — cause, so it must win (ai4bharat/IndicBART)."""
        line = (
            "P4 skipped: no prompts for profile 'task:denoise@hi' ... "
            "No phase produced a score (requested [4])"
        )
        assert _classify(driver, line) == "missing_profile"

    def test_a_real_failure_stays_a_failure(self, driver):
        assert _classify(driver, "SKIP: Estimated 214.3 GB exceeds 100.0 GB limit") == "no_score"

    def test_structural_outcomes_are_not_hard_failures(self, driver):
        """Hard outcomes gate the exit code and the abort streak."""
        assert "not_applicable" not in driver._HARD_OUTCOMES
        assert "missing_profile" not in driver._HARD_OUTCOMES
