"""Cheap pure-function unit tests for the verify_models control-gates.

These gates were previously exercised only by multi-hour real-model runs, so a slot-name
drift or a phase-filter regression would slip through CI. They are pure functions, so we
pin them directly:
- ``_phases_to_run`` — restricts requested phases to the adapter's ``applicable_phases``
  (SSM/recurrent families now run all four); phases 7/8 always pass through.
- ``_extract_phase_scores`` — per-phase pass-rate, with SKIPPED results excluded.
- ``_is_ssm_mixer_internal`` — the component-benchmark skip for SSM mixer internals.
"""
from types import SimpleNamespace

from transformer_lens.benchmarks.component_outputs import _is_ssm_mixer_internal
from transformer_lens.benchmarks.utils import BenchmarkSeverity
from transformer_lens.tools.model_registry.registry_io import (
    STATUS_PROVISIONAL,
    STATUS_VERIFIED,
)
from transformer_lens.tools.model_registry.verify_models import (
    _extract_phase_scores,
    _pass_status,
    _phases_to_run,
)

SSM_ARCHS = ["MambaForCausalLM", "Mamba2ForCausalLM", "NemotronHForCausalLM"]


class TestPhasesToRun:
    def test_ssm_families_run_all_four_phases(self):
        # Regression guard: SSM/hybrid families are no longer skipped or [4]-only.
        for arch in SSM_ARCHS:
            assert _phases_to_run(arch, [1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_phases_7_and_8_always_pass_through(self):
        # 7/8 are gated by is_multimodal/is_audio elsewhere, never filtered here.
        assert _phases_to_run("MambaForCausalLM", [1, 7, 8]) == [1, 7, 8]

    def test_unknown_architecture_defaults_to_all_phases(self):
        assert _phases_to_run("NotARealArchitecture", [1, 2, 3, 4]) == [1, 2, 3, 4]

    def test_restricted_architecture_is_filtered(self):
        # Find any adapter with a restricted applicable_phases and confirm filtering.
        from transformer_lens.factories.architecture_adapter_factory import (
            SUPPORTED_ARCHITECTURES,
        )

        restricted = None
        for name, adapter in SUPPORTED_ARCHITECTURES.items():
            phases = getattr(adapter, "applicable_phases", [1, 2, 3, 4])
            if phases and set(phases) != {1, 2, 3, 4}:
                restricted = (name, phases)
                break
        if restricted is None:
            return  # no restricted-phase adapter in the registry; nothing to assert
        name, phases = restricted
        result = _phases_to_run(name, [1, 2, 3, 4])
        assert result == [p for p in [1, 2, 3, 4] if p in phases]
        assert set(result) <= set(phases)


def _result(phase, passed, severity, details=None):
    return SimpleNamespace(phase=phase, passed=passed, severity=severity, details=details)


class TestExtractPhaseScores:
    def test_skipped_results_excluded_from_score(self):
        # A SKIPPED centering test (passed=False) must NOT drag the phase down — this is
        # exactly the SSM P3=90 bug. One real PASS + one SKIPPED → 100.
        results = [
            _result(3, True, BenchmarkSeverity.INFO),
            _result(3, False, BenchmarkSeverity.SKIPPED),
        ]
        assert _extract_phase_scores(results)[3] == 100.0

    def test_mixed_pass_fail_averaged(self):
        results = [
            _result(1, True, BenchmarkSeverity.INFO),
            _result(1, False, BenchmarkSeverity.DANGER),
        ]
        assert _extract_phase_scores(results)[1] == 50.0

    def test_all_skipped_phase_omitted(self):
        results = [_result(2, False, BenchmarkSeverity.SKIPPED)]
        assert 2 not in _extract_phase_scores(results)

    def test_phase4_uses_quality_score_from_details(self):
        results = [_result(4, True, BenchmarkSeverity.INFO, details={"score": 87.5})]
        assert _extract_phase_scores(results)[4] == 87.5


class TestIsSSMMixerInternal:
    def test_mixer_and_linear_attn_internals_skipped(self):
        assert _is_ssm_mixer_internal("blocks.0.mixer.conv1d")
        assert _is_ssm_mixer_internal("blocks.5.mixer.in_proj")
        assert _is_ssm_mixer_internal("blocks.0.linear_attn.conv1d")
        assert _is_ssm_mixer_internal("blocks.0.mixer.conv1d.weight")

    def test_mixer_node_itself_is_tested(self):
        # The mixer node (path ending in the slot) is still benchmarked end-to-end.
        assert not _is_ssm_mixer_internal("blocks.0.mixer")
        assert not _is_ssm_mixer_internal("blocks.0.linear_attn")

    def test_transformer_components_untouched(self):
        for path in ("blocks.0.attn.q", "blocks.0.mlp.out", "embed", "unembed", "ln_final"):
            assert not _is_ssm_mixer_internal(path)


class TestPassStatus:
    """A passing run is VERIFIED only when numerically compared to HF; a
    --no-hf-reference (structural-only) pass is PROVISIONAL, not verified."""

    def test_hf_reference_passes_verify(self):
        assert _pass_status(use_hf_reference=True) == STATUS_VERIFIED

    def test_no_hf_reference_is_provisional(self):
        assert _pass_status(use_hf_reference=False) == STATUS_PROVISIONAL
