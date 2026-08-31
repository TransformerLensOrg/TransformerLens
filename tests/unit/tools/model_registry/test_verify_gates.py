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
    _check_phase_scores,
    _measured_nothing,
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


class TestCheckPhaseScores:
    """A required modality phase must not pass by being absent from the scores."""

    def test_required_modality_phase_absent_fails(self):
        # All P7 tests SKIPPED -> phase 7 is omitted entirely (see
        # test_all_skipped_phase_omitted), so the gate must not infer a pass.
        scores = {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0}
        assert _check_phase_scores(scores, []) is None
        error = _check_phase_scores(scores, [], required_phases={1, 4, 7})
        assert error is not None and "P7=NULL" in error

    def test_absent_modality_phase_ignored_when_not_required(self):
        # A partial run (--phases 1 2) must not be failed for a missing P7.
        error = _check_phase_scores({1: 100.0, 2: 100.0}, [], required_phases={1})
        assert error is None


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


class TestMeasuredNothing:
    """A run whose requested phases were all pruned by the adapter's
    ``applicable_phases`` produces no scores, and no score check fires to catch
    it -- dicta-il/BEREL_3.0 was recorded VERIFIED with every phase None.
    """

    def test_empty_scores_measured_nothing(self):
        assert _measured_nothing({}) is True

    def test_all_none_measured_nothing(self):
        assert _measured_nothing({1: None, 4: None}) is True

    def test_zero_is_a_real_score(self):
        """P4=0.0 is a measurement, not a missing one -- truthiness would drop it."""
        assert _measured_nothing({4: 0.0}) is False

    def test_any_score_counts(self):
        assert _measured_nothing({1: 100.0, 4: None}) is False



class TestPublishedParamCount:
    """The config formula bills every layer for full attention and an MLP, which
    over-counts a hybrid Mamba/attention stack about fourfold and skips the model
    as too large. The hub publishes the real count for most repos, so prefer it.
    """

    def test_published_count_wins_over_the_formula(self, monkeypatch):
        """The formula must not even run when the hub has published a count."""
        import transformer_lens.utilities.hf_utils as hf_utils
        from transformer_lens.tools.model_registry import verify_models

        def _explode(*args, **kwargs):
            raise AssertionError("config formula ran despite a published count")

        monkeypatch.setattr(verify_models, "published_param_count", lambda _mid: 8_889_000_000)
        monkeypatch.setattr(hf_utils, "autoconfig_with_remote_post_init_compat", _explode)
        assert verify_models.estimate_model_params("nvidia/NVIDIA-Nemotron-Nano-9B-v2") == 8_889_000_000

    def test_falls_back_to_the_formula_when_unpublished(self, monkeypatch):
        import transformer_lens.utilities.hf_utils as hf_utils
        from transformer_lens.tools.model_registry import verify_models

        config = SimpleNamespace(
            hidden_size=768, num_attention_heads=12, num_hidden_layers=12,
            intermediate_size=3072, vocab_size=50257, hidden_act="gelu",
        )
        monkeypatch.setattr(verify_models, "published_param_count", lambda _mid: None)
        monkeypatch.setattr(
            hf_utils, "autoconfig_with_remote_post_init_compat", lambda *a, **k: config
        )
        assert verify_models.estimate_model_params("gpt2") > 100_000_000

    def test_hub_failure_returns_none_rather_than_raising(self, monkeypatch):
        """A gated repo or network blip must fall through to the formula."""
        import huggingface_hub
        from transformer_lens.tools.model_registry import verify_models

        class _Boom:
            def model_info(self, *a, **k):
                raise ConnectionError("hub unreachable")

        monkeypatch.setattr(huggingface_hub, "HfApi", _Boom)
        assert verify_models.published_param_count("any/model") is None
