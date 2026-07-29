"""Cheap pure-function unit tests for the verify_models control-gates.

These gates were previously exercised only by multi-hour real-model runs, so a slot-name
drift or a phase-filter regression would slip through CI. They are pure functions, so we
pin them directly:
- ``_phases_to_run`` — restricts requested phases to the adapter's ``applicable_phases``
  (SSM/recurrent families now run all four); phases 7/8 always pass through.
- ``_check_phase_scores`` — per-phase thresholds and required-test gating.
- ``_is_ssm_mixer_internal`` — the component-benchmark skip for SSM mixer internals.

``extract_phase_scores`` / ``pass_status`` moved to registry_io; their tests live in
test_registry_io_helpers.py.
"""
from types import SimpleNamespace

from transformer_lens.benchmarks.component_outputs import _is_ssm_mixer_internal
from transformer_lens.benchmarks.utils import BenchmarkSeverity
from transformer_lens.tools.model_registry.verify_models import (
    _check_phase_scores,
    _default_phases_for_architecture,
    _full_and_core_phases,
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

    def test_phase_9_always_passes_through(self):
        # 9 is gated by is_visual_model elsewhere, never filtered here.
        assert _phases_to_run("MambaForCausalLM", [1, 9]) == [1, 9]

    def test_vision_arch_runs_only_phase_9(self):
        # ViT/DeiT declare applicable_phases=[]; text phases filter out, 9 stays.
        assert _phases_to_run("ViTModel", [1, 2, 3, 4, 9]) == [9]

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


class TestVisionPhaseSets:
    """Vision-only encoders verify through P9 alone — no text phases apply."""

    def test_full_and_core_phases_vision(self):
        for arch in ("ViTModel", "ViTForImageClassification", "DeiTForImageClassification"):
            assert _full_and_core_phases(arch) == ({9}, {9})

    def test_default_phases_vision(self):
        assert _default_phases_for_architecture("ViTModel") == [9]

    def test_default_phases_text_unchanged(self):
        assert _default_phases_for_architecture("GPT2LMHeadModel") == [1, 2, 3, 4]


def _result(phase, passed, severity, details=None, name="test"):
    return SimpleNamespace(
        phase=phase, passed=passed, severity=severity, details=details, name=name
    )


class TestCheckPhaseScores:
    """Phase 9 gates like 7/8: min score 75, vision_forward required, NULL fails."""

    def test_phase9_pass(self):
        results = [_result(9, True, BenchmarkSeverity.INFO, name="vision_forward")]
        assert _check_phase_scores({9: 100.0}, results) is None

    def test_phase9_below_threshold_fails(self):
        results = [_result(9, False, BenchmarkSeverity.DANGER, name="vision_cache")]
        error = _check_phase_scores({9: 50.0}, results)
        assert error is not None and "P9" in error

    def test_phase9_required_test_failure_overrides_score(self):
        results = [_result(9, False, BenchmarkSeverity.DANGER, name="vision_forward")]
        error = _check_phase_scores({9: 80.0}, results)
        assert error is not None and "vision_forward" in error

    def test_phase9_null_score_fails(self):
        error = _check_phase_scores({9: None}, [])
        assert error is not None and "P9=NULL" in error


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
