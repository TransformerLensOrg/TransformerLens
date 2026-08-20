"""Gradient mismatches are graded on scale-aware statistics, not worst-case elements.

Registering backward hooks forces normalization off HF's native autograd onto the
python-norm path, which shifts results at float-rounding scale. Elementwise
`allclose` graded that shift as failure: one element of 55,296 crossing the
tolerance scored the same as a real divergence. The band accepts a mismatch only
when it is both diffuse (rel_l2) and localized to a handful of elements (COUNT,
not fraction — detection guarantees count >= 1, so a fractional guard was
unsatisfiable below 10,000 elements and re-created the false failure on
gemma-3-270m's 6,912-element MQA hook_rot_k).
"""

import pytest

from transformer_lens.benchmarks.backward_gradients import (
    OVER_TOLERANCE_MAX_ELEMENTS,
    REL_L2_TOLERANCE,
    gradient_mismatch_is_numerical_noise,
)

# Measured bridge-vs-HT fallback noise. Only mismatches the detection gate records
# can reach the classifier, so every fixture has count >= 1 (frac-zero rows from
# the original study never reach it and prove nothing).
NOISE = [
    ("Qwen3-0.6B rot_q [1,27,16,128]", 1.714e-05, 1),
    ("gemma-3-270m rot_k [1,27,1,256] (MQA, 6912 elements)", 5.0e-05, 1),
]
# Injected bugs of known severity on the Qwen3 rot_q tensor (55,296 elements).
BUGS = [
    ("one head scaled 1%", 2.97e-03, 758),
    ("uniform scale 0.1%", 1.00e-03, 432),
    ("60 elements +50", 1.09e-02, 60),
]


@pytest.mark.parametrize("label,rel_l2,count", NOISE)
def test_fallback_noise_is_accepted(label: str, rel_l2: float, count: int) -> None:
    assert gradient_mismatch_is_numerical_noise(rel_l2, count), label


@pytest.mark.parametrize("label,rel_l2,count", BUGS)
def test_real_divergence_is_rejected(label: str, rel_l2: float, count: int) -> None:
    assert not gradient_mismatch_is_numerical_noise(rel_l2, count), label


def test_thresholds_keep_a_margin_on_both_dimensions() -> None:
    """Both guards must sit clear of both populations, not graze either."""
    worst_noise_rel = max(rel for _, rel, _ in NOISE)
    best_bug_rel = min(rel for _, rel, _ in BUGS)
    assert worst_noise_rel * 2 <= REL_L2_TOLERANCE, worst_noise_rel
    assert best_bug_rel >= REL_L2_TOLERANCE * 5, best_bug_rel
    worst_noise_count = max(count for _, _, count in NOISE)
    best_bug_count = min(count for _, _, count in BUGS)
    assert worst_noise_count * 3 <= OVER_TOLERANCE_MAX_ELEMENTS + 1, worst_noise_count
    assert best_bug_count >= OVER_TOLERANCE_MAX_ELEMENTS * 10, best_bug_count


def test_localized_divergence_is_rejected_even_when_diffuse_error_is_small() -> None:
    """The count is the guard: a concentrated error rel_l2 would dilute still fails."""
    assert not gradient_mismatch_is_numerical_noise(REL_L2_TOLERANCE / 10, 60)
    assert not gradient_mismatch_is_numerical_noise(1e-2, 1)


def test_zero_reference_divergence_is_rejected() -> None:
    """A zero reference gradient with a nonzero bridge gradient records
    rel_l2=inf in the benchmark body — maximal divergence, never noise."""
    assert not gradient_mismatch_is_numerical_noise(float("inf"), 1)


def test_boundaries_are_inclusive() -> None:
    """Pin the <= contract on both dimensions."""
    assert gradient_mismatch_is_numerical_noise(REL_L2_TOLERANCE, OVER_TOLERANCE_MAX_ELEMENTS)
    assert not gradient_mismatch_is_numerical_noise(
        REL_L2_TOLERANCE * 1.01, OVER_TOLERANCE_MAX_ELEMENTS
    )
    assert not gradient_mismatch_is_numerical_noise(
        REL_L2_TOLERANCE, OVER_TOLERANCE_MAX_ELEMENTS + 1
    )


class TestMismatchStatsHelper:
    """The stats the classifier consumes, driven with synthetic tensors."""

    def test_zero_reference_nonzero_bridge_is_infinite(self) -> None:
        import torch

        from transformer_lens.benchmarks.backward_gradients import (
            gradient_mismatch_stats,
        )

        stats = gradient_mismatch_stats(torch.ones(100), torch.zeros(100), 0.2, 3e-4)
        assert stats["rel_l2"] == float("inf")
        assert not gradient_mismatch_is_numerical_noise(stats["rel_l2"], stats["over_count"])

    def test_matching_zeros_agree(self) -> None:
        import torch

        from transformer_lens.benchmarks.backward_gradients import (
            gradient_mismatch_stats,
        )

        stats = gradient_mismatch_stats(torch.zeros(100), torch.zeros(100), 0.2, 3e-4)
        assert stats["rel_l2"] == 0.0 and stats["over_count"] == 0

    def test_over_count_matches_detection_predicate(self) -> None:
        import torch

        from transformer_lens.benchmarks.backward_gradients import (
            gradient_mismatch_stats,
        )

        ref = torch.full((6912,), 258.0)  # gemma-3-270m rot_k scale
        bridge = ref.clone()
        bridge[0] += 0.5  # one element past atol + rtol*|ref|
        stats = gradient_mismatch_stats(bridge, ref, 0.2, 3e-4)
        assert stats["over_count"] == 1
        assert gradient_mismatch_is_numerical_noise(stats["rel_l2"], stats["over_count"]), stats


class TestFp32GradientPredicate:
    def test_reduced_precision_needs_upcast(self) -> None:
        import torch

        from transformer_lens.benchmarks.backward_gradients import needs_fp32_gradients

        assert needs_fp32_gradients(torch.bfloat16)
        assert needs_fp32_gradients(torch.float16)
        assert not needs_fp32_gradients(torch.float32)
        assert not needs_fp32_gradients(torch.float64)
        assert not needs_fp32_gradients(None)
