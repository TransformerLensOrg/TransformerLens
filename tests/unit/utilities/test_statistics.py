"""Closed-form checks for the per-example statistics helpers."""

import itertools
import math
from fractions import Fraction

import pytest
import torch

from transformer_lens.utilities import (
    bootstrap_ci,
    derive_generator,
    sign_flip_permutation_pvalue,
    standard_error,
    success_rate_ci,
)
from transformer_lens.utilities.statistics import _binomial_cdf, _count_hits


def test_derive_generator_is_keyed_and_stable():
    a = torch.rand(3, generator=derive_generator(0, "bootstrap"))
    b = torch.rand(3, generator=derive_generator(0, "bootstrap"))
    c = torch.rand(3, generator=derive_generator(0, "permutation"))
    d = torch.rand(3, generator=derive_generator(1, "bootstrap"))
    assert torch.equal(a, b)
    assert not torch.equal(a, c) and not torch.equal(a, d)


def test_standard_error_closed_form():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]])
    se = standard_error(x, dim=-1)
    expected = torch.tensor([math.sqrt(5.0 / 3.0) / 2.0, 0.0])
    torch.testing.assert_close(se, expected)
    assert standard_error(x, dim=0).shape == (4,)
    with pytest.raises(ValueError, match="at least 2"):
        standard_error(torch.ones(3, 1), dim=-1)


def test_bootstrap_ci_collapses_on_a_constant_and_brackets_the_mean():
    constant = torch.full((2, 8), 3.5)
    low, high = bootstrap_ci(constant, dim=-1, n_resamples=50)
    torch.testing.assert_close(low, constant[:, 0])
    torch.testing.assert_close(high, constant[:, 0])
    torch.manual_seed(0)
    x = torch.randn(3, 5, 40) + 2.0
    low, high = bootstrap_ci(x, dim=-1, n_resamples=400, confidence=0.9)
    mean = x.mean(-1)
    assert low.shape == (3, 5) and (low <= mean).all() and (high >= mean).all()
    assert (high - low > 0).all()


def test_bootstrap_ci_uses_the_resampled_axis_and_the_seed():
    torch.manual_seed(0)
    small = torch.randn(1, 8)
    big = small.repeat(1, 8)  # 64 samples with the same mean and spread
    lo_s, hi_s = bootstrap_ci(small, n_resamples=500, generator=derive_generator(3))
    lo_b, hi_b = bootstrap_ci(big, n_resamples=500, generator=derive_generator(3))
    assert (hi_b - lo_b) < (hi_s - lo_s)
    again = bootstrap_ci(small, n_resamples=500, generator=derive_generator(3))
    assert torch.equal(again[0], lo_s) and torch.equal(again[1], hi_s)
    other = bootstrap_ci(small, n_resamples=500, generator=derive_generator(4))
    assert not torch.equal(other[0], lo_s)


def test_bootstrap_ci_matches_per_cell_loop_and_median():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 12)
    lo, hi = bootstrap_ci(x, dim=-1, n_resamples=64, generator=derive_generator(7), chunk_size=16)
    lo_loop, hi_loop = bootstrap_ci(
        x.reshape(-1, 12), dim=-1, n_resamples=64, generator=derive_generator(7), chunk_size=16
    )
    torch.testing.assert_close(lo.reshape(-1), lo_loop)
    torch.testing.assert_close(hi.reshape(-1), hi_loop)
    lo_m, hi_m = bootstrap_ci(x, statistic="median", n_resamples=32)
    assert lo_m.shape == (2, 3) and (lo_m <= hi_m).all()
    with pytest.raises(ValueError, match="statistic"):
        bootstrap_ci(x, statistic="mode")
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(x, confidence=1.0)


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_bootstrap_ci_preserves_float64_values(statistic: str):
    x = torch.full((2, 8), 1.00000001, dtype=torch.float64)
    low, high = bootstrap_ci(x, statistic=statistic, n_resamples=32)
    torch.testing.assert_close(low, x[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(high, x[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_statistics_support_floating_dtypes(dtype: torch.dtype):
    x = torch.tensor([1.0, 2.0], dtype=dtype)
    low, high = bootstrap_ci(torch.ones(8, dtype=dtype), n_resamples=32)
    assert low.dtype == high.dtype == dtype
    assert low.item() == high.item() == 1.0
    pvalue = sign_flip_permutation_pvalue(x)
    assert pvalue.dtype == torch.promote_types(dtype, torch.float32)
    assert pvalue.item() == 0.5


@pytest.mark.parametrize("exact_max_n", [0, 12])
def test_sign_flip_float64_is_invariant_to_positive_scale(exact_max_n: int):
    effects = torch.tensor([1.0, 2.0], dtype=torch.float64)
    expected = sign_flip_permutation_pvalue(effects, exact_max_n=exact_max_n)
    actual = sign_flip_permutation_pvalue(effects * 1e-50, exact_max_n=exact_max_n)
    assert expected.item() < 1.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sign_flip_preserves_small_float64_effects():
    effects = torch.tensor([1.0, 1e-8], dtype=torch.float64)
    p = sign_flip_permutation_pvalue(effects)
    assert p.dtype == torch.float64
    assert p.item() == 0.5


def test_sign_flip_counts_float64_ties_after_rounding():
    effects = torch.tensor([-1.0, 0.4, -0.5, 0.6], dtype=torch.float64)
    p = sign_flip_permutation_pvalue(effects)
    assert p.dtype == torch.float64
    assert p.item() == 0.875

    scaled_p = sign_flip_permutation_pvalue(effects * 1e-50)
    assert scaled_p.dtype == torch.float64
    assert scaled_p.item() == 0.875


@pytest.mark.parametrize("alternative, expected", [("greater", 11 / 16), ("less", 7 / 16)])
def test_sign_flip_counts_float64_ties_for_one_sided_alternatives(
    alternative: str, expected: float
):
    effects = torch.tensor([-1.0, 0.4, -0.5, 0.6], dtype=torch.float64)
    p = sign_flip_permutation_pvalue(effects, alternative=alternative)
    assert p.dtype == torch.float64
    assert p.item() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_sign_flip_counts_a_one_ulp_tie_on_every_platform(dtype: torch.dtype, alternative: str):
    # Built directly: whether a sign-flip mean lands an ulp off its tie depends on summation order.
    observed = torch.tensor([0.3], dtype=dtype)
    away = torch.tensor(math.inf if alternative == "less" else -math.inf, dtype=dtype)
    tie = torch.nextafter(observed, away)
    distinct = torch.tensor([1.0 if alternative == "less" else 0.0], dtype=dtype)
    hits = _count_hits(torch.cat([tie, distinct]), observed, alternative)
    assert hits.item() == 1


def test_sign_flip_does_not_count_distinct_means_as_ties_near_zero():
    effects = torch.tensor([1.0, -1e-8], dtype=torch.float64)
    for scale in (1.0, 1e-50):
        scaled_effects = effects * scale
        greater = sign_flip_permutation_pvalue(scaled_effects, alternative="greater")
        less = sign_flip_permutation_pvalue(scaled_effects, alternative="less")
        assert greater.dtype == less.dtype == torch.float64
        assert greater.item() == 0.5
        assert less.item() == 0.75


def test_sign_flip_monte_carlo_preserves_float64_pvalue_dtype():
    effects = torch.ones(13, dtype=torch.float64)
    p = sign_flip_permutation_pvalue(
        effects, n_permutations=64, generator=derive_generator(23), exact_max_n=12
    )
    assert p.dtype == torch.float64


def test_sign_flip_exact_enumeration_matches_hand_count():
    effects = torch.tensor([[1.0, 2.0, 3.0, -0.5]])
    observed = effects.mean().abs()
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=4):
        if abs((effects[0] * torch.tensor(signs)).mean()) >= observed:
            hits += 1
    p = sign_flip_permutation_pvalue(effects, dim=-1)
    assert p.shape == (1,)
    torch.testing.assert_close(p[0], torch.tensor(hits / 16.0))
    # exact mode has no seed dependence
    assert torch.equal(p, sign_flip_permutation_pvalue(effects, generator=derive_generator(99)))
    greater = sign_flip_permutation_pvalue(effects, alternative="greater")
    less = sign_flip_permutation_pvalue(effects, alternative="less")
    assert greater < less


def test_sign_flip_monte_carlo_separates_effect_from_null():
    torch.manual_seed(0)
    strong = torch.randn(1, 40) * 0.1 + 2.0
    mirrored = torch.cat([torch.arange(1.0, 21.0), -torch.arange(1.0, 21.0)]).unsqueeze(0)
    p_strong = sign_flip_permutation_pvalue(
        strong, n_permutations=500, generator=derive_generator(1)
    )
    p_null = sign_flip_permutation_pvalue(
        mirrored, n_permutations=500, generator=derive_generator(1)
    )
    assert p_strong.item() < 0.01
    # the add-one correction keeps a Monte Carlo p-value away from zero
    assert p_strong.item() == pytest.approx(1.0 / 501.0, rel=1e-5)
    assert p_null.item() > 0.2
    same = sign_flip_permutation_pvalue(strong, n_permutations=500, generator=derive_generator(1))
    assert torch.equal(same, p_strong)
    with pytest.raises(ValueError, match="alternative"):
        sign_flip_permutation_pvalue(strong, alternative="both")


def test_success_rate_ci_bounds_bracket_point_estimate_and_lie_in_unit_interval():
    successes = [True, True, False, True, False, True, True, False]
    result = success_rate_ci(successes)
    assert result.ci_low <= result.point_estimate <= result.ci_high
    assert 0.0 <= result.ci_low and result.ci_high <= 1.0
    assert result.point_estimate == pytest.approx(sum(successes) / len(successes))


def test_success_rate_ci_pins_zero_of_six_upper_bound():
    # A percentile bootstrap returns [0, 0] here; the exact interval must stay wide.
    result = success_rate_ci([False] * 6)
    assert result.ci_low == 0.0
    assert result.ci_high == pytest.approx(0.4592581, abs=1e-6)


def test_success_rate_ci_pins_all_success_lower_bound():
    result = success_rate_ci([True] * 6)
    assert result.ci_high == 1.0
    assert result.ci_low == pytest.approx(0.5407419, abs=1e-6)


def test_success_rate_ci_matches_closed_form_at_the_boundaries():
    # Closed forms at k=0 and k=n pin the bisection direction; reversing it yields [0, 1].
    assert success_rate_ci([False] * 6).ci_high == pytest.approx(1 - 0.025 ** (1 / 6))
    assert success_rate_ci([True] * 6).ci_low == pytest.approx(0.025 ** (1 / 6))


def test_success_rate_ci_narrows_as_the_trial_count_grows():
    # A degenerate interval would not move with the trial count.
    small = success_rate_ci([False] * 6)
    large = success_rate_ci([False] * 600)
    assert large.ci_high < small.ci_high


@pytest.mark.parametrize(
    "k, n, p",
    [(1000, 2000, 0.5), (190, 200, 0.97), (1995, 2000, 0.999), (5, 2000, 0.01)],
)
def test_binomial_cdf_matches_exact_rational_sum_at_large_n(k: int, n: int, p: float):
    # At these n, (1 - p) ** n underflows to 0.0, so a pmf recurrence started from it returns 0.0.
    assert _binomial_cdf(k, n, p) == pytest.approx(float(_exact_binomial_cdf(k, n, p)), rel=1e-9)


def _exact_binomial_cdf(k: int, n: int, p: float) -> Fraction:
    """Exact ``P(X <= k)`` over integers, summing the shorter of the two tails."""
    # Per-term Fraction arithmetic re-reduces ~n-bit denominators and took ~18 s at n=2000.
    a, b = Fraction(p).as_integer_ratio()

    def numerator(indices: range) -> int:
        return sum(math.comb(n, i) * a**i * (b - a) ** (n - i) for i in indices)

    if k < n - k:
        return Fraction(numerator(range(k + 1)), b**n)
    return 1 - Fraction(numerator(range(k + 1, n + 1)), b**n)


@pytest.mark.parametrize(
    "k, n, low, high",
    [
        # Reference bounds from scipy.stats.beta.ppf, rounded to four places.
        (199, 200, 0.9725, 0.9999),
        (297, 300, 0.9711, 0.9979),
        (1000, 2000, 0.4779, 0.5221),
        (1990, 2000, 0.9908, 0.9976),
    ],
)
def test_success_rate_ci_matches_reference_bounds_at_large_n(
    k: int, n: int, low: float, high: float
):
    result = success_rate_ci([True] * k + [False] * (n - k))
    assert result.ci_low == pytest.approx(low, abs=1e-4)
    assert result.ci_high == pytest.approx(high, abs=1e-4)
    assert result.ci_low <= result.point_estimate <= result.ci_high


@pytest.mark.parametrize("n", [200, 1000])
def test_success_rate_ci_all_success_lower_bound_at_large_n(n: int):
    assert success_rate_ci([True] * n).ci_low == pytest.approx(0.025 ** (1 / n))


def test_success_rate_ci_records_trial_and_success_counts():
    result = success_rate_ci([True, False, True])
    assert result.n_trials == 3
    assert result.n_successes == 2


def test_success_rate_ci_is_deterministic():
    successes = [True, False, True]
    assert success_rate_ci(successes) == success_rate_ci(successes)


def test_success_rate_ci_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        success_rate_ci([])


def test_success_rate_ci_old_analysis_path_forwards_with_a_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="import it from transformer_lens.utilities"):
        from transformer_lens.tools.analysis import success_rate_ci as forwarded
    assert forwarded is success_rate_ci

    from transformer_lens.tools import analysis

    with pytest.raises(AttributeError):
        getattr(analysis, "not_an_analysis_name")
