"""Closed-form checks for the per-example statistics helpers."""

import itertools
import math

import pytest
import torch

from transformer_lens.utilities import (
    bootstrap_ci,
    derive_generator,
    sign_flip_permutation_pvalue,
    standard_error,
)


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
