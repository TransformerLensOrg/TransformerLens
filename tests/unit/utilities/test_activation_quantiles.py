"""Checks for streaming per-unit quantiles and the quantile-band ablation hook."""

import pytest
import torch
from torch import nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedRootModule import HookedRootModule
from transformer_lens.utilities import ActivationQuantiles, band_ablation_hook


class _Toy(HookedRootModule):
    def __init__(self, d: int = 4):
        super().__init__()
        torch.manual_seed(0)
        self.inp = nn.Linear(d, d)
        self.hook_post = HookPoint()
        self.setup()
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hook_post(self.inp(x))


def test_quantiles_are_exact_below_max_samples_across_batches_and_mask():
    torch.manual_seed(0)
    a, b = torch.randn(3, 5, 4), torch.randn(2, 7, 4)
    mask = torch.rand(2, 7) > 0.3
    stats = ActivationQuantiles(max_samples=1000)
    stats.update(a)
    stats.update(b, mask=mask)
    rows = torch.cat([a.reshape(-1, 4), b[mask]])
    levels = torch.tensor([0.0, 0.03, 0.5, 0.97, 1.0])
    torch.testing.assert_close(stats.quantiles(levels), torch.quantile(rows, levels, dim=0))
    torch.testing.assert_close(stats.quantiles(0.5), rows.quantile(0.5, dim=0))
    torch.testing.assert_close(stats.mean, rows.mean(0))
    assert stats.n_seen == stats.n_samples == rows.shape[0]


def test_sampling_bounds_memory_keeps_real_rows_and_an_exact_mean():
    rows = torch.randn(5000, 3)
    stats = ActivationQuantiles(max_samples=500, generator=torch.Generator().manual_seed(0))
    for chunk in rows.split(700):
        stats.update(chunk)
    assert stats.n_seen == 5000 and stats.n_samples == 500
    sample = stats._sample
    assert sample is not None
    # Every kept row is an input row, and none is kept twice.
    matches = (sample[:, None, :] == rows[None, :, :]).all(-1)
    assert (matches.sum(1) == 1).all() and matches.any(0).sum() == 500
    torch.testing.assert_close(stats.mean, rows.mean(0), rtol=1e-5, atol=1e-6)


def test_sample_is_uniform_over_the_stream():
    # Bottom-k keys over many small batches should not favour early or late rows.
    index = torch.arange(10_000, dtype=torch.float32)[:, None]
    kept_means = []
    for seed in range(20):
        stats = ActivationQuantiles(max_samples=1000, generator=torch.Generator().manual_seed(seed))
        for chunk in index.split(37):
            stats.update(chunk)
        assert stats._sample is not None
        kept_means.append(stats._sample.mean().item())
    # Mean kept index of a uniform 1000-of-10000 sample: 4999.5 with s.e. ~ 86 per run.
    assert abs(sum(kept_means) / len(kept_means) - 4999.5) < 100


def test_sampled_quantiles_are_close_and_reproducible():
    rows = torch.rand(20_000, 2)
    q = torch.tensor([0.03, 0.97])

    def run(seed: int) -> torch.Tensor:
        stats = ActivationQuantiles(max_samples=4000, generator=torch.Generator().manual_seed(seed))
        for chunk in rows.split(1000):
            stats.update(chunk)
        return stats.quantiles(q)

    torch.testing.assert_close(run(1), torch.quantile(rows, q, dim=0), atol=0.02, rtol=0)
    torch.testing.assert_close(run(1), run(1))


def test_low_precision_activations_are_accepted():
    x = torch.randn(10, 3).to(torch.bfloat16)
    stats = ActivationQuantiles()
    stats.update(x)
    torch.testing.assert_close(stats.quantiles(0.5), x.float().quantile(0.5, dim=0))


def test_quantiles_input_validation():
    with pytest.raises(ValueError, match="between 1 and"):
        ActivationQuantiles(max_samples=0)
    with pytest.raises(ValueError, match="between 1 and"):
        ActivationQuantiles(max_samples=2**24 + 1)
    with pytest.raises(ValueError, match="positive integer"):
        ActivationQuantiles(max_samples=True)
    stats = ActivationQuantiles()
    with pytest.raises(ValueError, match="no activations"):
        stats.quantiles(0.5)
    with pytest.raises(ValueError, match="no activations"):
        stats.mean
    with pytest.raises(TypeError, match="floating-point"):
        stats.update(torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="mask"):
        stats.update(torch.ones(2, 3), mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask"):
        stats.update(torch.ones(2, 3), mask=torch.ones(2))
    stats.update(torch.ones(2, 3))
    with pytest.raises(ValueError, match="3 units"):
        stats.update(torch.ones(2, 4))
    for bad in (1.5, -0.1, float("nan"), torch.tensor([[0.5]])):
        with pytest.raises(ValueError):
            stats.quantiles(bad)


def test_all_masked_batch_is_a_no_op():
    stats = ActivationQuantiles()
    stats.update(torch.ones(2, 3), mask=torch.zeros(2, dtype=torch.bool))
    assert stats.n_seen == 0
    with pytest.raises(ValueError, match="no activations"):
        stats.quantiles(0.5)


def test_band_hook_replaces_inside_inclusive_band_and_keeps_both_tails():
    low = torch.tensor([-1.0, 0.0, 2.0])
    high = torch.tensor([1.0, 0.0, 3.0])
    x = torch.tensor([[-1.0, 0.0, 3.0], [-1.5, 0.1, 1.0], [1.0, -0.1, 2.5]])
    original = x.clone()
    out = band_ablation_hook(low, high)(x, hook=None)
    expected = torch.tensor([[0.0, 0.0, 0.0], [-1.5, 0.1, 1.0], [0.0, -0.1, 0.0]])
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(x, original)


def test_band_hook_scalar_thresholds_and_per_unit_value():
    value = torch.tensor([10.0, 20.0])
    out = band_ablation_hook(-0.5, 0.5, value)(torch.tensor([[0.0, 0.2], [0.9, -0.4]]), hook=None)
    torch.testing.assert_close(out, torch.tensor([[10.0, 20.0], [0.9, 20.0]]))


def test_band_hook_preserves_dtype():
    x = torch.tensor([[0.25, 4.0]], dtype=torch.bfloat16)
    out = band_ablation_hook(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))(x, hook=None)
    assert out.dtype == torch.bfloat16
    assert out.tolist() == [[0.0, 4.0]]


def test_band_hook_does_not_round_scalar_thresholds_to_activation_precision():
    # 0.1001 rounds to bf16's 0.10009765625, which sits below the threshold and must be kept.
    x = torch.tensor([[0.10009765625]], dtype=torch.bfloat16)
    assert band_ablation_hook(0.1001, 1.0)(x, hook=None).item() == x.item()
    assert band_ablation_hook(torch.tensor([0.1001]), 1.0)(x, hook=None).item() == x.item()
    # A float64 threshold just above a float32 activation rounds onto it in float32.
    x32 = torch.tensor([0.1])
    low64 = x32.double() + 1e-12
    assert band_ablation_hook(low64, 1.0)(x32, hook=None).item() == x32.item()


def test_sample_does_not_alias_the_input():
    x = torch.randn(4, 3)
    stats = ActivationQuantiles()
    stats.update(x)
    expected = x.quantile(1.0, dim=0)
    x.zero_()
    torch.testing.assert_close(stats.quantiles(1.0), expected)


def test_band_hook_validation():
    with pytest.raises(ValueError, match="low must be <= high"):
        band_ablation_hook(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        band_ablation_hook(float("nan"), 1.0)
    with pytest.raises(ValueError, match="scalar or a"):
        band_ablation_hook(torch.zeros(2, 2), torch.ones(2, 2))
    with pytest.raises(ValueError, match="same number of units"):
        band_ablation_hook(torch.zeros(2), torch.ones(3))
    with pytest.raises(ValueError, match="same number of units"):
        band_ablation_hook(torch.zeros(3), torch.ones(3), torch.zeros(4))
    hook_fn = band_ablation_hook(torch.zeros(3), torch.ones(3))
    with pytest.raises(ValueError, match="3 units but the activation has 4"):
        hook_fn(torch.zeros(2, 4), hook=None)


def test_quantiles_then_band_ablation_through_run_with_hooks():
    model = _Toy()
    torch.manual_seed(1)
    inputs = [torch.randn(3, 5, 4) for _ in range(4)]
    stats = ActivationQuantiles()
    for x in inputs:
        _, cache = model.run_with_cache(x, names_filter="hook_post")
        stats.update(cache["hook_post"])
    low, high = stats.quantiles([0.1, 0.9])

    x = inputs[0]
    clean = model(x)
    ablated = model.run_with_hooks(x, fwd_hooks=[("hook_post", band_ablation_hook(low, high))])
    tail = (clean < low) | (clean > high)
    assert tail.any() and (~tail).any()
    torch.testing.assert_close(ablated[tail], clean[tail])
    assert (ablated[~tail] == 0).all()

    mean_ablated = model.run_with_hooks(
        x, fwd_hooks=[("hook_post", band_ablation_hook(low, high, stats.mean))]
    )
    torch.testing.assert_close(mean_ablated[~tail], stats.mean.expand_as(clean)[~tail])
