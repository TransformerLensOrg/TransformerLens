"""Analytic tests for centered unembedding covariance and factorization."""

import math

import pytest
import torch

from tests.typecheck_errors import TYPECHECK_ERRORS
from transformer_lens.tools.analysis.representation_geometry import (
    _fit_unembedding_geometry,
)


def known_readout(dtype=torch.float64):
    """Four shifted token vectors with covariance R diag(4, 1) R.T."""
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=dtype)
    axes = torch.tensor([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]], dtype=dtype)
    scale = torch.diag(torch.tensor([2.0, 1.0], dtype=dtype))
    offset = torch.tensor([[7.0], [-3.0]], dtype=dtype)
    return math.sqrt(2.0) * rotation @ scale @ axes + offset, rotation


def test_known_centered_population_covariance_and_whitening():
    readout, rotation = known_readout()
    fit = _fit_unembedding_geometry(readout)
    covariance = rotation @ torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64)) @ rotation.T
    whitening = rotation @ torch.diag(torch.tensor([0.5, 1.0], dtype=torch.float64)) @ rotation.T
    square_root = rotation @ torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64)) @ rotation.T

    torch.testing.assert_close(fit.mean, torch.tensor([7.0, -3.0], dtype=torch.float64))
    torch.testing.assert_close(fit.covariance, covariance)
    torch.testing.assert_close(fit.whitening, whitening)
    torch.testing.assert_close(fit.unwhitening, square_root)
    torch.testing.assert_close(
        fit.whitening @ fit.covariance @ fit.whitening.T, torch.eye(2, dtype=torch.float64)
    )
    torch.testing.assert_close(fit.whitening @ fit.unwhitening, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(fit.eigenvalues, torch.tensor([1.0, 4.0], dtype=torch.float64))
    assert fit.diagnostics.measured_rank == 2
    assert fit.diagnostics.condition_number == pytest.approx(4.0)
    assert fit.diagnostics.regularized_condition_number == pytest.approx(4.0)
    assert fit.diagnostics.ridge is None
    assert fit.diagnostics.input_shape == (2, 4)
    assert fit.diagnostics.token_ids is None
    assert fit.diagnostics.n_tokens == 4
    assert fit.diagnostics.centered is True
    assert fit.diagnostics.covariance_normalization == "population"


def test_centering_is_translation_invariant_not_an_uncentered_second_moment():
    readout, _ = known_readout()
    shifted = readout + torch.tensor([[19.0], [11.0]], dtype=readout.dtype)
    original = _fit_unembedding_geometry(readout)
    translated = _fit_unembedding_geometry(shifted)
    torch.testing.assert_close(original.covariance, translated.covariance)
    torch.testing.assert_close(original.whitening, translated.whitening)
    assert not torch.allclose(original.covariance, readout @ readout.T / readout.shape[1])


def test_population_normalization_differs_from_sample_covariance():
    readout, _ = known_readout()
    fit = _fit_unembedding_geometry(readout)
    torch.testing.assert_close(fit.covariance, torch.cov(readout, correction=0))
    assert not torch.allclose(fit.covariance, torch.cov(readout, correction=1))


def test_whitened_covariance_and_inverse_metric_match_independent_references():
    readout, _ = known_readout()
    fit = _fit_unembedding_geometry(readout)
    rows = (readout.T - fit.mean) @ fit.whitening.T
    torch.testing.assert_close(rows.T @ rows / len(rows), torch.eye(2, dtype=readout.dtype))
    a = torch.tensor([2.0, -1.0], dtype=readout.dtype)
    b = torch.tensor([-3.0, 4.0], dtype=readout.dtype)
    expected = a @ torch.linalg.solve(fit.covariance, b)
    torch.testing.assert_close((fit.whitening @ a) @ (fit.whitening @ b), expected)


def test_explicit_ridge_changes_the_metric_and_covariance_identity():
    readout, rotation = known_readout()
    fit = _fit_unembedding_geometry(readout, ridge=0.5)
    expected = (
        rotation
        @ torch.diag(torch.tensor([4.5**-0.5, 1.5**-0.5], dtype=readout.dtype))
        @ rotation.T
    )
    identity = torch.eye(2, dtype=readout.dtype)
    torch.testing.assert_close(fit.regularized_covariance, fit.covariance + 0.5 * identity)
    torch.testing.assert_close(fit.whitening, expected)
    torch.testing.assert_close(
        fit.whitening @ fit.regularized_covariance @ fit.whitening.T, identity
    )
    assert not torch.allclose(fit.whitening @ fit.covariance @ fit.whitening.T, identity)
    assert fit.diagnostics.ridge == 0.5
    assert fit.diagnostics.regularized_condition_number == pytest.approx(3.0)


@pytest.mark.parametrize(
    "readout", [torch.zeros(2, 4), torch.ones(2, 4), torch.tensor([[1.0, -1.0], [2.0, -2.0]])]
)
def test_singular_covariance_requires_explicit_ridge(readout):
    with pytest.raises(ValueError, match="rank-deficient or near-singular"):
        _fit_unembedding_geometry(readout)
    fit = _fit_unembedding_geometry(readout, ridge=0.25)
    torch.testing.assert_close(
        fit.whitening @ fit.regularized_covariance @ fit.whitening.T,
        torch.eye(2),
        atol=2e-5,
        rtol=2e-5,
    )
    assert fit.diagnostics.measured_rank < 2
    assert math.isinf(fit.diagnostics.condition_number)
    assert math.isfinite(fit.diagnostics.regularized_condition_number)


def test_zero_covariance_with_ridge_has_exact_known_factors():
    fit = _fit_unembedding_geometry(torch.ones(3, 2, dtype=torch.float64), ridge=0.25)
    torch.testing.assert_close(fit.whitening, 2.0 * torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(fit.unwhitening, 0.5 * torch.eye(3, dtype=torch.float64))
    assert fit.diagnostics.measured_rank == 0


def test_single_token_population_is_only_invertible_with_ridge():
    readout = torch.tensor([[2.0], [-1.0]])
    with pytest.raises(ValueError, match="rank-deficient or near-singular"):
        _fit_unembedding_geometry(readout)
    fit = _fit_unembedding_geometry(readout, ridge=1.0)
    torch.testing.assert_close(fit.covariance, torch.zeros(2, 2))
    torch.testing.assert_close(fit.whitening, torch.eye(2))


def test_near_singular_and_insufficient_ridge_are_rejected():
    readout = torch.tensor([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1e-5, -1e-5]])
    with pytest.raises(ValueError, match="rank-deficient or near-singular"):
        _fit_unembedding_geometry(readout)
    with pytest.raises(ValueError, match="ridge is too small"):
        _fit_unembedding_geometry(readout, ridge=1e-12)
    fit = _fit_unembedding_geometry(readout, ridge=0.01)
    assert fit.diagnostics.measured_rank == 1


def test_condition_diagnostics_do_not_overflow_at_input_precision():
    readout = torch.tensor([[1e10, -1e10, 0.0, 0.0], [0.0, 0.0, 1e-10, -1e-10]])
    fit = _fit_unembedding_geometry(readout, rtol=0.0)
    assert fit.diagnostics.condition_number == pytest.approx(1e40, rel=1e-6)
    assert fit.diagnostics.regularized_condition_number == pytest.approx(1e40, rel=1e-6)
    assert math.isfinite(fit.diagnostics.regularized_condition_number)


def test_explicit_rank_tolerance_controls_the_boundary():
    readout, _ = known_readout()
    with pytest.raises(ValueError, match="rank-deficient or near-singular"):
        _fit_unembedding_geometry(readout, rtol=0.3)
    fit = _fit_unembedding_geometry(readout, rtol=0.2)
    assert fit.diagnostics.threshold == pytest.approx(0.8)
    assert fit.diagnostics.rtol == 0.2


def test_structural_sample_rank_limit_cannot_be_overridden():
    readout = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="rank-deficient or near-singular"):
        _fit_unembedding_geometry(readout, rtol=0.0)


def test_token_subset_has_its_own_mean_population_and_metadata():
    readout, _ = known_readout()
    fit = _fit_unembedding_geometry(readout, token_ids=[3, 0, 2])
    selected = readout[:, [3, 0, 2]]
    torch.testing.assert_close(fit.mean, selected.mean(dim=1))
    torch.testing.assert_close(fit.covariance, torch.cov(selected, correction=0))
    assert fit.diagnostics.token_ids == (3, 0, 2)
    assert fit.diagnostics.n_tokens == 3
    assert fit.diagnostics.input_shape == (2, 4)
    reordered = _fit_unembedding_geometry(readout, token_ids=[0, 2, 3])
    torch.testing.assert_close(fit.covariance, reordered.covariance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_compute_dtype_policy_and_cpu_preservation(dtype):
    readout, _ = known_readout(dtype)
    fit = _fit_unembedding_geometry(readout)
    expected = torch.float64 if dtype == torch.float64 else torch.float32
    for value in [
        fit.mean,
        fit.covariance,
        fit.regularized_covariance,
        fit.eigenvalues,
        fit.eigenvectors,
        fit.whitening,
        fit.unwhitening,
    ]:
        assert value.dtype == expected
        assert value.device == readout.device
    assert fit.diagnostics.input_dtype == dtype
    assert fit.diagnostics.compute_dtype == expected
    assert fit.diagnostics.device == readout.device
    assert fit.diagnostics.rtol == pytest.approx(2 * torch.finfo(expected).eps)


def test_explicit_high_precision_compute_and_detached_snapshot():
    readout, _ = known_readout(torch.float32)
    readout.requires_grad_()
    before = readout.detach().clone()
    fit = _fit_unembedding_geometry(readout, compute_dtype=torch.float64)
    torch.testing.assert_close(readout.detach(), before)
    assert readout.requires_grad
    snapshot = [value.clone() for value in [fit.mean, fit.covariance, fit.whitening]]
    with torch.no_grad():
        readout.fill_(100.0)
    for actual, expected in zip([fit.mean, fit.covariance, fit.whitening], snapshot):
        torch.testing.assert_close(actual, expected)
        assert not actual.requires_grad
        assert actual.grad_fn is None


@pytest.mark.parametrize(
    "readout", [torch.ones(3), torch.ones(1, 2, 3), torch.empty(0, 2), torch.empty(2, 0)]
)
def test_invalid_shapes_and_empty_dimensions(readout):
    with pytest.raises(ValueError, match="two-dimensional|non-empty"):
        _fit_unembedding_geometry(readout)


@pytest.mark.parametrize(
    "readout",
    [[[1.0]], torch.ones(2, 2, dtype=torch.int64), torch.ones(2, 2, dtype=torch.complex64)],
)
def test_invalid_tensor_types_and_dtypes(readout):
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        _fit_unembedding_geometry(readout)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_readout_is_rejected(value):
    readout, _ = known_readout()
    readout[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        _fit_unembedding_geometry(readout)


def test_ridge_outside_compute_dtype_range_is_rejected():
    readout, _ = known_readout(torch.float32)
    with pytest.raises(ValueError, match="ridge.*representable"):
        _fit_unembedding_geometry(readout, ridge=1e100)


def test_covariance_overflow_is_reported_before_factorization():
    readout = torch.tensor([[1e30, -1e30], [1e30, 1e30]], dtype=torch.float32)
    with pytest.raises(ValueError, match="covariance.*finite"):
        _fit_unembedding_geometry(readout, ridge=1.0)


@pytest.mark.parametrize("ridge", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_ridge_is_rejected(ridge):
    readout, _ = known_readout()
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS, match="ridge"):
        _fit_unembedding_geometry(readout, ridge=ridge)


@pytest.mark.parametrize("rtol", [True, -1.0, 1.0, float("nan"), float("inf")])
def test_invalid_rank_tolerance_is_rejected(rtol):
    readout, _ = known_readout()
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS, match="rtol"):
        _fit_unembedding_geometry(readout, rtol=rtol)


@pytest.mark.parametrize("token_ids", [[], [0, 0], [-1, 0], [0, 4], [True, 1], [0.5, 1]])
def test_invalid_token_selection_is_rejected(token_ids):
    readout, _ = known_readout()
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        _fit_unembedding_geometry(readout, token_ids=token_ids)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_invalid_compute_dtype_is_rejected(dtype):
    readout, _ = known_readout()
    with pytest.raises(ValueError, match="compute_dtype must"):
        _fit_unembedding_geometry(readout, compute_dtype=dtype)


def test_unsupported_device_is_rejected_without_implicit_transfer():
    with pytest.raises(ValueError, match="CPU or CUDA"):
        _fit_unembedding_geometry(torch.empty(2, 4, device="meta"))


def test_sparse_layout_is_rejected():
    with pytest.raises(ValueError, match="strided"):
        _fit_unembedding_geometry(torch.eye(2).to_sparse())


def test_noncontiguous_readout_matches_contiguous_input():
    readout, _ = known_readout()
    noncontiguous = readout.T.contiguous().T
    assert not noncontiguous.is_contiguous()
    original = _fit_unembedding_geometry(readout)
    actual = _fit_unembedding_geometry(noncontiguous)
    torch.testing.assert_close(actual.covariance, original.covariance)
    torch.testing.assert_close(actual.whitening, original.whitening)
