"""Analytic tests for centered unembedding covariance and factorization."""

import math

import pytest
import torch

from tests.typecheck_errors import TYPECHECK_ERRORS
from transformer_lens.tools.analysis.representation_geometry import (
    RepresentationGeometry,
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


@pytest.fixture(params=[None, 0.5], ids=["exact", "ridge"])
def geometry_contract(request):
    readout, rotation = known_readout()
    ridge = request.param
    eigenvalues = torch.tensor([4.0, 1.0], dtype=torch.float64) + (ridge or 0.0)
    covariance = rotation @ torch.diag(eigenvalues) @ rotation.T
    whitening = rotation @ torch.diag(eigenvalues.rsqrt()) @ rotation.T
    unwhitening = rotation @ torch.diag(eigenvalues.sqrt()) @ rotation.T
    return RepresentationGeometry(readout, ridge=ridge), covariance, whitening, unwhitening


@pytest.mark.parametrize("shape", [(2,), (3, 2), (2, 3, 2)])
def test_dual_transforms_match_independent_factors_and_round_trip(geometry_contract, shape):
    geometry, _, whitening, unwhitening = geometry_contract
    generator = torch.Generator().manual_seed(83)
    vector = torch.randn(shape, generator=generator, dtype=torch.float64)
    before = vector.clone()
    measured = geometry.whiten_measurement(vector)
    intervention = geometry.whiten_intervention(vector)
    torch.testing.assert_close(measured, vector @ whitening.T)
    torch.testing.assert_close(intervention, vector @ unwhitening.T)
    torch.testing.assert_close(geometry.unwhiten_measurement(measured), vector)
    torch.testing.assert_close(geometry.unwhiten_intervention(intervention), vector)
    torch.testing.assert_close(vector, before)
    assert measured.shape == shape
    assert not torch.allclose(measured, intervention)


def test_dual_transforms_preserve_raw_pairing_with_broadcasting(geometry_contract):
    geometry, _, _, _ = geometry_contract
    generator = torch.Generator().manual_seed(12)
    measurement = torch.randn(3, 1, 2, generator=generator, dtype=torch.float64)
    intervention = torch.randn(1, 4, 2, generator=generator, dtype=torch.float64)
    actual = (
        geometry.whiten_measurement(measurement) * geometry.whiten_intervention(intervention)
    ).sum(-1)
    torch.testing.assert_close(actual, (measurement * intervention).sum(-1))
    assert actual.shape == (3, 4)


def test_metric_operations_match_independent_inverse_and_covariance(geometry_contract):
    geometry, covariance, _, _ = geometry_contract
    a = torch.tensor([[[2.0, -1.0]], [[-3.0, 0.5]], [[0.0, 0.0]]], dtype=torch.float64)
    b = torch.tensor([[[1.0, 4.0], [-2.0, 3.0]]], dtype=torch.float64)
    measurement = geometry.measurement_inner_product(a, b)
    intervention = geometry.intervention_inner_product(a, b)
    inverse = torch.linalg.solve(covariance, torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(measurement, torch.einsum("...i,ij,...j->...", a, inverse, b))
    torch.testing.assert_close(intervention, torch.einsum("...i,ij,...j->...", a, covariance, b))
    assert measurement.shape == (3, 2)
    torch.testing.assert_close(measurement[-1], torch.zeros(2, dtype=torch.float64))
    scalar = geometry.measurement_inner_product(a[0, 0], b[0, 0])
    assert scalar.shape == torch.Size([])


def test_derived_intervention_is_the_metric_identification_not_observed_evidence(geometry_contract):
    geometry, covariance, _, _ = geometry_contract
    measurements = torch.tensor([[2.0, -1.0], [-3.0, 4.0]], dtype=torch.float64)
    derived = geometry.derived_intervention(measurements)
    expected = torch.linalg.solve(covariance, measurements.T).T
    torch.testing.assert_close(derived, expected)
    torch.testing.assert_close(
        geometry.whiten_intervention(derived), geometry.whiten_measurement(measurements)
    )
    torch.testing.assert_close(
        geometry.derived_intervention(torch.zeros(2, dtype=torch.float64)),
        torch.zeros(2, dtype=torch.float64),
    )


@pytest.mark.parametrize("space", ["measurement", "intervention"])
def test_cosines_match_independent_references_and_preserve_sign(geometry_contract, space):
    geometry, covariance, _, _ = geometry_contract
    metric = torch.linalg.inv(covariance) if space == "measurement" else covariance
    cosine = getattr(geometry, f"{space}_cosine")
    a = torch.tensor([2.0, -1.0], dtype=torch.float64)
    b = torch.tensor([[3.0, 4.0], [2.0, -1.0], [-2.0, 1.0]], dtype=torch.float64)
    expected = (b @ metric @ a) / torch.sqrt(
        (a @ metric @ a) * torch.einsum("bi,ij,bj->b", b, metric, b)
    )
    torch.testing.assert_close(cosine(a, b), expected)
    assert cosine(a, a).item() == pytest.approx(1.0)
    assert cosine(a, -a).item() == pytest.approx(-1.0)


@pytest.mark.parametrize("space", ["measurement", "intervention"])
def test_cosine_is_stable_under_large_and_small_positive_rescaling(geometry_contract, space):
    geometry, _, _, _ = geometry_contract
    cosine = getattr(geometry, f"{space}_cosine")
    a = torch.tensor([2.0, -1.0], dtype=torch.float64)
    b = torch.tensor([3.0, 4.0], dtype=torch.float64)
    expected = cosine(a, b)
    torch.testing.assert_close(cosine(a * 1e300, b * 1e-300), expected)
    torch.testing.assert_close(cosine(a * 1e-300, b * 1e300), expected)


@pytest.mark.parametrize("space", ["measurement", "intervention"])
def test_cosine_reports_orthogonal_vectors(geometry_contract, space):
    geometry, _, whitening, unwhitening = geometry_contract
    raw = unwhitening if space == "measurement" else whitening
    cosine = getattr(geometry, f"{space}_cosine")
    assert abs(cosine(raw[0], raw[1]).item()) < 1e-14


@pytest.mark.parametrize("space", ["measurement", "intervention"])
def test_zero_direction_cosine_is_rejected_without_epsilon_or_silent_zero(geometry_contract, space):
    geometry, _, _, _ = geometry_contract
    cosine = getattr(geometry, f"{space}_cosine")
    zero = torch.zeros(2, dtype=torch.float64)
    one = torch.ones(2, dtype=torch.float64)
    for a, b in [(zero, one), (one, zero), (zero, zero), (torch.stack([one, zero]), one)]:
        with pytest.raises(ValueError, match="zero"):
            cosine(a, b)


@pytest.mark.parametrize(
    "method",
    [
        "whiten_measurement",
        "unwhiten_measurement",
        "whiten_intervention",
        "unwhiten_intervention",
        "derived_intervention",
    ],
)
@pytest.mark.parametrize(
    "vector", [torch.tensor(1.0), torch.ones(3), torch.empty(0, 2), torch.ones(2, 3)]
)
def test_transform_rejects_invalid_feature_shapes(method, vector):
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    with pytest.raises(ValueError, match="trailing dimension|non-empty"):
        getattr(geometry, method)(vector)


@pytest.mark.parametrize(
    "vector",
    [
        [[1.0, 2.0]],
        torch.ones(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.complex64),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([float("inf"), 1.0]),
    ],
)
def test_transform_rejects_invalid_types_and_non_finite_values(vector):
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        geometry.whiten_measurement(vector)


def test_transform_rejects_device_mismatch_and_sparse_layout():
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    with pytest.raises(ValueError, match="same device"):
        geometry.whiten_measurement(torch.empty(2, device="meta"))
    with pytest.raises(ValueError, match="strided"):
        geometry.whiten_measurement(torch.ones(2).to_sparse())


@pytest.mark.parametrize(
    "method",
    [
        "measurement_inner_product",
        "intervention_inner_product",
        "measurement_cosine",
        "intervention_cosine",
    ],
)
def test_binary_operations_reject_nonbroadcastable_batches(method):
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    with pytest.raises(ValueError, match="broadcast"):
        getattr(geometry, method)(torch.ones(3, 2), torch.ones(4, 2))
    with pytest.raises(ValueError, match="trailing dimension"):
        getattr(geometry, method)(torch.ones(2), torch.ones(3))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_operations_use_fit_precision_and_preserve_noncontiguous_shapes(dtype):
    readout, _ = known_readout(torch.float64)
    geometry = RepresentationGeometry(readout)
    vector = torch.tensor([[2.0, 4.0, -1.0], [1.0, -2.0, 3.0]], dtype=dtype).T
    assert not vector.is_contiguous()
    result = geometry.whiten_measurement(vector)
    assert result.dtype == torch.float64
    assert result.device == vector.device
    assert result.shape == (3, 2)
    torch.testing.assert_close(geometry.unwhiten_measurement(result), vector.to(torch.float64))


def test_operation_overflow_and_unrepresentable_cast_are_rejected():
    geometry = RepresentationGeometry(torch.tensor([[-0.1, 0.1]]))
    with pytest.raises(ValueError, match="finite"):
        geometry.whiten_measurement(torch.tensor([1e38]))
    with pytest.raises(ValueError, match="finite"):
        geometry.whiten_measurement(torch.tensor([1e100], dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        geometry.measurement_inner_product(torch.tensor([1e20]), torch.tensor([1e20]))


def test_operations_preserve_vector_gradients_without_refitting_weights(geometry_contract):
    geometry, _, _, _ = geometry_contract
    vector = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    for operation in [
        geometry.whiten_measurement,
        geometry.unwhiten_measurement,
        geometry.whiten_intervention,
        geometry.unwhiten_intervention,
        geometry.derived_intervention,
    ]:
        assert torch.autograd.gradcheck(operation, (vector,))
    other = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    for operation in [
        geometry.measurement_inner_product,
        geometry.intervention_inner_product,
        geometry.measurement_cosine,
        geometry.intervention_cosine,
    ]:
        assert torch.autograd.gradcheck(operation, (vector, other))


def test_vector_gradients_do_not_backpropagate_into_fitted_weights():
    readout, _ = known_readout()
    readout.requires_grad_()
    geometry = RepresentationGeometry(readout)
    vector = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    geometry.whiten_measurement(vector).sum().backward()
    assert readout.grad is None
    assert vector.grad is not None
    assert torch.isfinite(vector.grad).all()


def test_operations_convert_vectors_to_the_declared_fit_precision():
    geometry = RepresentationGeometry(torch.tensor([[-2.0, 2.0]]))
    result = geometry.whiten_measurement(torch.tensor([2.0], dtype=torch.float64))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([1.0]))
    with pytest.raises(ValueError, match="zero"):
        geometry.measurement_cosine(torch.tensor([1e-100], dtype=torch.float64), torch.ones(1))


def test_geometry_snapshot_and_matrix_accessors_do_not_mutate_the_fit():
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    vector = torch.tensor([2.0, -1.0], dtype=torch.float64)
    expected = geometry.whiten_measurement(vector)
    covariance = geometry.covariance
    regularized = geometry.regularized_covariance
    mean = geometry.mean
    readout.fill_(0.0)
    covariance.fill_(0.0)
    regularized.fill_(0.0)
    mean.fill_(0.0)
    torch.testing.assert_close(geometry.whiten_measurement(vector), expected)
    assert geometry.diagnostics.input_shape == (2, 4)
    assert torch.count_nonzero(geometry.covariance) == 4


def test_constructor_passes_fit_options_through_to_the_snapshot():
    readout, _ = known_readout(torch.float32)
    geometry = RepresentationGeometry(
        readout, token_ids=[0, 2, 3], ridge=0.25, rtol=1e-8, compute_dtype=torch.float64
    )
    fit = _fit_unembedding_geometry(
        readout, token_ids=[0, 2, 3], ridge=0.25, rtol=1e-8, compute_dtype=torch.float64
    )
    torch.testing.assert_close(geometry.covariance, fit.covariance)
    assert geometry.diagnostics == fit.diagnostics


def test_exact_metrics_and_derived_map_are_invariant_under_general_basis_changes():
    readout, _ = known_readout()
    change = torch.tensor([[2.0, 1.0], [-0.3, 1.4]], dtype=torch.float64)
    inverse = torch.linalg.inv(change)
    original = RepresentationGeometry(readout)
    transformed = RepresentationGeometry(change @ readout)
    g = torch.tensor([2.0, -1.0], dtype=torch.float64)
    k = torch.tensor([-3.0, 4.0], dtype=torch.float64)
    h = torch.tensor([1.0, 3.0], dtype=torch.float64)
    l = torch.tensor([4.0, -2.0], dtype=torch.float64)
    torch.testing.assert_close(
        original.measurement_inner_product(g, k),
        transformed.measurement_inner_product(g @ change.T, k @ change.T),
    )
    torch.testing.assert_close(
        original.intervention_inner_product(h, l),
        transformed.intervention_inner_product(h @ inverse, l @ inverse),
    )
    torch.testing.assert_close(
        original.derived_intervention(g) @ inverse, transformed.derived_intervention(g @ change.T)
    )
    torch.testing.assert_close(
        original.measurement_cosine(g, k),
        transformed.measurement_cosine(g @ change.T, k @ change.T),
    )
    torch.testing.assert_close(
        original.intervention_cosine(h, l),
        transformed.intervention_cosine(h @ inverse, l @ inverse),
    )


def test_isotropic_ridge_is_not_invariant_under_general_basis_changes():
    readout, _ = known_readout()
    change = torch.tensor([[2.0, 1.0], [-0.3, 1.4]], dtype=torch.float64)
    original = RepresentationGeometry(readout, ridge=0.5)
    transformed = RepresentationGeometry(change @ readout, ridge=0.5)
    g = torch.tensor([2.0, -1.0], dtype=torch.float64)
    k = torch.tensor([-3.0, 4.0], dtype=torch.float64)
    assert not torch.allclose(
        original.measurement_inner_product(g, k),
        transformed.measurement_inner_product(g @ change.T, k @ change.T),
    )


def test_isotropic_ridge_preserves_orthogonal_basis_invariance():
    readout, rotation = known_readout()
    original = RepresentationGeometry(readout, ridge=0.5)
    transformed = RepresentationGeometry(rotation @ readout, ridge=0.5)
    g = torch.tensor([2.0, -1.0], dtype=torch.float64)
    k = torch.tensor([-3.0, 4.0], dtype=torch.float64)
    torch.testing.assert_close(
        original.measurement_inner_product(g, k),
        transformed.measurement_inner_product(g @ rotation.T, k @ rotation.T),
    )
