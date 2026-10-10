"""Analytic tests for dual-space concept and categorical geometry."""

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from tests.typecheck_errors import TYPECHECK_ERRORS
from transformer_lens.tools.analysis.representation_geometry import (
    CategoricalGeometry,
    ConceptDirection,
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


@pytest.mark.parametrize("ridge", [None, 0.5])
def test_concept_direction_matches_independent_oriented_pair_aggregation(ridge):
    readout, rotation = known_readout()
    geometry = RepresentationGeometry(readout, ridge=ridge)
    result = geometry.concept_direction(
        [(1, 0), (3, 2)], label="contrast", pair_labels=[("low-a", "high-a"), ("low-b", "high-b")]
    )
    differences = torch.stack([readout[:, 0] - readout[:, 1], readout[:, 2] - readout[:, 3]])
    mean = differences.mean(dim=0)
    eigenvalues = torch.tensor([4.0, 1.0], dtype=torch.float64) + (ridge or 0.0)
    covariance = rotation @ torch.diag(eigenvalues) @ rotation.T
    whitening = rotation @ torch.diag(eigenvalues.rsqrt()) @ rotation.T
    intervention = torch.linalg.solve(covariance, mean)
    deviations = differences - mean
    assert isinstance(result, ConceptDirection)
    torch.testing.assert_close(result.raw_pair_differences, differences)
    torch.testing.assert_close(result.whitened_pair_differences, differences @ whitening.T)
    torch.testing.assert_close(result.raw_measurement, mean)
    torch.testing.assert_close(result.whitened_measurement, whitening @ mean)
    torch.testing.assert_close(result.raw_derived_intervention, intervention)
    torch.testing.assert_close(result.whitened_derived_intervention, whitening @ mean)
    torch.testing.assert_close(result.raw_dispersion, deviations.square().sum(-1).mean())
    torch.testing.assert_close(
        result.whitened_dispersion, (deviations @ whitening.T).square().sum(-1).mean()
    )
    assert result.pairs == ((1, 0), (3, 2))
    assert result.pair_labels == (("low-a", "high-a"), ("low-b", "high-b"))
    assert result.label == "contrast"
    assert result.n_pairs == 2
    assert result.aggregation == "mean"
    assert result.intervention_kind == "metric-derived"
    assert result.geometry_diagnostics == geometry.diagnostics
    assert result.raw_measurement.shape == (2,)
    assert result.raw_dispersion.shape == torch.Size([])


@pytest.mark.parametrize("ridge", [None, 0.5])
def test_reversing_all_pairs_reverses_directions_but_not_dispersion(ridge):
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout, ridge=ridge)
    original = geometry.concept_direction([(1, 0), (3, 2)])
    reversed_result = geometry.concept_direction([(0, 1), (2, 3)])
    for name in [
        "raw_pair_differences",
        "whitened_pair_differences",
        "raw_measurement",
        "whitened_measurement",
        "raw_derived_intervention",
        "whitened_derived_intervention",
    ]:
        torch.testing.assert_close(getattr(reversed_result, name), -getattr(original, name))
    torch.testing.assert_close(reversed_result.raw_dispersion, original.raw_dispersion)
    torch.testing.assert_close(reversed_result.whitened_dispersion, original.whitened_dispersion)


def test_single_pair_dispersion_is_population_zero_without_a_confidence_claim():
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    result = geometry.concept_direction([(1, 0)])
    torch.testing.assert_close(result.raw_measurement, readout[:, 0] - readout[:, 1])
    assert result.raw_dispersion.item() == 0.0
    assert result.whitened_dispersion.item() == 0.0
    assert result.n_pairs == 1
    assert result.label is None
    assert result.pair_labels is None


def test_concept_dispersion_is_population_variance_not_sample_variance():
    readout, _ = known_readout()
    result = RepresentationGeometry(readout).concept_direction([(1, 0), (3, 2)])
    sample = result.raw_pair_differences.var(dim=0, correction=1).sum()
    torch.testing.assert_close(result.raw_dispersion * 2, sample)
    assert not torch.allclose(result.raw_dispersion, sample)


def test_contrasts_do_not_subtract_the_token_mean_twice():
    readout, _ = known_readout()
    shifted = readout + torch.tensor([[13.0], [-11.0]], dtype=torch.float64)
    original = RepresentationGeometry(readout).concept_direction([(1, 0), (3, 2)])
    translated = RepresentationGeometry(shifted).concept_direction([(1, 0), (3, 2)])
    torch.testing.assert_close(original.raw_measurement, translated.raw_measurement)
    torch.testing.assert_close(original.whitened_measurement, translated.whitened_measurement)
    torch.testing.assert_close(original.raw_dispersion, translated.raw_dispersion)


def test_pair_order_is_preserved_but_does_not_change_the_mean():
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    original = geometry.concept_direction([(1, 0), (3, 2)])
    reordered = geometry.concept_direction([(3, 2), (1, 0)])
    assert reordered.pairs == ((3, 2), (1, 0))
    torch.testing.assert_close(reordered.raw_measurement, original.raw_measurement)
    torch.testing.assert_close(reordered.whitened_dispersion, original.whitened_dispersion)


def test_population_selection_does_not_limit_valid_contrast_token_ids():
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout, token_ids=[0, 2, 3])
    result = geometry.concept_direction([(1, 0)])
    assert result.geometry_diagnostics.token_ids == (0, 2, 3)
    torch.testing.assert_close(result.raw_measurement, readout[:, 0] - readout[:, 1])


@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [(0, 0)],
        [(0, 1), (0, 1)],
        [(0, 1), (1, 0)],
        [(-1, 0)],
        [(0, 4)],
        [(True, 0)],
        [(0.5, 1)],
        [(0,)],
        [(0, 1, 2)],
        ["01"],
        [("low", "high")],
        None,
        "01",
    ],
)
def test_invalid_contrast_pairs_are_rejected(pairs):
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        geometry.concept_direction(pairs)


def test_distinct_token_ids_with_identical_readout_vectors_are_rejected():
    readout = torch.tensor([[0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 2.0, -2.0]])
    geometry = RepresentationGeometry(readout, ridge=0.25)
    with pytest.raises(ValueError, match="zero"):
        geometry.concept_direction([(0, 1), (0, 2)])


def test_canceling_distinct_contrasts_have_no_mean_concept_direction():
    readout = torch.tensor([[0.0, 1.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    geometry = RepresentationGeometry(readout)
    with pytest.raises(ValueError, match="mean.*zero"):
        geometry.concept_direction([(0, 1), (2, 3)])


def test_noisy_contrasts_are_reported_without_a_minimum_pair_count():
    readout = torch.tensor([[0.0, 1.0, 2.0, 1.1], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    geometry = RepresentationGeometry(readout)
    result = geometry.concept_direction([(0, 1), (2, 3)])
    torch.testing.assert_close(
        result.raw_measurement, torch.tensor([0.05, 0.0], dtype=torch.float64)
    )
    assert result.raw_dispersion.item() > 0.9
    assert result.n_pairs == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": ""},
        {"label": True},
        {"pair_labels": []},
        {"pair_labels": [("low", "high"), ("extra", "pair")]},
        {"pair_labels": [("", "high")]},
        {"pair_labels": [("low", 1)]},
        {"pair_labels": ["ab"]},
        {"pair_labels": "ab"},
    ],
)
def test_invalid_concept_labels_are_rejected(kwargs):
    readout, _ = known_readout()
    geometry = RepresentationGeometry(readout)
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        geometry.concept_direction([(1, 0)], **kwargs)


def test_concept_metadata_is_frozen_and_input_sequences_are_snapshotted():
    readout, _ = known_readout()
    pairs = [[1, 0], [3, 2]]
    labels = [["a", "b"], ["c", "d"]]
    result = RepresentationGeometry(readout).concept_direction(pairs, pair_labels=labels)
    pairs[0][0] = 0
    labels[0][0] = "changed"
    assert result.pairs == ((1, 0), (3, 2))
    assert result.pair_labels == (("a", "b"), ("c", "d"))
    with pytest.raises(FrozenInstanceError):
        result.label = "changed"


def test_concept_results_are_detached_and_cannot_mutate_geometry_or_other_results():
    readout, _ = known_readout()
    readout.requires_grad_()
    geometry = RepresentationGeometry(readout)
    before = geometry.concept_direction([(1, 0), (3, 2)])
    with torch.no_grad():
        readout.fill_(0.0)
    after = geometry.concept_direction([(1, 0), (3, 2)])
    for name in [
        "raw_measurement",
        "whitened_measurement",
        "raw_derived_intervention",
        "whitened_derived_intervention",
        "raw_pair_differences",
        "whitened_pair_differences",
        "raw_dispersion",
        "whitened_dispersion",
    ]:
        actual = getattr(after, name)
        expected = getattr(before, name)
        torch.testing.assert_close(actual, expected)
        assert not actual.requires_grad
        assert actual.grad_fn is None
        actual.fill_(0.0)
    fresh = geometry.concept_direction([(1, 0), (3, 2)])
    torch.testing.assert_close(fresh.raw_measurement, before.raw_measurement)
    torch.testing.assert_close(fresh.raw_pair_differences, before.raw_pair_differences)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_concept_results_use_the_fit_precision_and_device(dtype):
    readout, _ = known_readout(dtype)
    geometry = RepresentationGeometry(readout)
    result = geometry.concept_direction([(1, 0), (3, 2)])
    for name in [
        "raw_measurement",
        "whitened_measurement",
        "raw_derived_intervention",
        "whitened_derived_intervention",
        "raw_pair_differences",
        "whitened_pair_differences",
        "raw_dispersion",
        "whitened_dispersion",
    ]:
        value = getattr(result, name)
        assert value.dtype == geometry.diagnostics.compute_dtype
        assert value.device == geometry.diagnostics.device


def test_concept_metric_dispersion_and_derived_map_respect_exact_basis_changes():
    readout, _ = known_readout()
    change = torch.tensor([[2.0, 1.0], [-0.3, 1.4]], dtype=torch.float64)
    original = RepresentationGeometry(readout).concept_direction([(1, 0), (3, 2)])
    transformed = RepresentationGeometry(change @ readout).concept_direction([(1, 0), (3, 2)])
    torch.testing.assert_close(transformed.raw_measurement, original.raw_measurement @ change.T)
    torch.testing.assert_close(
        transformed.raw_derived_intervention,
        original.raw_derived_intervention @ torch.linalg.inv(change),
    )
    torch.testing.assert_close(transformed.whitened_dispersion, original.whitened_dispersion)
    assert not torch.allclose(transformed.raw_dispersion, original.raw_dispersion)


def test_non_finite_contrast_difference_is_rejected_at_compute_precision():
    readout = torch.tensor([[0.0, 1.0, 1e100, -1e100]], dtype=torch.float64)
    geometry = RepresentationGeometry(readout, token_ids=[0, 1], compute_dtype=torch.float32)
    with pytest.raises(ValueError, match="finite"):
        geometry.concept_direction([(2, 3)])


def test_population_dispersion_overflow_is_rejected_not_reported_as_infinity():
    readout = torch.tensor([[0.0, 1.0, 1e200, -1e200]], dtype=torch.float64)
    geometry = RepresentationGeometry(readout, token_ids=[0, 1])
    with pytest.raises(ValueError, match="finite"):
        geometry.concept_direction([(0, 2), (0, 3), (1, 2)])


def isotropic_geometry(dimension=2, dtype=torch.float64):
    readout = math.sqrt(dimension) * torch.cat(
        [torch.eye(dimension, dtype=dtype), -torch.eye(dimension, dtype=dtype)], dim=1
    )
    return RepresentationGeometry(readout)


def regular_triangle(dtype=torch.float64):
    return torch.tensor(
        [[1.0, 0.0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]], dtype=dtype
    )


@pytest.mark.parametrize("space", ["measurement", "intervention"])
def test_constructed_regular_simplex_matches_independent_geometry(geometry_contract, space):
    geometry, _, whitening, unwhitening = geometry_contract
    expected = regular_triangle()
    raw_map = unwhitening if space == "measurement" else whitening
    vertices = expected @ raw_map.T
    result = geometry.categorical_geometry(vertices, space=space, labels=["a", "b", "c"])
    assert isinstance(result, CategoricalGeometry)
    assert result.space == space
    assert result.labels == ("a", "b", "c")
    assert result.n_vertices == 3
    assert result.affine_rank == 2
    assert result.is_simplex()
    assert result.is_regular_simplex()
    assert result.relative_distance_spread < 1e-14
    torch.testing.assert_close(result.whitened_vertices, expected)
    torch.testing.assert_close(result.gram, expected @ expected.T)
    distances = math.sqrt(3) * (
        torch.ones(3, 3, dtype=torch.float64) - torch.eye(3, dtype=torch.float64)
    )
    torch.testing.assert_close(result.distances, distances)
    torch.testing.assert_close(result.cosines, expected @ expected.T)
    angles = (2 * math.pi / 3) * (
        torch.ones(3, 3, dtype=torch.float64) - torch.eye(3, dtype=torch.float64)
    )
    torch.testing.assert_close(result.angles, angles)
    torch.testing.assert_close(
        result.singular_values, torch.full((2,), math.sqrt(1.5), dtype=torch.float64)
    )
    assert result.angle_valid_mask.all()
    assert result.geometry_diagnostics == geometry.diagnostics


@pytest.mark.parametrize("space", ["measurement", "intervention"])
def test_category_gram_and_distances_match_independent_metric_references(geometry_contract, space):
    geometry, covariance, _, _ = geometry_contract
    vertices = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    result = geometry.categorical_geometry(vertices, space=space)
    centered = vertices - vertices.mean(dim=0)
    metric = torch.linalg.inv(covariance) if space == "measurement" else covariance
    expected_gram = centered @ metric @ centered.T
    delta = vertices[:, None, :] - vertices[None, :, :]
    expected_distances = torch.einsum("...i,ij,...j->...", delta, metric, delta).sqrt()
    torch.testing.assert_close(result.centroid, vertices.mean(dim=0))
    torch.testing.assert_close(result.centered_raw_vertices, centered)
    torch.testing.assert_close(result.gram, expected_gram)
    torch.testing.assert_close(result.distances, expected_distances)
    norms = expected_gram.diagonal().sqrt()
    torch.testing.assert_close(result.cosines, expected_gram / (norms[:, None] * norms[None, :]))
    assert result.is_simplex()


def test_non_regular_simplex_is_not_confused_with_regular_simplex():
    vertices = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    result = isotropic_geometry().categorical_geometry(vertices, space="measurement")
    assert result.affine_rank == 2
    assert result.is_simplex()
    assert not result.is_regular_simplex()
    expected_spread = (math.sqrt(5) - 1) / ((1 + 2 + math.sqrt(5)) / 3)
    assert result.relative_distance_spread == pytest.approx(expected_spread)


def test_collinear_categories_report_undefined_centroid_angles_with_a_mask():
    vertices = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    result = isotropic_geometry().categorical_geometry(vertices, space="measurement")
    assert result.affine_rank == 1
    assert not result.is_simplex()
    assert not result.is_regular_simplex()
    expected_mask = torch.tensor([[True, False, True], [False, False, False], [True, False, True]])
    torch.testing.assert_close(result.angle_valid_mask, expected_mask)
    assert torch.isnan(result.cosines[~expected_mask]).all()
    assert torch.isnan(result.angles[~expected_mask]).all()
    assert result.angles[0, 2].item() == pytest.approx(math.pi)
    torch.testing.assert_close(
        result.distances,
        torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], dtype=torch.float64),
    )


def test_duplicate_vertices_are_diagnosed_without_being_silently_dropped():
    vertices = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    result = isotropic_geometry().categorical_geometry(vertices, space="measurement")
    assert result.n_vertices == 3
    assert result.affine_rank == 1
    assert not result.is_simplex()
    assert not result.is_regular_simplex()
    assert result.distances[0, 1].item() == 0.0
    assert result.cosines[0, 1].item() == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_coincident_vertices_have_zero_rank_and_no_defined_angles(dtype):
    geometry = isotropic_geometry(dtype=dtype)
    vertices = torch.full((3, 2), 1e38, dtype=dtype)
    result = geometry.categorical_geometry(vertices, space="measurement")
    assert result.affine_rank == 0
    assert not result.is_simplex()
    assert not result.is_regular_simplex()
    assert result.relative_distance_spread is None
    assert not result.angle_valid_mask.any()
    assert torch.isnan(result.angles).all()
    assert torch.isnan(result.cosines).all()
    torch.testing.assert_close(result.gram, torch.zeros(3, 3, dtype=dtype))
    torch.testing.assert_close(result.distances, torch.zeros(3, 3, dtype=dtype))
    torch.testing.assert_close(result.centroid, vertices[0])


def test_invalid_angle_rows_are_masked_before_cosine_range_validation():
    vertices = torch.tensor(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, -0.9], [0.1, 0.1, 0.1], [-0.1, -0.1, -0.2]],
        dtype=torch.float64,
    )
    result = isotropic_geometry(3).categorical_geometry(vertices, space="measurement", rtol=0.99)
    assert result.angle_valid_mask[0, 0]
    assert not result.angle_valid_mask[1].any()
    assert torch.isnan(result.angles[1]).all()


def test_too_many_vertices_for_the_ambient_space_cannot_form_a_simplex():
    vertices = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    result = isotropic_geometry().categorical_geometry(vertices, space="measurement")
    assert result.affine_rank == 2
    assert result.n_vertices == 4
    assert not result.is_simplex()
    assert not result.is_regular_simplex()


def test_two_distinct_vertices_form_a_regular_line_segment():
    result = isotropic_geometry(1).categorical_geometry(
        torch.tensor([[0.0], [3.0]], dtype=torch.float64), space="intervention"
    )
    assert result.affine_rank == 1
    assert result.is_simplex()
    assert result.is_regular_simplex(rtol=0.0)
    assert result.relative_distance_spread == 0.0


def test_near_zero_categories_use_declared_rank_and_angle_thresholds():
    vertices = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1e-12]], dtype=torch.float64)
    geometry = isotropic_geometry()
    strict = geometry.categorical_geometry(vertices, space="measurement", rtol=1e-14)
    loose = geometry.categorical_geometry(vertices, space="measurement", rtol=1e-6)
    assert strict.affine_rank == 2
    assert strict.is_simplex()
    assert strict.angle_valid_mask.all()
    assert loose.affine_rank == 1
    assert not loose.is_simplex()
    assert not loose.angle_valid_mask[2].any()
    assert torch.isnan(loose.angles[2]).all()
    assert loose.rank_rtol == 1e-6
    assert loose.rank_threshold == pytest.approx(float(loose.singular_values[0]) * 1e-6)


@pytest.mark.parametrize("scale", [1e-6, -3.0, 1e6])
def test_category_rank_angles_and_relative_regularity_are_scale_invariant(scale):
    vertices = regular_triangle()
    geometry = isotropic_geometry()
    original = geometry.categorical_geometry(vertices, space="measurement")
    scaled = geometry.categorical_geometry(vertices * scale, space="measurement")
    assert scaled.affine_rank == original.affine_rank
    assert scaled.is_regular_simplex()
    torch.testing.assert_close(scaled.gram, original.gram * scale**2)
    torch.testing.assert_close(scaled.distances, original.distances * abs(scale))
    torch.testing.assert_close(scaled.cosines, original.cosines)
    torch.testing.assert_close(scaled.angles, original.angles)
    torch.testing.assert_close(scaled.singular_values, original.singular_values * abs(scale))


def test_category_diagnostics_are_translation_invariant():
    vertices = regular_triangle()
    geometry = isotropic_geometry()
    original = geometry.categorical_geometry(vertices, space="measurement")
    translated = geometry.categorical_geometry(
        vertices + torch.tensor([13.0, -7.0]), space="measurement"
    )
    torch.testing.assert_close(translated.centered_raw_vertices, original.centered_raw_vertices)
    torch.testing.assert_close(translated.gram, original.gram)
    torch.testing.assert_close(translated.distances, original.distances)
    torch.testing.assert_close(translated.angles, original.angles)
    assert translated.is_regular_simplex()


@pytest.mark.parametrize("space", ["measurement", "intervention"])
def test_exact_categorical_reports_respect_consistent_dual_basis_changes(space):
    readout, _ = known_readout()
    change = torch.tensor([[2.0, 1.0], [-0.3, 1.4]], dtype=torch.float64)
    vertices = regular_triangle()
    transformed_vertices = vertices @ (
        change.T if space == "measurement" else torch.linalg.inv(change)
    )
    original = RepresentationGeometry(readout).categorical_geometry(vertices, space=space)
    transformed = RepresentationGeometry(change @ readout).categorical_geometry(
        transformed_vertices, space=space
    )
    torch.testing.assert_close(transformed.gram, original.gram)
    torch.testing.assert_close(transformed.distances, original.distances)
    torch.testing.assert_close(transformed.cosines, original.cosines)
    torch.testing.assert_close(transformed.angles, original.angles)
    assert transformed.affine_rank == original.affine_rank


def test_vertex_order_and_labels_are_preserved_with_permuted_reports():
    vertices = regular_triangle()
    order = [2, 0, 1]
    geometry = isotropic_geometry()
    original = geometry.categorical_geometry(vertices, space="measurement", labels=["a", "b", "c"])
    reordered = geometry.categorical_geometry(
        vertices[order], space="measurement", labels=["c", "a", "b"]
    )
    assert reordered.labels == ("c", "a", "b")
    torch.testing.assert_close(reordered.gram, original.gram[order][:, order])
    torch.testing.assert_close(reordered.distances, original.distances[order][:, order])
    assert reordered.affine_rank == original.affine_rank


def test_regularity_tolerance_is_relative_explicit_and_separate_from_rank():
    vertices = regular_triangle() * torch.tensor([1.02, 1.0])
    result = isotropic_geometry().categorical_geometry(vertices, space="measurement")
    assert result.is_simplex()
    assert not result.is_regular_simplex(rtol=1e-3)
    assert result.is_regular_simplex(rtol=0.05)


@pytest.mark.parametrize("rtol", [True, -1.0, 1.0, float("nan"), float("inf")])
def test_invalid_category_tolerances_are_rejected_even_for_degenerate_data(rtol):
    geometry = isotropic_geometry()
    vertices = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        geometry.categorical_geometry(vertices, space="measurement", rtol=rtol)
    result = geometry.categorical_geometry(vertices, space="measurement")
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        result.is_regular_simplex(rtol=rtol)


@pytest.mark.parametrize(
    "vertices",
    [torch.ones(2), torch.ones(2, 2, 2), torch.empty(0, 2), torch.ones(1, 2), torch.ones(3, 1)],
)
def test_invalid_category_shapes_and_insufficient_vertices_are_rejected(vertices):
    with pytest.raises(
        ValueError, match="two-dimensional|non-empty|at least two|trailing dimension"
    ):
        isotropic_geometry().categorical_geometry(vertices, space="measurement")


@pytest.mark.parametrize(
    "vertices",
    [
        [[1.0, 2.0], [2.0, 3.0]],
        torch.ones(3, 2, dtype=torch.int64),
        torch.ones(3, 2, dtype=torch.complex64),
        torch.tensor([[float("nan"), 1.0], [2.0, 3.0]]),
        torch.tensor([[float("inf"), 1.0], [2.0, 3.0]]),
    ],
)
def test_invalid_category_types_and_non_finite_data_are_rejected(vertices):
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        isotropic_geometry().categorical_geometry(vertices, space="measurement")


@pytest.mark.parametrize("space", ["raw", "whitened", None])
def test_undeclared_or_incompatible_category_spaces_are_rejected(space):
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        isotropic_geometry().categorical_geometry(regular_triangle(), space=space)


def test_category_space_must_be_supplied_explicitly():
    with pytest.raises(TypeError):
        isotropic_geometry().categorical_geometry(regular_triangle())


@pytest.mark.parametrize(
    "labels", [[], ["a", "b"], ["a", "a", "c"], ["a", "", "c"], ["a", 1, "c"], "abc"]
)
def test_invalid_category_labels_are_rejected(labels):
    with pytest.raises((ValueError,) + TYPECHECK_ERRORS):
        isotropic_geometry().categorical_geometry(
            regular_triangle(), space="measurement", labels=labels
        )


def test_category_sparse_layout_and_device_mismatch_are_rejected():
    geometry = isotropic_geometry()
    with pytest.raises(ValueError, match="strided"):
        geometry.categorical_geometry(regular_triangle().to_sparse(), space="measurement")
    with pytest.raises(ValueError, match="same device"):
        geometry.categorical_geometry(torch.empty(3, 2, device="meta"), space="measurement")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_category_noncontiguous_inputs_use_fit_precision_and_device(dtype):
    geometry = isotropic_geometry(dtype=dtype)
    vertices = regular_triangle(dtype).T.contiguous().T
    assert not vertices.is_contiguous()
    result = geometry.categorical_geometry(vertices, space="measurement")
    for name in [
        "raw_vertices",
        "centroid",
        "centered_raw_vertices",
        "whitened_vertices",
        "singular_values",
        "gram",
        "distances",
        "cosines",
        "angles",
    ]:
        value = getattr(result, name)
        assert value.dtype == geometry.diagnostics.compute_dtype
        assert value.device == geometry.diagnostics.device
    assert result.angle_valid_mask.dtype == torch.bool
    assert result.rank_rtol == pytest.approx(
        3 * torch.finfo(geometry.diagnostics.compute_dtype).eps
    )


def test_category_report_metadata_and_tensors_are_independent_snapshots():
    geometry = isotropic_geometry()
    vertices = regular_triangle().requires_grad_()
    before = vertices.detach().clone()
    labels = ["a", "b", "c"]
    result = geometry.categorical_geometry(vertices, space="measurement", labels=labels)
    labels[0] = "changed"
    with torch.no_grad():
        vertices.fill_(0.0)
    torch.testing.assert_close(result.raw_vertices, before)
    assert result.labels == ("a", "b", "c")
    with pytest.raises(FrozenInstanceError):
        result.space = "intervention"
    for name in [
        "raw_vertices",
        "centroid",
        "centered_raw_vertices",
        "whitened_vertices",
        "singular_values",
        "gram",
        "distances",
        "cosines",
        "angles",
    ]:
        value = getattr(result, name)
        assert not value.requires_grad
        assert value.grad_fn is None
        value.fill_(0.0)
    fresh = geometry.categorical_geometry(before, space="measurement")
    assert fresh.is_regular_simplex()
    torch.testing.assert_close(fresh.raw_vertices, before)


def test_transform_underflow_is_not_misreported_as_coincident_categories():
    geometry = RepresentationGeometry(torch.tensor([[-1e150, 1e150]], dtype=torch.float64))
    vertices = torch.tensor([[-1e-200], [1e-200]], dtype=torch.float64)
    with pytest.raises(ValueError, match="transform underflow"):
        geometry.categorical_geometry(vertices, space="measurement")


@pytest.mark.parametrize("scale, message", [(1e200, "finite"), (1e-200, "underflow")])
def test_unrepresentable_category_gram_values_are_rejected(scale, message):
    vertices = torch.tensor([[-scale, 0.0], [scale, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match=message):
        isotropic_geometry().categorical_geometry(vertices, space="measurement")
