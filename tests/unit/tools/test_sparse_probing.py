"""Unit tests for leakage-safe model-free sparse probing."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from tests.typecheck_errors import TYPECHECK_ERRORS
from transformer_lens.tools.analysis import (
    fit_sparse_probe as exported_fit_sparse_probe,
)
from transformer_lens.tools.analysis import (
    sweep_sparse_probe as exported_sweep_sparse_probe,
)
from transformer_lens.tools.analysis.sparse_probing import (
    SparseProbeControl,
    SparseProbeResult,
    SparseProbeSweep,
    _binary_metrics,
    fit_sparse_probe,
    sweep_sparse_probe,
)


def test_public_analysis_exports():
    assert exported_fit_sparse_probe is fit_sparse_probe
    assert exported_sweep_sparse_probe is sweep_sparse_probe


def _planted_data(
    *, n_examples: int = 400, n_features: int = 16, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(n_examples) % 2
    features = torch.randn(n_examples, n_features, generator=generator)
    features[:, 3] += 2.5 * (2 * labels - 1)
    permutation = torch.randperm(n_examples, generator=generator)
    return features[permutation], labels[permutation]


def _balanced_objective_gradient(
    features: torch.Tensor,
    labels: torch.Tensor,
    coefficients: torch.Tensor,
    intercept: torch.Tensor,
    l2_strength: float,
) -> torch.Tensor:
    labels = labels.double()
    count = labels.numel()
    positive_count = labels.sum()
    weights = torch.where(
        labels == 1,
        count / (2 * positive_count),
        count / (2 * (count - positive_count)),
    )
    residual = weights * (torch.sigmoid(features @ coefficients + intercept) - labels) / count
    return torch.cat(
        (features.T @ residual + l2_strength * coefficients, residual.sum().reshape(1))
    )


def _newton_reference(
    features: torch.Tensor,
    labels: torch.Tensor,
    l2_strength: float,
) -> torch.Tensor:
    features = features.double()
    labels = labels.double()
    design = torch.cat((features, torch.ones(features.shape[0], 1, dtype=torch.float64)), dim=1)
    count = labels.numel()
    positive_count = labels.sum()
    weights = torch.where(
        labels == 1,
        count / (2 * positive_count),
        count / (2 * (count - positive_count)),
    )
    penalty = torch.diag(
        torch.tensor([l2_strength] * features.shape[1] + [0.0], dtype=torch.float64)
    )
    parameters = torch.zeros(design.shape[1], dtype=torch.float64)
    for _ in range(100):
        probability = torch.sigmoid(design @ parameters)
        residual = weights * (probability - labels) / count
        gradient = design.T @ residual + penalty @ parameters
        curvature = weights * probability * (1 - probability) / count
        hessian = design.T @ (curvature[:, None] * design) + penalty
        parameters -= torch.linalg.solve(hessian, gradient)
        if float(gradient.abs().max()) < 1e-12:
            break
    return parameters


def test_fit_recovers_exact_train_only_mean_difference_and_planted_feature():
    features, labels = _planted_data()

    result = fit_sparse_probe(features, labels, k=1, positive_label=1, seed=17)

    train_features = features[result.train_indices]
    train_labels = labels[result.train_indices]
    expected_scores = train_features[train_labels == 1].mean(0) - train_features[
        train_labels == 0
    ].mean(0)
    expected_selected = torch.argsort(expected_scores.abs(), descending=True, stable=True)[:1]
    assert isinstance(result, SparseProbeResult)
    assert result.selected_features.tolist() == [3]
    assert torch.equal(result.selected_features, expected_selected)
    assert torch.allclose(result.feature_scores, expected_scores.double(), atol=1e-6)
    assert result.metrics.f1 > 0.98
    assert result.metrics.accuracy > 0.98
    assert result.k == 1
    assert result.max_iter == 200
    assert result.gradient_tolerance == 1e-7


def test_positive_label_controls_score_sign_and_class_metadata():
    features, labels = _planted_data()
    signed_labels = 1 - 2 * labels

    result = fit_sparse_probe(features, signed_labels, k=1, positive_label=-1, seed=11)

    assert result.positive_label == -1
    assert result.negative_label == 1
    assert result.feature_scores[3] > 0


def test_unweighted_classification_policy_is_explicit():
    features, labels = _planted_data(n_examples=100, n_features=6)

    result = fit_sparse_probe(features, labels, k=2, class_weight=None, seed=2)
    sweep = sweep_sparse_probe(features, labels, ks=[1], class_weight=None, seed=2)

    assert result.class_weight is None
    assert sweep.results[0].class_weight is None


def test_standardization_is_train_only_and_heldout_values_do_not_change_selection():
    features, labels = _planted_data(n_examples=200, n_features=8)
    first = fit_sparse_probe(features, labels, k=2, preprocess="standardize", seed=9)
    changed = features.clone()
    changed[first.test_indices] += 10_000 * torch.randn_like(changed[first.test_indices])

    second = fit_sparse_probe(changed, labels, k=2, preprocess="standardize", seed=9)

    selected_train = features[first.train_indices][:, first.selected_features].double()
    expected_mean = selected_train.mean(0)
    expected_scale = selected_train.std(0, correction=0)
    assert torch.equal(first.selected_features, second.selected_features)
    assert torch.equal(first.feature_scores, second.feature_scores)
    assert torch.allclose(first.preprocess_mean, expected_mean)
    assert torch.allclose(first.preprocess_scale, expected_scale)
    assert torch.equal(first.preprocess_mean, second.preprocess_mean)
    assert torch.equal(first.preprocess_scale, second.preprocess_scale)
    assert torch.equal(first.coefficients, second.coefficients)
    assert torch.equal(first.intercept, second.intercept)
    assert first.objective == second.objective


def test_none_preprocessing_has_identity_metadata_and_constant_tie_order():
    features = torch.zeros(20, 5)
    labels = torch.arange(20) % 2

    result = fit_sparse_probe(features, labels, k=3, preprocess="none", seed=1)

    assert result.selected_features.tolist() == [0, 1, 2]
    assert torch.equal(result.preprocess_mean, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(result.preprocess_scale, torch.ones(3, dtype=torch.float64))
    assert result.constant_features.tolist() == [True, True, True]


def test_lbfgs_matches_independent_newton_solution_and_gradient():
    features, labels = _planted_data(n_examples=120, n_features=4, seed=4)
    l2_strength = 0.02
    result = fit_sparse_probe(
        features,
        labels,
        k=4,
        l2_strength=l2_strength,
        gradient_tolerance=1e-7,
        seed=3,
    )
    train_features = features[result.train_indices][:, result.selected_features].double()
    train_labels = labels[result.train_indices]
    expected = _newton_reference(train_features, train_labels, l2_strength)
    gradient = _balanced_objective_gradient(
        train_features,
        train_labels,
        result.coefficients,
        result.intercept,
        l2_strength,
    )

    assert torch.allclose(result.coefficients, expected[:-1], atol=2e-6, rtol=2e-6)
    assert result.intercept.item() == pytest.approx(expected[-1].item(), abs=2e-6)
    assert float(gradient.abs().max()) == pytest.approx(result.gradient_inf_norm, abs=1e-12)
    assert result.gradient_inf_norm <= 1e-7


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_supported_source_dtypes_return_detached_cpu_float64_results(dtype):
    features, labels = _planted_data(n_examples=80, n_features=6)

    result = fit_sparse_probe(features.to(dtype), labels, k=2, seed=0)

    for tensor in (
        result.feature_scores,
        result.coefficients,
        result.intercept,
        result.preprocess_mean,
        result.preprocess_scale,
    ):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert not tensor.requires_grad


@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.uint8, torch.int64])
def test_supported_label_dtypes(dtype):
    features, labels = _planted_data(n_examples=80, n_features=6)

    result = fit_sparse_probe(features, labels.to(dtype), k=2, seed=0)

    assert result.positive_label == 1
    assert result.negative_label == 0


def test_stratification_preserves_each_class_and_reports_realized_counts():
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(100, 5, generator=generator)
    labels = torch.tensor([1] * 4 + [0] * 96)
    features[:4, 0] += 3

    result = fit_sparse_probe(features, labels, k=1, test_fraction=0.3, seed=2)

    assert result.train_positive_count == 2
    assert result.test_positive_count == 2
    assert result.train_negative_count == 67
    assert result.test_negative_count == 29
    assert 0 <= result.metrics.f1 <= 1
    assert not bool(torch.isin(result.train_indices, result.test_indices).any())
    assert torch.equal(
        torch.cat((result.train_indices, result.test_indices)).sort().values,
        torch.arange(features.shape[0]),
    )


def test_inputs_and_global_rng_are_unchanged_and_results_are_frozen():
    features, labels = _planted_data(n_examples=80, n_features=6)
    features_before = features.clone()
    labels_before = labels.clone()
    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state()

    result = fit_sparse_probe(features, labels, k=2, seed=99)

    assert torch.equal(features, features_before)
    assert torch.equal(labels, labels_before)
    assert torch.equal(torch.random.get_rng_state(), state_before)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "seed", 0)


def test_forced_nonconvergence_raises():
    features, labels = _planted_data(n_examples=100, n_features=5)

    with pytest.raises(RuntimeError, match="did not converge"):
        fit_sparse_probe(
            features,
            labels,
            k=3,
            max_iter=1,
            gradient_tolerance=1e-12,
        )


def test_binary_metrics_zero_division_policy():
    metrics = _binary_metrics(torch.tensor([-2.0, -1.0]), torch.tensor([0, 1]))

    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.precision == 0
    assert metrics.recall == 0
    assert metrics.f1 == 0


@pytest.mark.parametrize(
    ("features", "labels", "kwargs", "message"),
    [
        (torch.ones(0, 2), torch.empty(0, dtype=torch.int64), {}, "non-empty"),
        (torch.tensor([[1.0], [float("nan")]]), torch.tensor([0, 1]), {}, "finite"),
        (
            torch.ones(4, 2, dtype=torch.float8_e4m3fn),
            torch.tensor([0, 1, 0, 1]),
            {},
            "supported dtype",
        ),
        (torch.ones(4, 2), torch.zeros(4, dtype=torch.int64), {}, "exactly two"),
        (torch.ones(6, 2), torch.tensor([0, 1, 2, 0, 1, 2]), {}, "exactly two"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"positive_label": 2}, "positive_label"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"k": 0}, "k must be"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"k": 3}, "k must be"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"test_fraction": 0}, "test_fraction"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"l2_strength": 0}, "l2_strength"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"class_weight": "bad"}, "class_weight"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"seed": -1}, "seed"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"preprocess": "bad"}, "preprocess"),
        (torch.ones(4, 2), torch.tensor([0, 1, 0, 1]), {"max_iter": 0}, "max_iter"),
        (
            torch.ones(4, 2),
            torch.tensor([0, 1, 0, 1]),
            {"gradient_tolerance": 0},
            "gradient_tolerance",
        ),
        (
            torch.ones(4, 2),
            torch.tensor([0, 1, 0, 1]),
            {"gradient_tolerance": 2},
            "gradient_tolerance",
        ),
    ],
)
def test_rejects_invalid_inputs(features, labels, kwargs, message):
    options = {"k": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        fit_sparse_probe(features, labels, **options)


@pytest.mark.parametrize(
    ("features", "labels"),
    [
        ([[1.0], [2.0], [3.0], [4.0]], torch.tensor([0, 1, 0, 1])),
        (torch.ones(4), torch.tensor([0, 1, 0, 1])),
        (torch.ones(4, 2, dtype=torch.int64), torch.tensor([0, 1, 0, 1])),
        (torch.ones(4, 2), torch.tensor([0.0, 1.0, 0.0, 1.0])),
        (torch.ones(4, 2), torch.tensor([[0, 1], [0, 1]])),
        (torch.ones(4, 2), torch.tensor([0, 1, 0])),
    ],
)
def test_runtime_typecheck_rejects_invalid_tensor_contracts(features, labels):
    with pytest.raises(TYPECHECK_ERRORS):
        fit_sparse_probe(features, labels, k=1)


def test_sweep_reuses_one_split_and_has_nested_selected_supports():
    features, labels = _planted_data(n_examples=180, n_features=10)

    sweep = sweep_sparse_probe(features, labels, ks=[1, 2, 4], seed=23)
    independent = fit_sparse_probe(features, labels, k=2, seed=23)

    assert isinstance(sweep, SparseProbeSweep)
    assert sweep.ks == (1, 2, 4)
    for result in sweep.results:
        assert torch.equal(result.train_indices, sweep.results[0].train_indices)
        assert torch.equal(result.test_indices, sweep.results[0].test_indices)
    assert torch.equal(sweep.results[0].selected_features, sweep.results[1].selected_features[:1])
    assert torch.equal(sweep.results[1].selected_features, sweep.results[2].selected_features[:2])
    assert torch.equal(sweep.results[1].selected_features, independent.selected_features)
    assert sweep.results[1].metrics == independent.metrics


def test_disabled_controls_return_empty_aligned_results():
    features, labels = _planted_data(n_examples=100, n_features=8)

    sweep = sweep_sparse_probe(features, labels, ks=[1, 3], seed=1)

    for k, random_control, shuffle_control in zip(
        sweep.ks,
        sweep.random_coordinate_controls,
        sweep.label_shuffle_controls,
        strict=True,
    ):
        assert isinstance(random_control, SparseProbeControl)
        assert random_control.supports.shape == (0, k)
        assert shuffle_control.supports.shape == (0, k)
        for metric_values in (
            random_control.accuracy,
            random_control.precision,
            random_control.recall,
            random_control.f1,
            shuffle_control.accuracy,
            shuffle_control.precision,
            shuffle_control.recall,
            shuffle_control.f1,
        ):
            assert metric_values.shape == (0,)
            assert metric_values.dtype == torch.float64


def test_controls_are_deterministic_use_unique_supports_and_do_not_touch_global_rng():
    features, labels = _planted_data(n_examples=140, n_features=12)
    torch.manual_seed(919)
    state_before = torch.random.get_rng_state()

    first = sweep_sparse_probe(
        features,
        labels,
        ks=[2],
        n_random_subsets=4,
        n_label_shuffles=4,
        seed=5,
    )
    second = sweep_sparse_probe(
        features,
        labels,
        ks=[2],
        n_random_subsets=4,
        n_label_shuffles=4,
        seed=5,
    )

    assert torch.equal(torch.random.get_rng_state(), state_before)
    for left, right in (
        (first.random_coordinate_controls[0], second.random_coordinate_controls[0]),
        (first.label_shuffle_controls[0], second.label_shuffle_controls[0]),
    ):
        assert torch.equal(left.supports, right.supports)
        assert torch.equal(left.accuracy, right.accuracy)
        assert torch.equal(left.precision, right.precision)
        assert torch.equal(left.recall, right.recall)
        assert torch.equal(left.f1, right.f1)
        for support in left.supports:
            assert torch.unique(support).numel() == 2


def test_controls_remain_below_a_strong_planted_feature():
    features, labels = _planted_data(n_examples=200, n_features=32, seed=6)

    sweep = sweep_sparse_probe(
        features,
        labels,
        ks=[1],
        n_random_subsets=8,
        n_label_shuffles=8,
        seed=18,
    )

    actual_f1 = sweep.results[0].metrics.f1
    assert actual_f1 > 0.98
    assert float(sweep.random_coordinate_controls[0].f1.median()) < actual_f1 - 0.2
    assert float(sweep.label_shuffle_controls[0].f1.median()) < actual_f1 - 0.2


def test_larger_k_improves_distributed_decodability_without_assigning_a_representation_label():
    generator = torch.Generator().manual_seed(77)
    labels = torch.arange(800) % 2
    features = torch.randn(800, 20, generator=generator)
    features[:, :4] += 0.55 * (2 * labels[:, None] - 1)
    permutation = torch.randperm(800, generator=generator)

    sweep = sweep_sparse_probe(features[permutation], labels[permutation], ks=[1, 4], seed=3)

    assert sweep.results[1].metrics.f1 > sweep.results[0].metrics.f1 + 0.08
    assert not hasattr(sweep, "representation_label")


@pytest.mark.parametrize(
    ("ks", "kwargs", "message"),
    [
        ([], {}, "ks must"),
        ([1, 1], {}, "strictly increasing"),
        ([2, 1], {}, "strictly increasing"),
        ([1, 9], {}, "feature count"),
        ([True], {}, "positive integers"),
        ([1], {"n_random_subsets": -1}, "n_random_subsets"),
        ([1], {"n_label_shuffles": -1}, "n_label_shuffles"),
    ],
)
def test_sweep_rejects_invalid_grid_and_control_counts(ks, kwargs, message):
    features, labels = _planted_data(n_examples=80, n_features=8)

    with pytest.raises(ValueError, match=message):
        sweep_sparse_probe(features, labels, ks=ks, **kwargs)
