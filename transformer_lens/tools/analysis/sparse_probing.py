"""Leakage-safe k-sparse binary probes over activation tensors.

This module is model-free: callers supply an ``[example, feature]`` activation
matrix and binary labels. Feature selection and optional preprocessing use the
training split only. Probe decodability does not establish causal model use,
monosemanticity, or superposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from jaxtyping import Bool, Float, Int, Integer

PreprocessMode = Literal["none", "standardize"]
ClassWeightMode = Literal["balanced"] | None
_SUPPORTED_FEATURE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@dataclass(frozen=True)
class SparseProbeMetrics:
    """Held-out binary-classification metrics and confusion counts."""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class SparseProbeResult:
    """Result of one train/test k-sparse binary probe fit.

    All tensors are detached CPU tensors. Floating-point tensors use float64;
    ``coefficients`` align with ``selected_features``.
    """

    feature_scores: Float[torch.Tensor, "feature"]
    selected_features: Int[torch.Tensor, "selected_feature"]
    coefficients: Float[torch.Tensor, "selected_feature"]
    intercept: Float[torch.Tensor, ""]
    preprocess_mean: Float[torch.Tensor, "selected_feature"]
    preprocess_scale: Float[torch.Tensor, "selected_feature"]
    constant_features: Bool[torch.Tensor, "selected_feature"]
    train_indices: Int[torch.Tensor, "train_example"]
    test_indices: Int[torch.Tensor, "test_example"]
    metrics: SparseProbeMetrics
    positive_label: int
    negative_label: int
    train_positive_count: int
    train_negative_count: int
    test_positive_count: int
    test_negative_count: int
    preprocess: PreprocessMode
    class_weight: ClassWeightMode
    l2_strength: float
    test_fraction: float
    seed: int
    k: int
    max_iter: int
    gradient_tolerance: float
    objective: float
    gradient_inf_norm: float
    iterations: int
    function_evaluations: int


@dataclass(frozen=True)
class _ValidatedInputs:
    features: torch.Tensor
    canonical_labels: torch.Tensor
    positive_label: int
    negative_label: int
    k: int
    test_fraction: float
    preprocess: PreprocessMode
    class_weight: ClassWeightMode
    l2_strength: float
    seed: int
    max_iter: int
    gradient_tolerance: float


@dataclass(frozen=True)
class _FitOutcome:
    coefficients: torch.Tensor
    intercept: torch.Tensor
    objective: float
    gradient_inf_norm: float
    iterations: int
    function_evaluations: int


def _finite_positive_real(value: int | float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive real number, got {value!r}")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_inputs(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    k: int,
    test_fraction: int | float,
    positive_label: int | bool,
    preprocess: str,
    class_weight: str | None,
    l2_strength: int | float,
    seed: int,
    max_iter: int,
    gradient_tolerance: int | float,
) -> _ValidatedInputs:
    if not isinstance(features, torch.Tensor):
        raise ValueError(f"features must be a torch.Tensor, got {type(features).__name__}")
    if features.ndim != 2:
        raise ValueError(f"features must be two-dimensional, got shape {tuple(features.shape)}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"feature dimensions must be non-empty, got shape {tuple(features.shape)}")
    if not torch.is_floating_point(features) or torch.is_complex(features):
        raise ValueError(f"features must have a real floating-point dtype, got {features.dtype}")
    if features.dtype not in _SUPPORTED_FEATURE_DTYPES:
        raise ValueError(
            "features must have a supported dtype: float16, bfloat16, float32, or float64; "
            f"got {features.dtype}"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must contain only finite values")

    if not isinstance(labels, torch.Tensor):
        raise ValueError(f"labels must be a torch.Tensor, got {type(labels).__name__}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be one-dimensional, got shape {tuple(labels.shape)}")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("features and labels must contain the same number of examples")
    if torch.is_floating_point(labels) or torch.is_complex(labels):
        raise ValueError(f"labels must have a Boolean or integer dtype, got {labels.dtype}")

    labels_cpu = labels.detach().to(device="cpu")
    classes = [int(value) for value in torch.unique(labels_cpu, sorted=True).tolist()]
    if len(classes) != 2:
        raise ValueError(f"labels must contain exactly two classes, got {classes}")
    if isinstance(positive_label, bool):
        resolved_positive = int(positive_label)
    elif isinstance(positive_label, int):
        resolved_positive = positive_label
    else:
        raise ValueError(f"positive_label must be an integer or Boolean, got {positive_label!r}")
    if resolved_positive not in classes:
        raise ValueError(f"positive_label {resolved_positive} is not present in labels {classes}")
    resolved_negative = classes[0] if classes[1] == resolved_positive else classes[1]
    canonical_labels = labels_cpu == resolved_positive
    counts = torch.bincount(canonical_labels.to(torch.int64), minlength=2)
    if int(counts.min()) < 2:
        raise ValueError("each class must contain at least two examples")

    validated_k = _positive_integer(k, "k")
    if validated_k > features.shape[1]:
        raise ValueError(f"k must be at most the feature count {features.shape[1]}, got {k}")
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)):
        raise ValueError(f"test_fraction must be a finite real in (0, 1), got {test_fraction!r}")
    validated_fraction = float(test_fraction)
    if not math.isfinite(validated_fraction) or not 0 < validated_fraction < 1:
        raise ValueError(f"test_fraction must be a finite real in (0, 1), got {test_fraction!r}")
    if preprocess not in ("none", "standardize"):
        raise ValueError(f"preprocess must be 'none' or 'standardize', got {preprocess!r}")
    validated_preprocess = cast(PreprocessMode, preprocess)
    if class_weight not in ("balanced", None):
        raise ValueError(f"class_weight must be 'balanced' or None, got {class_weight!r}")
    validated_class_weight = cast(ClassWeightMode, class_weight)
    validated_l2 = _finite_positive_real(l2_strength, "l2_strength")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
    validated_max_iter = _positive_integer(max_iter, "max_iter")
    validated_tolerance = _finite_positive_real(gradient_tolerance, "gradient_tolerance")
    if validated_tolerance > 1:
        raise ValueError(
            "gradient_tolerance must be a finite real in (0, 1], " f"got {gradient_tolerance!r}"
        )
    return _ValidatedInputs(
        features=features.detach(),
        canonical_labels=canonical_labels,
        positive_label=resolved_positive,
        negative_label=resolved_negative,
        k=validated_k,
        test_fraction=validated_fraction,
        preprocess=validated_preprocess,
        class_weight=validated_class_weight,
        l2_strength=validated_l2,
        seed=seed,
        max_iter=validated_max_iter,
        gradient_tolerance=validated_tolerance,
    )


def _stratified_split(
    canonical_labels: torch.Tensor,
    test_fraction: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    train_parts = []
    test_parts = []
    for class_value in (False, True):
        indices = torch.where(canonical_labels == class_value)[0]
        permutation = torch.randperm(indices.numel(), generator=generator)
        shuffled = indices[permutation]
        test_count = min(max(math.ceil(test_fraction * indices.numel()), 1), indices.numel() - 1)
        test_parts.append(shuffled[:test_count])
        train_parts.append(shuffled[test_count:])
    return torch.cat(train_parts), torch.cat(test_parts)


def _feature_scores(
    features: torch.Tensor,
    train_labels: torch.Tensor,
    train_indices: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    device_indices = train_indices.to(device=features.device)
    train_features = features.index_select(0, device_indices).to(dtype=compute_dtype)
    positive = train_labels.to(device=features.device)
    scores = train_features[positive].mean(dim=0) - train_features[~positive].mean(dim=0)
    return scores.to(device="cpu").to(dtype=torch.float64)


def _selected_data(
    features: torch.Tensor,
    selected_features: torch.Tensor,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    preprocess: PreprocessMode,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_device = selected_features.to(device=features.device)
    train_device = train_indices.to(device=features.device)
    test_device = test_indices.to(device=features.device)
    train = (
        features.index_select(0, train_device)
        .index_select(1, selected_device)
        .to(device="cpu")
        .to(dtype=torch.float64)
    )
    test = (
        features.index_select(0, test_device)
        .index_select(1, selected_device)
        .to(device="cpu")
        .to(dtype=torch.float64)
    )
    raw_scale = train.std(dim=0, correction=0)
    constant = raw_scale == 0
    if preprocess == "standardize":
        mean = train.mean(dim=0)
        scale = torch.where(constant, torch.ones_like(raw_scale), raw_scale)
        return (train - mean) / scale, (test - mean) / scale, mean, scale, constant
    mean = torch.zeros(train.shape[1], dtype=torch.float64)
    scale = torch.ones(train.shape[1], dtype=torch.float64)
    return train, test, mean, scale, constant


def _sample_weights(labels: torch.Tensor, class_weight: ClassWeightMode) -> torch.Tensor:
    if class_weight is None:
        return torch.ones_like(labels)
    count = labels.numel()
    positive_count = labels.sum()
    return torch.where(
        labels == 1,
        count / (2 * positive_count),
        count / (2 * (count - positive_count)),
    )


def _objective(
    features: torch.Tensor,
    labels: torch.Tensor,
    parameters: torch.Tensor,
    sample_weights: torch.Tensor,
    l2_strength: float,
) -> torch.Tensor:
    coefficients = parameters[:-1]
    logits = features @ coefficients + parameters[-1]
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (sample_weights * losses).mean() + 0.5 * l2_strength * coefficients.square().sum()


def _objective_gradient(
    features: torch.Tensor,
    labels: torch.Tensor,
    parameters: torch.Tensor,
    sample_weights: torch.Tensor,
    l2_strength: float,
) -> torch.Tensor:
    coefficients = parameters[:-1]
    residual = sample_weights * (torch.sigmoid(features @ coefficients + parameters[-1]) - labels)
    residual = residual / labels.numel()
    return torch.cat(
        (features.T @ residual + l2_strength * coefficients, residual.sum().reshape(1))
    )


def _fit_logistic(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weight: ClassWeightMode,
    l2_strength: float,
    max_iter: int,
    gradient_tolerance: float,
) -> _FitOutcome:
    labels = labels.to(dtype=torch.float64)
    sample_weights = _sample_weights(labels, class_weight)
    parameters = torch.zeros(features.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters],
        max_iter=max_iter,
        tolerance_grad=gradient_tolerance,
        tolerance_change=max(torch.finfo(torch.float64).eps, gradient_tolerance**2),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _objective(features, labels, parameters, sample_weights, l2_strength)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError as error:
        raise RuntimeError("sparse probe optimizer failed") from error
    detached = parameters.detach()
    objective = float(
        _objective(features, labels, detached, sample_weights, l2_strength).detach().item()
    )
    gradient = _objective_gradient(features, labels, detached, sample_weights, l2_strength)
    gradient_inf_norm = float(gradient.abs().max().item())
    if not math.isfinite(objective) or not math.isfinite(gradient_inf_norm):
        raise RuntimeError("sparse probe optimizer produced non-finite output")
    if gradient_inf_norm > gradient_tolerance:
        raise RuntimeError(
            "sparse probe optimizer did not converge: "
            f"gradient infinity norm {gradient_inf_norm:.6g} exceeds {gradient_tolerance:.6g}"
        )
    state = optimizer.state[parameters]
    return _FitOutcome(
        coefficients=detached[:-1].clone(),
        intercept=detached[-1].clone(),
        objective=objective,
        gradient_inf_norm=gradient_inf_norm,
        iterations=int(state.get("n_iter", 0)),
        function_evaluations=int(state.get("func_evals", 0)),
    )


def _binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> SparseProbeMetrics:
    predictions = logits >= 0
    positive = labels.to(dtype=torch.bool)
    true_positives = int((predictions & positive).sum().item())
    true_negatives = int((~predictions & ~positive).sum().item())
    false_positives = int((predictions & ~positive).sum().item())
    false_negatives = int((~predictions & positive).sum().item())
    count = labels.numel()
    accuracy = (true_positives + true_negatives) / count
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = 0.0 if precision_denominator == 0 else true_positives / precision_denominator
    recall = 0.0 if recall_denominator == 0 else true_positives / recall_denominator
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    f1 = 0.0 if f1_denominator == 0 else 2 * true_positives / f1_denominator
    return SparseProbeMetrics(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _fit_result(
    validated: _ValidatedInputs,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    feature_scores: torch.Tensor,
    selected_features: torch.Tensor,
) -> SparseProbeResult:
    train_features, test_features, mean, scale, constant = _selected_data(
        validated.features,
        selected_features,
        train_indices,
        test_indices,
        validated.preprocess,
    )
    train_labels = validated.canonical_labels[train_indices]
    fit = _fit_logistic(
        train_features,
        train_labels,
        class_weight=validated.class_weight,
        l2_strength=validated.l2_strength,
        max_iter=validated.max_iter,
        gradient_tolerance=validated.gradient_tolerance,
    )
    test_labels = validated.canonical_labels[test_indices]
    metrics = _binary_metrics(test_features @ fit.coefficients + fit.intercept, test_labels)
    return SparseProbeResult(
        feature_scores=feature_scores,
        selected_features=selected_features,
        coefficients=fit.coefficients,
        intercept=fit.intercept,
        preprocess_mean=mean,
        preprocess_scale=scale,
        constant_features=constant,
        train_indices=train_indices,
        test_indices=test_indices,
        metrics=metrics,
        positive_label=validated.positive_label,
        negative_label=validated.negative_label,
        train_positive_count=int(train_labels.sum().item()),
        train_negative_count=int((~train_labels).sum().item()),
        test_positive_count=int(test_labels.sum().item()),
        test_negative_count=int((~test_labels).sum().item()),
        preprocess=validated.preprocess,
        class_weight=validated.class_weight,
        l2_strength=validated.l2_strength,
        test_fraction=validated.test_fraction,
        seed=validated.seed,
        k=validated.k,
        max_iter=validated.max_iter,
        gradient_tolerance=validated.gradient_tolerance,
        objective=fit.objective,
        gradient_inf_norm=fit.gradient_inf_norm,
        iterations=fit.iterations,
        function_evaluations=fit.function_evaluations,
    )


def fit_sparse_probe(
    features: Float[torch.Tensor, "example feature"],
    labels: Bool[torch.Tensor, "example"] | Integer[torch.Tensor, "example"],
    *,
    k: int,
    test_fraction: int | float = 0.3,
    positive_label: int | bool = 1,
    preprocess: str = "none",
    class_weight: str | None = "balanced",
    l2_strength: int | float = 1e-2,
    seed: int = 0,
    max_iter: int = 200,
    gradient_tolerance: int | float = 1e-7,
) -> SparseProbeResult:
    """Fit a train-only-selected k-sparse binary logistic probe.

    The stratified split is created before feature scoring or optional
    standardization. Floating result tensors are detached CPU float64 tensors.

    Args:
        features: Finite float16/bfloat16/float32/float64 tensor shaped
            ``[example, feature]``.
        labels: Boolean or integer binary labels shaped ``[example]``.
        k: Number of coordinates selected by absolute train class-mean difference.
        test_fraction: Requested held-out fraction within each class.
        positive_label: Label defining the positive class and score sign.
        preprocess: ``"none"`` or train-only ``"standardize"``.
        class_weight: ``"balanced"`` or ``None`` for unweighted BCE.
        l2_strength: Positive coefficient penalty in the logistic objective.
        seed: Local CPU-generator seed used only for the stratified split.
        max_iter: Maximum LBFGS iterations.
        gradient_tolerance: Required final objective-gradient infinity norm.

    Returns:
        Selected support, fitted parameters, split/preprocessing metadata, metrics,
        and optimizer diagnostics.

    Raises:
        ValueError: If inputs or options violate the binary-probe contract.
        RuntimeError: If the optimizer fails or misses its convergence threshold.
    """
    validated = _validate_inputs(
        features,
        labels,
        k=k,
        test_fraction=test_fraction,
        positive_label=positive_label,
        preprocess=preprocess,
        class_weight=class_weight,
        l2_strength=l2_strength,
        seed=seed,
        max_iter=max_iter,
        gradient_tolerance=gradient_tolerance,
    )
    generator = torch.Generator(device="cpu").manual_seed(validated.seed)
    train_indices, test_indices = _stratified_split(
        validated.canonical_labels, validated.test_fraction, generator
    )
    feature_scores = _feature_scores(
        validated.features, validated.canonical_labels[train_indices], train_indices
    )
    selected_features = torch.argsort(feature_scores.abs(), descending=True, stable=True)[
        : validated.k
    ]
    return _fit_result(
        validated,
        train_indices,
        test_indices,
        feature_scores,
        selected_features,
    )
