"""Leakage-safe k-sparse binary probes over ``[example, feature]`` activation tensors.

Model-free; feature selection and optional standardization see only the training split.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from jaxtyping import Bool, Float, Int, Integer

from transformer_lens.utilities.statistics import derive_generator

PreprocessMode = Literal["none", "standardize"]
ClassWeightMode = Literal["balanced"] | None
ArmMode = Literal["random_coordinate", "label_shuffle"]
_SUPPORTED_FEATURE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


class SparseProbeConvergenceError(RuntimeError):
    """A fit failed to converge or went non-finite; sweeps catch only this for control fits."""


@dataclass(frozen=True)
class SparseProbeMetrics:
    """Held-out metrics; confusion counts and ratios use the logit-zero threshold, while
    ``roc_auc`` and ``average_precision`` are threshold-free and tie-aware."""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float


@dataclass(frozen=True)
class SparseProbeResult:
    """One k-sparse probe fit; tensors are detached CPU and floating ones are float64."""

    feature_scores: Float[torch.Tensor, "feature"]  # positive minus negative class train mean
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
    std_floor: float
    class_weight: ClassWeightMode
    l2_strength: float
    test_fraction: float
    seed: int
    k: int
    max_iter: int
    max_refinement_steps: int
    decrement_tolerance: float
    objective: float
    gradient_inf_norm: float
    newton_decrement: float
    refinement_steps: int
    iterations: int
    function_evaluations: int
    stop_reason: str


@dataclass(frozen=True)
class SparseProbeControl:
    """Raw held-out metrics of one control at one k; rows cover only converged repeats."""

    supports: Int[torch.Tensor, "repeat selected_feature"]
    accuracy: Float[torch.Tensor, "repeat"]
    precision: Float[torch.Tensor, "repeat"]
    recall: Float[torch.Tensor, "repeat"]
    f1: Float[torch.Tensor, "repeat"]
    roc_auc: Float[torch.Tensor, "repeat"]
    average_precision: Float[torch.Tensor, "repeat"]


@dataclass(frozen=True)
class SparseProbeRejection:
    """A control fit excluded from the sweep because it did not converge."""

    arm: ArmMode
    k: int
    repeat: int
    support: Int[torch.Tensor, "selected_feature"]
    reason: str


@dataclass(frozen=True)
class SparseProbeSweep:
    """Probe results and aligned controls over a strictly increasing k-grid."""

    ks: tuple[int, ...]
    results: tuple[SparseProbeResult, ...]
    random_coordinate_controls: tuple[SparseProbeControl, ...]
    label_shuffle_controls: tuple[SparseProbeControl, ...]
    seed: int
    rejections: tuple[SparseProbeRejection, ...] = ()


@dataclass(frozen=True)
class _ValidatedInputs:
    features: torch.Tensor
    canonical_labels: torch.Tensor
    positive_label: int
    negative_label: int
    k: int
    test_fraction: float
    preprocess: PreprocessMode
    std_floor: float
    class_weight: ClassWeightMode
    l2_strength: float
    seed: int
    max_iter: int
    max_refinement_steps: int
    decrement_tolerance: float


@dataclass(frozen=True)
class _FitOutcome:
    coefficients: torch.Tensor
    intercept: torch.Tensor
    objective: float
    gradient_inf_norm: float
    newton_decrement: float
    refinement_steps: int
    iterations: int
    function_evaluations: int
    stop_reason: str


def _finite_real(
    value: int | float, name: str, requirement: str, valid: Callable[[float], bool]
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite {requirement}, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or not valid(result):
        raise ValueError(f"{name} must be a finite {requirement}, got {value!r}")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return value


def _validate_inputs(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    k: int,
    test_fraction: int | float,
    positive_label: int | bool,
    preprocess: str,
    std_floor: int | float,
    class_weight: str | None,
    l2_strength: int | float,
    seed: int,
    max_iter: int,
    max_refinement_steps: int,
    decrement_tolerance: int | float,
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
    validated_fraction = _finite_real(
        test_fraction, "test_fraction", "real in (0, 1)", lambda value: 0 < value < 1
    )
    if preprocess not in ("none", "standardize"):
        raise ValueError(f"preprocess must be 'none' or 'standardize', got {preprocess!r}")
    validated_std_floor = _finite_real(
        std_floor, "std_floor", "nonnegative real", lambda value: value >= 0
    )
    validated_preprocess = cast(PreprocessMode, preprocess)
    if class_weight not in ("balanced", None):
        raise ValueError(f"class_weight must be 'balanced' or None, got {class_weight!r}")
    validated_class_weight = cast(ClassWeightMode, class_weight)
    validated_l2 = _finite_real(
        l2_strength, "l2_strength", "positive real number", lambda value: value > 0
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError(f"seed must be an integer in [0, 2**63), got {seed!r}")
    validated_max_iter = _positive_integer(max_iter, "max_iter")
    validated_refinement_steps = _nonnegative_integer(max_refinement_steps, "max_refinement_steps")
    validated_tolerance = _finite_real(
        decrement_tolerance, "decrement_tolerance", "positive real number", lambda value: value > 0
    )
    if validated_tolerance >= 1:
        raise ValueError(
            f"decrement_tolerance must be a finite real in (0, 1), got {decrement_tolerance!r}"
        )
    return _ValidatedInputs(
        features=features.detach(),
        canonical_labels=canonical_labels,
        positive_label=resolved_positive,
        negative_label=resolved_negative,
        k=validated_k,
        test_fraction=validated_fraction,
        preprocess=validated_preprocess,
        std_floor=validated_std_floor,
        class_weight=validated_class_weight,
        l2_strength=validated_l2,
        seed=seed,
        max_iter=validated_max_iter,
        max_refinement_steps=validated_refinement_steps,
        decrement_tolerance=validated_tolerance,
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
    std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_device = selected_features.to(device=features.device)

    def gather(indices: torch.Tensor) -> torch.Tensor:
        rows = features.index_select(0, indices.to(device=features.device))
        return rows.index_select(1, selected_device).to(device="cpu").to(dtype=torch.float64)

    train, test = gather(train_indices), gather(test_indices)
    raw_scale = train.std(dim=0, correction=0)
    constant = raw_scale == 0
    if preprocess == "standardize":
        mean = train.mean(dim=0)
        # Guards 0/0 when std_floor=0; zero-variance columns get a zero coefficient either way.
        scale = torch.where(constant, torch.ones_like(raw_scale), raw_scale.clamp(min=std_floor))
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


_LBFGS_TOLERANCE_GRAD = 1e-7
_NEWTON_RIDGE = 1e-12
_ARMIJO_SLOPE = 1e-4
_MAX_BACKTRACKS = 60


def _newton_direction(
    features: torch.Tensor,
    labels: torch.Tensor,
    parameters: torch.Tensor,
    sample_weights: torch.Tensor,
    l2_strength: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the gradient, Newton direction ``H^-1 g`` and decrement ``g^T H^-1 g / 2``."""
    count = labels.numel()
    design = torch.cat((features, torch.ones(count, 1, dtype=torch.float64)), dim=1)
    probabilities = torch.sigmoid(design @ parameters)
    gradient = _objective_gradient(features, labels, parameters, sample_weights, l2_strength)
    curvature = sample_weights * probabilities * (1.0 - probabilities) / count
    hessian = (design * curvature[:, None]).T @ design
    hessian.diagonal()[: features.shape[1]] += l2_strength
    # Saturated sigmoids zero the (unregularised) intercept curvature, so the solve is done on
    # the Jacobi-equilibrated matrix with a ridge relative to each pivot; a raw factorisation
    # fails on converged fits there.
    diagonal = hessian.diagonal()
    scale = torch.where(diagonal > 0, diagonal.rsqrt(), torch.ones_like(diagonal))
    scaled = scale[:, None] * hessian * scale[None, :]
    scaled.diagonal().add_(_NEWTON_RIDGE)
    factor, info = torch.linalg.cholesky_ex(scaled)
    if int(info.item()) != 0:
        raise SparseProbeConvergenceError(
            "sparse probe optimizer did not converge: "
            "Hessian factorization failed during Newton refinement"
        )
    direction = scale * torch.cholesky_solve((scale * gradient)[:, None], factor)[:, 0]
    decrement = 0.5 * float((gradient @ direction).item())
    if not math.isfinite(decrement) or decrement < 0.0:
        raise SparseProbeConvergenceError(
            "sparse probe optimizer did not converge: "
            f"Newton decrement {decrement!r} is not a finite nonnegative number"
        )
    return gradient, direction, decrement


def _refine_with_newton(
    features: torch.Tensor,
    labels: torch.Tensor,
    parameters: torch.Tensor,
    sample_weights: torch.Tensor,
    l2_strength: float,
    *,
    max_refinement_steps: int,
    decrement_tolerance: float,
) -> tuple[torch.Tensor, float, int]:
    """Damped Newton steps until the decrement meets the tolerance; the check precedes any step."""
    current = parameters
    value = float(_objective(features, labels, current, sample_weights, l2_strength).item())
    steps = 0
    while True:
        gradient, direction, decrement = _newton_direction(
            features, labels, current, sample_weights, l2_strength
        )
        if decrement <= decrement_tolerance:
            return current, decrement, steps
        if steps >= max_refinement_steps:
            raise SparseProbeConvergenceError(
                "sparse probe optimizer did not converge: "
                f"Newton decrement {decrement:.6g} exceeds {decrement_tolerance:.6g} "
                f"after {steps} refinement steps"
            )
        slope = float((gradient @ direction).item())
        step = 1.0
        for _ in range(_MAX_BACKTRACKS):
            candidate = current - step * direction
            candidate_value = float(
                _objective(features, labels, candidate, sample_weights, l2_strength).item()
            )
            if math.isfinite(candidate_value) and (
                candidate_value <= value - _ARMIJO_SLOPE * step * slope
            ):
                break
            step *= 0.5
        else:
            raise SparseProbeConvergenceError(
                "sparse probe optimizer did not converge: "
                "Newton refinement line search found no descent step"
            )
        current, value = candidate, candidate_value
        steps += 1


def _fit_logistic(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weight: ClassWeightMode,
    l2_strength: float,
    max_iter: int,
    max_refinement_steps: int,
    decrement_tolerance: float,
) -> _FitOutcome:
    labels = labels.to(dtype=torch.float64)
    sample_weights = _sample_weights(labels, class_weight)
    parameters = torch.zeros(features.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    max_eval = max_iter * 5 // 4
    # The solver's own stop is fixed so that no user argument can end the solve early;
    # acceptance is decided afterwards by the Newton decrement.
    optimizer = torch.optim.LBFGS(
        [parameters],
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=_LBFGS_TOLERANCE_GRAD,
        tolerance_change=0.0,
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
        raise SparseProbeConvergenceError("sparse probe optimizer failed") from error
    detached = parameters.detach()
    lbfgs_gradient = _objective_gradient(features, labels, detached, sample_weights, l2_strength)
    lbfgs_gradient_inf_norm = float(lbfgs_gradient.abs().max().item())
    if not math.isfinite(lbfgs_gradient_inf_norm):
        raise SparseProbeConvergenceError("sparse probe optimizer produced non-finite output")
    state = optimizer.state[parameters]
    iterations = int(state.get("n_iter", 0))
    function_evaluations = int(state.get("func_evals", 0))
    # Same order LBFGS tests its exits; with tolerance_change at 0.0 the only exit left after
    # the caps and the gradient stop is a strong-Wolfe search that made no progress.
    if iterations >= max_iter:
        stop_reason = "max_iter"
    elif function_evaluations >= max_eval:
        stop_reason = "max_eval"
    elif lbfgs_gradient_inf_norm <= _LBFGS_TOLERANCE_GRAD:
        stop_reason = "tolerance_grad"
    else:
        stop_reason = "line_search"
    refined, decrement, refinement_steps = _refine_with_newton(
        features,
        labels,
        detached,
        sample_weights,
        l2_strength,
        max_refinement_steps=max_refinement_steps,
        decrement_tolerance=decrement_tolerance,
    )
    objective = float(_objective(features, labels, refined, sample_weights, l2_strength).item())
    gradient = _objective_gradient(features, labels, refined, sample_weights, l2_strength)
    gradient_inf_norm = float(gradient.abs().max().item())
    if not math.isfinite(objective) or not math.isfinite(gradient_inf_norm):
        raise SparseProbeConvergenceError("sparse probe optimizer produced non-finite output")
    return _FitOutcome(
        coefficients=refined[:-1].clone(),
        intercept=refined[-1].clone(),
        objective=objective,
        gradient_inf_norm=gradient_inf_norm,
        newton_decrement=decrement,
        refinement_steps=refinement_steps,
        iterations=iterations,
        function_evaluations=function_evaluations,
        stop_reason=stop_reason,
    )


def _roc_auc(logits: torch.Tensor, positive: torch.Tensor) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied logits; NaN if a class is absent."""
    positive_count = int(positive.sum().item())
    negative_count = positive.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        return math.nan
    _, inverse, counts = torch.unique(logits, return_inverse=True, return_counts=True)
    counts = counts.to(dtype=torch.float64)
    # A tie group occupying ranks start+1..end gets their mean, end - (count - 1) / 2.
    average_ranks = (counts.cumsum(0) - (counts - 1) / 2)[inverse]
    positive_rank_sum = float(average_ranks[positive].sum().item())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def _average_precision(logits: torch.Tensor, positive: torch.Tensor) -> float:
    """Step-wise average precision over distinct logits; NaN if there is no positive."""
    positive_count = int(positive.sum().item())
    if positive_count == 0:
        return math.nan
    _, inverse = torch.unique(logits, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    # Tied logits form one threshold, so the recall gained there is scored at a single precision.
    group_positives = torch.zeros(group_count, dtype=torch.float64).index_add_(
        0, inverse, positive.to(dtype=torch.float64)
    )
    group_sizes = torch.bincount(inverse, minlength=group_count).to(dtype=torch.float64)
    positives_descending = group_positives.flip(0)
    precision = positives_descending.cumsum(0) / group_sizes.flip(0).cumsum(0)
    return float((positives_descending * precision).sum().item()) / positive_count


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
        roc_auc=_roc_auc(logits, positive),
        average_precision=_average_precision(logits, positive),
    )


def _fit_selected(
    validated: _ValidatedInputs,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    selected_features: torch.Tensor,
    train_labels: torch.Tensor,
) -> tuple[_FitOutcome, SparseProbeMetrics, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_features, test_features, mean, scale, constant = _selected_data(
        validated.features,
        selected_features,
        train_indices,
        test_indices,
        validated.preprocess,
        validated.std_floor,
    )
    fit = _fit_logistic(
        train_features,
        train_labels,
        class_weight=validated.class_weight,
        l2_strength=validated.l2_strength,
        max_iter=validated.max_iter,
        max_refinement_steps=validated.max_refinement_steps,
        decrement_tolerance=validated.decrement_tolerance,
    )
    test_labels = validated.canonical_labels[test_indices]
    metrics = _binary_metrics(test_features @ fit.coefficients + fit.intercept, test_labels)
    return fit, metrics, mean, scale, constant


def _fit_result(
    validated: _ValidatedInputs,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    feature_scores: torch.Tensor,
    selected_features: torch.Tensor,
    k: int,
) -> SparseProbeResult:
    train_labels = validated.canonical_labels[train_indices]
    test_labels = validated.canonical_labels[test_indices]
    fit, metrics, mean, scale, constant = _fit_selected(
        validated, train_indices, test_indices, selected_features, train_labels
    )
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
        std_floor=validated.std_floor,
        class_weight=validated.class_weight,
        l2_strength=validated.l2_strength,
        test_fraction=validated.test_fraction,
        seed=validated.seed,
        k=k,
        max_iter=validated.max_iter,
        max_refinement_steps=validated.max_refinement_steps,
        decrement_tolerance=validated.decrement_tolerance,
        objective=fit.objective,
        gradient_inf_norm=fit.gradient_inf_norm,
        newton_decrement=fit.newton_decrement,
        refinement_steps=fit.refinement_steps,
        iterations=fit.iterations,
        function_evaluations=fit.function_evaluations,
        stop_reason=fit.stop_reason,
    )


def _fit_control(
    validated: _ValidatedInputs,
    train_indices: torch.Tensor,
    test_indices: torch.Tensor,
    selected_features: torch.Tensor,
    train_labels: torch.Tensor,
) -> SparseProbeMetrics:
    return _fit_selected(validated, train_indices, test_indices, selected_features, train_labels)[1]


def _control_result(
    supports: list[torch.Tensor],
    metrics: list[SparseProbeMetrics],
    k: int,
) -> SparseProbeControl:
    support_tensor = torch.stack(supports) if supports else torch.empty((0, k), dtype=torch.int64)

    def metric_tensor(name: str) -> torch.Tensor:
        return torch.tensor(
            [float(getattr(metric, name)) for metric in metrics], dtype=torch.float64
        )

    return SparseProbeControl(
        supports=support_tensor,
        accuracy=metric_tensor("accuracy"),
        precision=metric_tensor("precision"),
        recall=metric_tensor("recall"),
        f1=metric_tensor("f1"),
        roc_auc=metric_tensor("roc_auc"),
        average_precision=metric_tensor("average_precision"),
    )


def fit_sparse_probe(
    features: Float[torch.Tensor, "example feature"],
    labels: Bool[torch.Tensor, "example"] | Integer[torch.Tensor, "example"],
    *,
    k: int,
    test_fraction: int | float = 0.3,
    positive_label: int | bool = 1,
    preprocess: str = "none",
    std_floor: int | float = 1e-3,
    class_weight: str | None = "balanced",
    l2_strength: int | float = 1e-2,
    seed: int = 0,
    max_iter: int = 200,
    max_refinement_steps: int = 25,
    decrement_tolerance: int | float = 1e-12,
) -> SparseProbeResult:
    """Fit a logistic probe on the k features with the largest absolute train class-mean gap.

    Accepted only once the Newton decrement ``g^T H^-1 g / 2`` (nats) is ``<= decrement_tolerance``.
    """
    validated = _validate_inputs(
        features,
        labels,
        k=k,
        test_fraction=test_fraction,
        positive_label=positive_label,
        preprocess=preprocess,
        std_floor=std_floor,
        class_weight=class_weight,
        l2_strength=l2_strength,
        seed=seed,
        max_iter=max_iter,
        max_refinement_steps=max_refinement_steps,
        decrement_tolerance=decrement_tolerance,
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
        validated, train_indices, test_indices, feature_scores, selected_features, validated.k
    )


def _control_generator(seed: int, k: int, arm: ArmMode, repeat: int) -> torch.Generator:
    """Per-draw CPU generator keyed on ``(seed, k, arm, repeat)``."""
    # Short arm names in the key keep previously released control draws reproducible.
    key_arm = "random" if arm == "random_coordinate" else "shuffle"
    return derive_generator(seed, k, key_arm, repeat)


def sweep_sparse_probe(
    features: Float[torch.Tensor, "example feature"],
    labels: Bool[torch.Tensor, "example"] | Integer[torch.Tensor, "example"],
    *,
    ks: Sequence[int],
    test_fraction: int | float = 0.3,
    positive_label: int | bool = 1,
    preprocess: str = "none",
    std_floor: int | float = 1e-3,
    class_weight: str | None = "balanced",
    l2_strength: int | float = 1e-2,
    n_random_subsets: int = 0,
    n_label_shuffles: int = 0,
    seed: int = 0,
    max_iter: int = 200,
    max_refinement_steps: int = 25,
    decrement_tolerance: int | float = 1e-12,
) -> SparseProbeSweep:
    """Fit probes for each k of a strictly increasing grid on one split, plus raw null controls.

    Shuffle controls score on true held-out labels; non-converged controls go to ``rejections``.
    """
    if isinstance(ks, (str, bytes)) or not isinstance(ks, Sequence) or not ks:
        raise ValueError("ks must be a non-empty sequence of positive integers")
    k_values = tuple(ks)
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in k_values):
        raise ValueError("ks must contain positive integers")
    if any(right <= left for left, right in zip(k_values, k_values[1:])):
        raise ValueError("ks must be strictly increasing and unique")
    random_count = _nonnegative_integer(n_random_subsets, "n_random_subsets")
    shuffle_count = _nonnegative_integer(n_label_shuffles, "n_label_shuffles")
    validated = _validate_inputs(
        features,
        labels,
        k=k_values[-1],
        test_fraction=test_fraction,
        positive_label=positive_label,
        preprocess=preprocess,
        std_floor=std_floor,
        class_weight=class_weight,
        l2_strength=l2_strength,
        seed=seed,
        max_iter=max_iter,
        max_refinement_steps=max_refinement_steps,
        decrement_tolerance=decrement_tolerance,
    )
    generator = torch.Generator(device="cpu").manual_seed(validated.seed)
    train_indices, test_indices = _stratified_split(
        validated.canonical_labels, validated.test_fraction, generator
    )
    train_labels = validated.canonical_labels[train_indices]
    feature_scores = _feature_scores(validated.features, train_labels, train_indices)
    ranked_features = torch.argsort(feature_scores.abs(), descending=True, stable=True)
    results = tuple(
        _fit_result(validated, train_indices, test_indices, feature_scores, ranked_features[:k], k)
        for k in k_values
    )

    random_controls: list[SparseProbeControl] = []
    shuffle_controls: list[SparseProbeControl] = []
    rejections: list[SparseProbeRejection] = []
    feature_count = validated.features.shape[1]
    arms: tuple[tuple[ArmMode, int, list[SparseProbeControl]], ...] = (
        ("random_coordinate", random_count, random_controls),
        ("label_shuffle", shuffle_count, shuffle_controls),
    )
    for k in k_values:
        for arm, count, controls in arms:
            supports = []
            metrics = []
            for repeat in range(count):
                draw_generator = _control_generator(validated.seed, k, arm, repeat)
                if arm == "random_coordinate":
                    fit_labels = train_labels
                    support = torch.randperm(feature_count, generator=draw_generator)[:k]
                    support = support.sort().values
                else:
                    permutation = torch.randperm(train_labels.numel(), generator=draw_generator)
                    fit_labels = train_labels[permutation]
                    scores = _feature_scores(validated.features, fit_labels, train_indices)
                    support = torch.argsort(scores.abs(), descending=True, stable=True)[:k]
                try:
                    metrics.append(
                        _fit_control(validated, train_indices, test_indices, support, fit_labels)
                    )
                except SparseProbeConvergenceError as error:
                    rejections.append(SparseProbeRejection(arm, k, repeat, support, str(error)))
                    continue
                supports.append(support)
            controls.append(_control_result(supports, metrics, k))

    return SparseProbeSweep(
        ks=k_values,
        results=results,
        random_coordinate_controls=tuple(random_controls),
        label_shuffle_controls=tuple(shuffle_controls),
        seed=validated.seed,
        rejections=tuple(rejections),
    )
