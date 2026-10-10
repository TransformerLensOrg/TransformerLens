"""Dual-space unembedding geometry and counterfactual concept directions.

The population covariance convention follows Park, Choe and Veitch,
https://arxiv.org/abs/2311.03658v2, Section 3.2, Equation (3.3). Its inverse is
one choice of causal inner product under the paper's assumptions, not an
unconditional guarantee of causal separability. Explicit ridge regularization
changes that metric and does not whiten the original covariance to identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Optional, Sequence, Tuple

import torch
from jaxtyping import Float

_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_COMPUTE_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class _GeometryDiagnostics:
    """Population, precision and conditioning metadata for a detached fit.

    ``token_ids=None`` denotes the complete input vocabulary. Explicit IDs retain
    caller order and have uniform weight without repetitions. ``measured_rank``
    counts covariance eigenvalues strictly above ``threshold``, capped by the
    centered population's structural rank. ``condition_number`` is infinite
    when that numerical rank is deficient. The regularized condition number
    refers to the matrix actually factorized, including any requested ridge.
    """

    input_shape: Tuple[int, int]
    token_ids: Optional[Tuple[int, ...]]
    n_tokens: int
    input_dtype: torch.dtype
    compute_dtype: torch.dtype
    device: torch.device
    centered: bool
    covariance_normalization: Literal["population"]
    ridge: Optional[float]
    rtol: float
    threshold: float
    regularized_threshold: float
    measured_rank: int
    condition_number: float
    regularized_condition_number: float


@dataclass(frozen=True)
class _GeometryFit:
    """Detached tensor snapshot of covariance and its symmetric factors.

    Eigenvalues are ascending and ``whitening`` is the inverse square root of
    ``regularized_covariance``. ``unwhitening`` is its square root. All tensors
    use the compute dtype on the input device, own storage independent of the
    input, and have no autograd history. Frozen fields do not make tensor
    contents immutable; treat the snapshot tensors as read-only.
    """

    mean: Float[torch.Tensor, "model"]
    covariance: Float[torch.Tensor, "model model"]
    regularized_covariance: Float[torch.Tensor, "model model"]
    eigenvalues: Float[torch.Tensor, "model"]
    regularized_eigenvalues: Float[torch.Tensor, "model"]
    eigenvectors: Float[torch.Tensor, "model model"]
    whitening: Float[torch.Tensor, "model model"]
    unwhitening: Float[torch.Tensor, "model model"]
    diagnostics: _GeometryDiagnostics


@dataclass(frozen=True)
class ConceptDirection:
    """An oriented, unnormalized mean of counterfactual token contrasts.

    Each ``(lo, hi)`` pair contributes ``unembedding[:, hi] - unembedding[:, lo]``.
    Measurement fields contain the uniformly weighted mean in raw and whitened
    measurement coordinates. Intervention fields are explicitly metric-derived,
    not independently estimated from contexts or input token embeddings.

    Dispersion is ``mean_i ||difference_i - mean_difference||^2`` in the named
    space, with population normalization. A single pair has zero dispersion;
    this is not a confidence estimate or evidence of a linear concept. Near
    cancellation can have high dispersion even when the mean is nonzero.

    All tensors use the fit dtype/device and are detached, independently owned
    results. Frozen fields prevent reassignment, not in-place tensor mutation;
    treat tensors as read-only. Mutating a result cannot change the geometry.
    Diagnostics describe the metric policy, not a unique model/basis identity.
    """

    pairs: Tuple[Tuple[int, int], ...]
    pair_labels: Optional[Tuple[Tuple[str, str], ...]]
    label: Optional[str]
    raw_measurement: Float[torch.Tensor, "model"]
    whitened_measurement: Float[torch.Tensor, "model"]
    raw_derived_intervention: Float[torch.Tensor, "model"]
    whitened_derived_intervention: Float[torch.Tensor, "model"]
    raw_pair_differences: Float[torch.Tensor, "pair model"]
    whitened_pair_differences: Float[torch.Tensor, "pair model"]
    raw_dispersion: Float[torch.Tensor, ""]
    whitened_dispersion: Float[torch.Tensor, ""]
    geometry_diagnostics: _GeometryDiagnostics
    aggregation: Literal["mean"] = "mean"
    intervention_kind: Literal["metric-derived"] = "metric-derived"

    @property
    def n_pairs(self) -> int:
        """Number of equally weighted, oriented contrast pairs."""
        return len(self.pairs)


def _validate_scalar(value: float, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else not 0 <= result < 1):
        requirement = "positive" if positive else "in [0, 1)"
        raise ValueError(f"{name} must be finite and {requirement}, got {value!r}")
    return result


def _fit_unembedding_geometry(
    unembedding: torch.Tensor,
    *,
    token_ids: Optional[Sequence[int]] = None,
    ridge: Optional[float] = None,
    rtol: Optional[float] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> _GeometryFit:
    """Fit a centered population covariance from ``[d_model, d_vocab]`` weights.

    Tokens in the selected population have uniform weight. No special tokens
    are removed implicitly. Half/bfloat16 inputs promote to float32, float32
    remains float32, and float64 remains float64 unless ``compute_dtype`` is
    explicitly set to float32 or float64. CPU and CUDA are supported without
    implicit device transfer. The input and its autograd state are not mutated.

    The relative covariance eigenvalue threshold defaults to
    ``d_model * finfo(compute_dtype).eps``. Covariance eigenvalues at or below
    ``rtol * largest_eigenvalue`` are numerically deficient. Deficient exact
    fits raise; explicit positive ridge fits must themselves remain above the
    corresponding regularized threshold. Tiny negative covariance eigenvalues
    within the compute error scale are retained, not silently clamped.

    Args:
        unembedding: Finite, real, strided readout matrix in a declared basis.
        token_ids: Non-empty unique vocabulary IDs, or None for all tokens.
        ridge: Explicit positive diagonal covariance shift, or None for exact.
        rtol: Relative rank threshold in [0, 1), or None for the dtype default.
        compute_dtype: Optional float32/float64 accumulation and eigensolver dtype.

    Returns:
        Detached covariance, factors and population/conditioning diagnostics.

    Examples:
        >>> readout = torch.tensor([[-1.0, 1.0]])
        >>> exact = _fit_unembedding_geometry(readout)
        >>> exact.covariance.tolist()
        [[1.0]]
        >>> regularized = _fit_unembedding_geometry(readout, ridge=3.0)
        >>> regularized.whitening.tolist()
        [[0.5]]

    Raises:
        ValueError: For invalid input, unsupported device/dtype/layout,
            non-finite covariance, deficient exact fit, or insufficient ridge.
    """
    if not isinstance(unembedding, torch.Tensor):
        raise ValueError("unembedding must be a torch.Tensor")
    if unembedding.ndim != 2:
        raise ValueError("unembedding must be two-dimensional with shape [d_model, d_vocab]")
    d_model, d_vocab = unembedding.shape
    if d_model == 0 or d_vocab == 0:
        raise ValueError("unembedding dimensions must be non-empty")
    if unembedding.layout != torch.strided:
        raise ValueError("unembedding must have strided layout")
    if unembedding.dtype not in _INPUT_DTYPES:
        raise ValueError("unembedding must have float16, bfloat16, float32 or float64 dtype")
    if unembedding.device.type not in ("cpu", "cuda"):
        raise ValueError("unembedding geometry requires a CPU or CUDA device")
    if not bool(torch.isfinite(unembedding).all()):
        raise ValueError("unembedding must contain only finite values")

    dtype = compute_dtype
    if dtype is None:
        dtype = torch.float64 if unembedding.dtype == torch.float64 else torch.float32
    if dtype not in _COMPUTE_DTYPES:
        raise ValueError("compute_dtype must be float32 or float64")
    effective_rtol = (
        d_model * torch.finfo(dtype).eps
        if rtol is None
        else _validate_scalar(rtol, "rtol", positive=False)
    )
    epsilon = None if ridge is None else _validate_scalar(ridge, "ridge", positive=True)
    if epsilon is not None and epsilon > torch.finfo(dtype).max:
        raise ValueError("ridge must be representable in compute_dtype")

    selected_ids: Optional[Tuple[int, ...]] = None
    if token_ids is not None:
        selected_ids = tuple(token_ids)
        if not selected_ids:
            raise ValueError("token_ids must be non-empty")
        if any(isinstance(token, bool) or not isinstance(token, int) for token in selected_ids):
            raise ValueError("token_ids must contain integer vocabulary IDs")
        if any(token < 0 or token >= d_vocab for token in selected_ids):
            raise ValueError(f"token_ids must lie in [0, {d_vocab})")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("token_ids must be unique for uniform population weighting")

    with torch.no_grad():
        work = unembedding.detach()
        if selected_ids is not None:
            indices = torch.tensor(selected_ids, device=work.device, dtype=torch.long)
            work = work.index_select(1, indices)
        work = work.to(dtype=dtype)
        n_tokens = work.shape[1]
        mean = work.mean(dim=1)
        centered = work - mean[:, None]
        covariance = (centered @ centered.T) / n_tokens
        covariance = covariance * 0.5 + covariance.T * 0.5
        if not bool(torch.isfinite(covariance).all()):
            raise ValueError("computed covariance must be finite; rescale weights or use float64")
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        if not bool(torch.isfinite(eigenvalues).all()) or not bool(
            torch.isfinite(eigenvectors).all()
        ):
            raise ValueError("covariance eigendecomposition must be finite")
        largest = float(eigenvalues[-1].item())
        smallest = float(eigenvalues[0].item())
        error_scale = d_model * torch.finfo(dtype).eps * float(eigenvalues.abs().max().item())
        if smallest < -error_scale:
            raise ValueError(
                "computed covariance is not positive semidefinite at compute precision"
            )
        threshold = effective_rtol * largest
        measured_rank = min(int((eigenvalues > threshold).sum().item()), n_tokens - 1)
        condition_number = largest / smallest if measured_rank == d_model else math.inf
        if epsilon is None and measured_rank < d_model:
            raise ValueError(
                "covariance is rank-deficient or near-singular "
                f"(rank {measured_rank}/{d_model}, threshold {threshold:.6g}); "
                "request an explicit positive ridge to regularize"
            )

        regularized = covariance.clone()
        if epsilon is not None:
            regularized.diagonal().add_(epsilon)
        shifted_eigenvalues = eigenvalues + (0.0 if epsilon is None else epsilon)
        regularized_threshold = effective_rtol * float(shifted_eigenvalues[-1].item())
        if not bool(torch.isfinite(shifted_eigenvalues).all()) or not bool(
            torch.isfinite(regularized).all()
        ):
            raise ValueError("regularized covariance must be finite")
        if float(shifted_eigenvalues[0].item()) <= regularized_threshold:
            raise ValueError("ridge is too small for a stable positive-definite covariance fit")
        roots = shifted_eigenvalues.sqrt()
        whitening = (eigenvectors * roots.reciprocal()[None, :]) @ eigenvectors.T
        unwhitening = (eigenvectors * roots[None, :]) @ eigenvectors.T
        if not bool(torch.isfinite(whitening).all()) or not bool(torch.isfinite(unwhitening).all()):
            raise ValueError("covariance factors must be finite; rescale weights or use float64")

    return _GeometryFit(
        mean=mean,
        covariance=covariance,
        regularized_covariance=regularized,
        eigenvalues=eigenvalues,
        regularized_eigenvalues=shifted_eigenvalues,
        eigenvectors=eigenvectors,
        whitening=whitening,
        unwhitening=unwhitening,
        diagnostics=_GeometryDiagnostics(
            input_shape=(d_model, d_vocab),
            token_ids=selected_ids,
            n_tokens=n_tokens,
            input_dtype=unembedding.dtype,
            compute_dtype=dtype,
            device=unembedding.device,
            centered=True,
            covariance_normalization="population",
            ridge=epsilon,
            rtol=effective_rtol,
            threshold=threshold,
            regularized_threshold=regularized_threshold,
            measured_rank=measured_rank,
            condition_number=condition_number,
            regularized_condition_number=(
                float(shifted_eigenvalues[-1].item()) / float(shifted_eigenvalues[0].item())
            ),
        ),
    )


class RepresentationGeometry:
    """Dual-space geometry fitted from a tensor-level unembedding readout.

    Let ``M = covariance + ridge * I`` (with no shift for an exact fit),
    ``W = M^(-1/2)``, and ``S = M^(1/2)``. For column-vector measurement ``g``
    and intervention ``h``, transformed coordinates are ``W g`` and ``S h``.
    Their pairing is preserved: ``(S h).T @ (W g) = h.T @ g``. The two spaces
    therefore must not use the same coordinate transform.

    Methods accept non-empty real tensors with trailing dimension ``d_model``
    on the fit device. Leading batch dimensions broadcast for binary operations;
    rows are compared elementwise, not as an implicit all-pairs Gram matrix.
    Vectors are converted to the fit's compute dtype without implicit device
    transfer. Gradients flow through vector operations, not through fitted
    weights. Matrix properties return copies of the detached snapshot. The full
    readout is retained as a detached copy in its storage dtype for token
    contrasts, even when covariance uses a selected vocabulary population.

    Inner products and cosines take raw coordinates in the named space, not
    already-whitened vectors. Transform methods are linear and do not subtract
    the token mean. Callers must declare the correct basis/space: bare tensors
    carry no metadata that can detect a semantically mislabeled direction.
    Zero-vector inner products are valid; cosines with any zero direction raise.

    Exact metrics are invariant under consistent invertible dual basis changes.
    An isotropic ridge is invariant under orthogonal changes, but not arbitrary
    invertible changes unless its regularizer is transformed consistently.
    Derived interventions are metric identifications, not evidence of model use.

    Args:
        unembedding: Finite readout tensor with shape ``[d_model, d_vocab]``.
        token_ids: Optional unique token population for uniform covariance.
        ridge: Explicit positive covariance shift, or None for an exact fit.
        rtol: Relative eigenvalue threshold, or None for the compute default.
        compute_dtype: Optional float32/float64 accumulation dtype.

    Examples:
        >>> geometry = RepresentationGeometry(torch.tensor([[-2.0, 2.0]]))
        >>> geometry.whiten_measurement(torch.tensor([2.0])).tolist()
        [1.0]
        >>> geometry.whiten_intervention(torch.tensor([3.0])).tolist()
        [6.0]
        >>> geometry.derived_intervention(torch.tensor([2.0])).tolist()
        [0.5]
    """

    def __init__(
        self,
        unembedding: torch.Tensor,
        *,
        token_ids: Optional[Sequence[int]] = None,
        ridge: Optional[float] = None,
        rtol: Optional[float] = None,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._fit = _fit_unembedding_geometry(
            unembedding,
            token_ids=token_ids,
            ridge=ridge,
            rtol=rtol,
            compute_dtype=compute_dtype,
        )
        self._unembedding = unembedding.detach().clone()

    @property
    def diagnostics(self) -> _GeometryDiagnostics:
        """Return immutable population, precision and conditioning metadata."""
        return self._fit.diagnostics

    @property
    def mean(self) -> Float[torch.Tensor, "model"]:
        """Return a copy of the selected population's token mean."""
        return self._fit.mean.clone()

    @property
    def covariance(self) -> Float[torch.Tensor, "model model"]:
        """Return a copy of the centered, unregularized population covariance."""
        return self._fit.covariance.clone()

    @property
    def regularized_covariance(self) -> Float[torch.Tensor, "model model"]:
        """Return a copy of the positive-definite matrix used for the fit."""
        return self._fit.regularized_covariance.clone()

    def _validate_vector(self, vector: torch.Tensor) -> torch.Tensor:
        if not isinstance(vector, torch.Tensor):
            raise ValueError("vector must be a torch.Tensor")
        dimension = self.diagnostics.input_shape[0]
        if vector.ndim == 0 or vector.shape[-1] != dimension:
            raise ValueError(f"vector must have trailing dimension {dimension}")
        if vector.numel() == 0:
            raise ValueError("vector batches must be non-empty")
        if vector.layout != torch.strided:
            raise ValueError("vector must have strided layout")
        if vector.dtype not in _INPUT_DTYPES:
            raise ValueError("vector must have float16, bfloat16, float32 or float64 dtype")
        if vector.device != self.diagnostics.device:
            raise ValueError("vector must be on the same device as the geometry fit")
        return self._require_finite(vector.to(dtype=self.diagnostics.compute_dtype))

    @staticmethod
    def _require_finite(value: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(value).all()):
            raise ValueError("geometry vectors and results must be finite at compute precision")
        return value

    def _transform(self, vector: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        return self._require_finite(self._validate_vector(vector) @ factor.T)

    def whiten_measurement(self, vector: torch.Tensor) -> torch.Tensor:
        """Map raw measurement rows to whitened rows via ``g @ M^(-1/2)``."""
        return self._transform(vector, self._fit.whitening)

    def unwhiten_measurement(self, vector: torch.Tensor) -> torch.Tensor:
        """Recover raw measurement rows via ``g_whitened @ M^(1/2)``."""
        return self._transform(vector, self._fit.unwhitening)

    def whiten_intervention(self, vector: torch.Tensor) -> torch.Tensor:
        """Map raw intervention rows dually via ``h @ M^(1/2)``."""
        return self._transform(vector, self._fit.unwhitening)

    def unwhiten_intervention(self, vector: torch.Tensor) -> torch.Tensor:
        """Recover raw intervention rows via ``h_whitened @ M^(-1/2)``."""
        return self._transform(vector, self._fit.whitening)

    def derived_intervention(self, measurement: torch.Tensor) -> torch.Tensor:
        """Return ``M^(-1) g`` in raw intervention coordinates.

        This Riesz identification follows Equation (4.1) of
        https://arxiv.org/abs/2311.03658v2 for the exact metric. With ridge it
        uses the regularized metric. Transformed-coordinate equality is a
        construction identity, not an independently estimated intervention.
        """
        return self._transform(self.whiten_measurement(measurement), self._fit.whitening)

    def concept_direction(
        self,
        pairs: Sequence[Sequence[int]],
        *,
        label: Optional[str] = None,
        pair_labels: Optional[Sequence[Sequence[str]]] = None,
    ) -> ConceptDirection:
        """Estimate an oriented measurement mean and its derived intervention.

        ``(lo, hi)`` means ``unembedding[:, hi] - unembedding[:, lo]``. Contrasts
        use the full snapshotted vocabulary; covariance population selection
        does not restrict valid contrast IDs. Centering cancels in differences
        and is not applied again. Aggregation is an equally weighted,
        unnormalized mean, not a unit-length canonical concept representation.

        A single pair is valid. Self-pairs, repeated or reversed duplicates,
        zero contrasts and zero aggregate directions raise rather than being
        silently filtered. Near-canceling nonzero means retain their measured
        dispersion; no arbitrary minimum pair count establishes concept quality.
        Strings are optional metadata, not tokenization inputs. Finite outputs
        must be representable at the declared compute precision.

        Args:
            pairs: Non-empty sequence of two integer vocabulary IDs per pair.
            label: Optional non-empty concept label.
            pair_labels: Optional two non-empty strings per pair in input order.

        Returns:
            Typed, detached directions, contrast differences and population
            dispersion, with frozen pair/label metadata and metric diagnostics.

        Raises:
            ValueError: For malformed/out-of-range IDs, invalid labels,
                duplicates, degenerate contrasts or non-finite computations.

        Examples:
            >>> geometry = RepresentationGeometry(torch.tensor([[0.0, 1.0, 2.0]]))
            >>> concept = geometry.concept_direction([(0, 1), (1, 2)], label="step")
            >>> concept.raw_measurement.tolist()
            [1.0]
            >>> concept.raw_dispersion.item()
            0.0
            >>> concept.n_pairs
            2
        """
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or not pairs:
            raise ValueError("pairs must be a non-empty sequence of token-ID pairs")
        vocabulary_size = self.diagnostics.input_shape[1]
        oriented_pairs: list[Tuple[int, int]] = []
        seen: set[Tuple[int, int]] = set()
        for pair in pairs:
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("each contrast pair must contain exactly two token IDs")
            lo, hi = pair
            if any(isinstance(token, bool) or not isinstance(token, int) for token in (lo, hi)):
                raise ValueError("contrast pairs must contain integer token IDs")
            if not (0 <= lo < vocabulary_size and 0 <= hi < vocabulary_size):
                raise ValueError(f"contrast token IDs must lie in [0, {vocabulary_size})")
            if lo == hi:
                raise ValueError("contrast self-pairs have a zero direction")
            unordered_pair = (min(lo, hi), max(lo, hi))
            if unordered_pair in seen:
                raise ValueError(
                    "contrast pairs must not contain duplicates or reversed duplicates"
                )
            seen.add(unordered_pair)
            oriented_pairs.append((lo, hi))

        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("label must be a non-empty string or None")
        frozen_labels: Optional[Tuple[Tuple[str, str], ...]] = None
        if pair_labels is not None:
            if (
                isinstance(pair_labels, (str, bytes))
                or not isinstance(pair_labels, Sequence)
                or len(pair_labels) != len(oriented_pairs)
            ):
                raise ValueError("pair_labels must have one label pair per contrast pair")
            labels: list[Tuple[str, str]] = []
            for label_pair in pair_labels:
                if (
                    isinstance(label_pair, (str, bytes))
                    or not isinstance(label_pair, Sequence)
                    or len(label_pair) != 2
                    or any(not isinstance(value, str) or not value for value in label_pair)
                ):
                    raise ValueError("each pair_labels entry must contain two non-empty strings")
                labels.append((label_pair[0], label_pair[1]))
            frozen_labels = tuple(labels)

        indices = torch.tensor(oriented_pairs, device=self.diagnostics.device, dtype=torch.long)
        lo_rows = self._unembedding.index_select(1, indices[:, 0]).T.to(
            dtype=self.diagnostics.compute_dtype
        )
        hi_rows = self._unembedding.index_select(1, indices[:, 1]).T.to(
            dtype=self.diagnostics.compute_dtype
        )
        differences = self._require_finite(hi_rows - lo_rows)
        whitened_differences = self.whiten_measurement(differences)
        if bool((differences.abs().amax(dim=-1) == 0).any()) or bool(
            (whitened_differences.abs().amax(dim=-1) == 0).any()
        ):
            raise ValueError("contrast pairs must not have a zero direction at compute precision")
        measurement = self._require_finite(differences.mean(dim=0))
        whitened_measurement = self.whiten_measurement(measurement)
        if bool((measurement.abs().amax() == 0)) or bool((whitened_measurement.abs().amax() == 0)):
            raise ValueError("mean concept direction is zero at compute precision")
        deviations = self._require_finite(differences - measurement)
        whitened_deviations = self.whiten_measurement(deviations)
        raw_dispersion = self._require_finite(deviations.square().sum(dim=-1).mean())
        whitened_dispersion = self._require_finite(whitened_deviations.square().sum(dim=-1).mean())
        derived = self.derived_intervention(measurement)
        return ConceptDirection(
            pairs=tuple(oriented_pairs),
            pair_labels=frozen_labels,
            label=label,
            raw_measurement=measurement,
            whitened_measurement=whitened_measurement,
            raw_derived_intervention=derived,
            whitened_derived_intervention=self.whiten_intervention(derived),
            raw_pair_differences=differences,
            whitened_pair_differences=whitened_differences,
            raw_dispersion=raw_dispersion,
            whitened_dispersion=whitened_dispersion,
            geometry_diagnostics=self.diagnostics,
        )

    def _validate_pair(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self._validate_vector(a)
        b = self._validate_vector(b)
        try:
            torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        except RuntimeError as error:
            raise ValueError("vector batch dimensions must broadcast") from error
        return a, b

    def _inner_product(
        self, a: torch.Tensor, b: torch.Tensor, factor: torch.Tensor
    ) -> torch.Tensor:
        a, b = self._validate_pair(a, b)
        transformed_a = self._require_finite(a @ factor.T)
        transformed_b = self._require_finite(b @ factor.T)
        return self._require_finite((transformed_a * transformed_b).sum(dim=-1))

    def measurement_inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute raw-coordinate ``a.T M^(-1) b`` over broadcast batches."""
        return self._inner_product(a, b, self._fit.whitening)

    def intervention_inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute raw-coordinate ``a.T M b`` over broadcast batches."""
        return self._inner_product(a, b, self._fit.unwhitening)

    def _unit_direction(self, vector: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        scale = vector.abs().amax(dim=-1, keepdim=True)
        if bool((scale == 0).any()):
            raise ValueError("cosine is undefined for a zero direction")
        transformed = self._require_finite((vector / scale) @ factor.T)
        transformed_scale = transformed.abs().amax(dim=-1, keepdim=True)
        if bool((transformed_scale == 0).any()):
            raise ValueError("cosine has a zero transformed direction at compute precision")
        scaled = transformed / transformed_scale
        return self._require_finite(scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True))

    def _cosine(self, a: torch.Tensor, b: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
        a, b = self._validate_pair(a, b)
        a_unit = self._unit_direction(a, factor)
        b_unit = self._unit_direction(b, factor)
        result = self._require_finite((a_unit * b_unit).sum(dim=-1))
        tolerance = 4 * self.diagnostics.input_shape[0] * torch.finfo(result.dtype).eps
        if bool((result.abs() > 1 + tolerance).any()):
            raise ValueError("cosine exceeds its valid range at compute precision")
        return result.clamp(-1.0, 1.0)

    def measurement_cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute measurement-metric cosine; reject zero directions.

        Positive rescaling avoids norm overflow/underflow. Final values are
        clamped to [-1, 1] only within the accumulation roundoff bound.
        """
        return self._cosine(a, b, self._fit.whitening)

    def intervention_cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute intervention-metric cosine with the same zero/range policy."""
        return self._cosine(a, b, self._fit.unwhitening)
