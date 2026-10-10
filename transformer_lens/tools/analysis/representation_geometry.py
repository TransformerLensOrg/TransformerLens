"""Centered unembedding covariance and explicit inverse-square-root geometry.

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
