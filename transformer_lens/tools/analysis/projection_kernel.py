"""Projection Kernel utilities for comparing linear subspaces.

The Projection Kernel (PK) between subspaces with orthonormal bases ``U`` and
``V`` is ``||U.T @ V||_F^2``. It is invariant to basis choices within either
subspace and equals the sum of squared principal-angle cosines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class SubspaceBasis:
    """An explicitly ranked orthonormal basis extracted from a matrix."""

    basis: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    measured_rank: int
    rtol: float
    threshold: float
    input_shape: Tuple[int, int]

    @property
    def ambient_dim(self) -> int:
        """Dimension of the space containing the subspace."""
        return self.basis.shape[0]


@dataclass(frozen=True)
class ProjectionKernelResult:
    """Projection Kernel score and its principal-angle decomposition."""

    score: torch.Tensor
    normalized: torch.Tensor
    cosines: torch.Tensor
    angles: torch.Tensor
    rank_a: int
    rank_b: int
    ambient_dim: int


@dataclass(frozen=True)
class RandomSubspaceReference:
    """Analytic PK moments for independent random equal-rank subspaces."""

    ambient_dim: int
    rank: int
    mean: float
    variance: float


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a dtype supported by stable SVD on common PyTorch backends."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _validate_rtol(rtol: Optional[float], shape: Tuple[int, int], dtype: torch.dtype) -> float:
    if rtol is None:
        return max(shape) * torch.finfo(dtype).eps
    if isinstance(rtol, bool) or not isinstance(rtol, Real):
        raise ValueError(f"rtol must be a finite non-negative real number, got {rtol!r}")
    value = float(rtol)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"rtol must be a finite non-negative real number, got {rtol!r}")
    return value


def _validate_rank(rank: Optional[int], max_rank: int) -> Optional[int]:
    if rank is None:
        return None
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError(f"rank must be an integer or None, got {rank!r}")
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be between 1 and min(matrix.shape)={max_rank}, got {rank}")
    return rank


def orthonormal_subspace(
    matrix: torch.Tensor,
    *,
    rank: Optional[int] = None,
    rtol: Optional[float] = None,
) -> SubspaceBasis:
    """Extract an explicitly ranked orthonormal column-space basis.

    Low-precision inputs are promoted to float32 before the reduced SVD. With no
    explicit ``rtol``, numerical rank uses ``max(matrix.shape) * eps`` relative
    to the largest singular value. An explicit ``rank`` truncates the measured
    subspace but may not exceed its measured rank.

    Args:
        matrix: Finite floating-point matrix with shape ``[ambient_dim, width]``.
        rank: Optional number of leading singular directions to retain.
        rtol: Optional non-negative relative singular-value threshold.

    Returns:
        Basis, complete singular spectrum, and rank metadata.

    Raises:
        ValueError: If the matrix or rank policy is invalid.
    """
    if not isinstance(matrix, torch.Tensor):
        raise ValueError(f"matrix must be a torch.Tensor, got {type(matrix).__name__}")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got shape {tuple(matrix.shape)}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"matrix dimensions must be non-empty, got shape {tuple(matrix.shape)}")
    if not torch.is_floating_point(matrix):
        raise ValueError(f"matrix must have a real floating-point dtype, got {matrix.dtype}")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("matrix must contain only finite values")

    input_shape = (matrix.shape[0], matrix.shape[1])
    requested_rank = _validate_rank(rank, min(input_shape))
    compute_dtype = _compute_dtype(matrix.dtype)
    effective_rtol = _validate_rtol(rtol, input_shape, compute_dtype)
    work = matrix.to(dtype=compute_dtype)
    left, singular_values, _ = torch.linalg.svd(work, full_matrices=False)
    threshold = float(singular_values[0].item()) * effective_rtol
    measured_rank = int((singular_values > threshold).sum().item())
    if measured_rank == 0:
        raise ValueError(
            "matrix numerical rank is zero "
            f"for shape {input_shape}, dtype {matrix.dtype}, and threshold {threshold:.6g}"
        )
    if requested_rank is not None and requested_rank > measured_rank:
        raise ValueError(
            f"requested rank {requested_rank} exceeds measured rank {measured_rank} "
            f"at threshold {threshold:.6g}"
        )

    selected_rank = measured_rank if requested_rank is None else requested_rank
    return SubspaceBasis(
        basis=left[:, :selected_rank],
        singular_values=singular_values,
        rank=selected_rank,
        measured_rank=measured_rank,
        rtol=effective_rtol,
        threshold=threshold,
        input_shape=input_shape,
    )


def _validate_subspace(value: SubspaceBasis, name: str) -> None:
    if not isinstance(value, SubspaceBasis):
        raise ValueError(f"{name} must be a SubspaceBasis, got {type(value).__name__}")
    basis = value.basis
    if not isinstance(basis, torch.Tensor) or basis.ndim != 2:
        raise ValueError(f"{name}.basis must be a two-dimensional tensor")
    if not torch.is_floating_point(basis) or not bool(torch.isfinite(basis).all()):
        raise ValueError(f"{name}.basis must be finite and real floating-point")
    singular_values = value.singular_values
    if (
        not isinstance(singular_values, torch.Tensor)
        or singular_values.ndim != 1
        or not torch.is_floating_point(singular_values)
        or not bool(torch.isfinite(singular_values).all())
    ):
        raise ValueError(f"{name}.singular_values must be a finite floating-point vector")
    if singular_values.device != basis.device:
        raise ValueError(f"{name}.singular_values must be on the basis device")
    if (
        not isinstance(value.input_shape, tuple)
        or len(value.input_shape) != 2
        or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1
            for dimension in value.input_shape
        )
    ):
        raise ValueError(f"{name}.input_shape must contain two positive dimensions")
    if singular_values.shape[0] != min(value.input_shape):
        raise ValueError(f"{name}.singular_values length must equal min(input_shape)")
    if (
        isinstance(value.rank, bool)
        or not isinstance(value.rank, int)
        or value.rank < 1
        or basis.shape[1] != value.rank
    ):
        raise ValueError(f"{name}.rank must equal its positive basis width")
    if (
        isinstance(value.measured_rank, bool)
        or not isinstance(value.measured_rank, int)
        or value.measured_rank < value.rank
        or value.measured_rank > min(value.input_shape)
    ):
        raise ValueError(f"{name}.measured_rank must be at least its selected rank")
    if value.input_shape[0] != basis.shape[0]:
        raise ValueError(f"{name}.input_shape must share its basis ambient dimension")
    if (
        isinstance(value.rtol, bool)
        or not isinstance(value.rtol, Real)
        or not math.isfinite(value.rtol)
        or value.rtol < 0
    ):
        raise ValueError(f"{name}.rtol must be finite and non-negative")
    if (
        isinstance(value.threshold, bool)
        or not isinstance(value.threshold, Real)
        or not math.isfinite(value.threshold)
        or value.threshold < 0
    ):
        raise ValueError(f"{name}.threshold must be finite and non-negative")


def _clamp_projection_scores(scores: torch.Tensor, upper_bound: int) -> torch.Tensor:
    """Clamp roundoff-scale PK bound violations and reject larger violations."""
    tolerance = 100 * torch.finfo(scores.dtype).eps * max(1, upper_bound)
    minimum = float(scores.detach().min().item())
    maximum = float(scores.detach().max().item())
    if minimum < -tolerance or maximum > upper_bound + tolerance:
        raise ValueError(
            "Projection Kernel score lies outside its theoretical bounds: "
            f"observed [{minimum:.6g}, {maximum:.6g}], expected [0, {upper_bound}] "
            f"within tolerance {tolerance:.6g}"
        )
    return scores.clamp(min=0.0, max=float(upper_bound))


def projection_kernel(
    subspace_a: SubspaceBasis,
    subspace_b: SubspaceBasis,
    *,
    check_orthonormal: bool = True,
) -> ProjectionKernelResult:
    """Measure overlap between two explicitly extracted subspaces.

    Raw PK lies in ``[0, min(rank_a, rank_b)]``. The normalized value is
    ``PK / sqrt(rank_a * rank_b)``, the cosine between the two projection
    matrices. Principal angles are returned in radians.
    """
    if not isinstance(check_orthonormal, bool):
        raise ValueError(f"check_orthonormal must be a Boolean, got {check_orthonormal!r}")
    _validate_subspace(subspace_a, "subspace_a")
    _validate_subspace(subspace_b, "subspace_b")
    if subspace_a.ambient_dim != subspace_b.ambient_dim:
        raise ValueError(
            "subspaces must have equal ambient dimensions, got "
            f"{subspace_a.ambient_dim} and {subspace_b.ambient_dim}"
        )
    if subspace_a.basis.device != subspace_b.basis.device:
        raise ValueError(
            "subspace bases must be on the same device, got "
            f"{subspace_a.basis.device} and {subspace_b.basis.device}"
        )

    dtype = torch.promote_types(subspace_a.basis.dtype, subspace_b.basis.dtype)
    dtype = _compute_dtype(dtype)
    first = subspace_a.basis.to(dtype=dtype)
    second = subspace_b.basis.to(dtype=dtype)
    if check_orthonormal:
        eps = torch.finfo(dtype).eps
        tolerance = 10 * max(first.shape[0], first.shape[1], second.shape[1]) * eps
        first_identity = torch.eye(first.shape[1], dtype=dtype, device=first.device)
        second_identity = torch.eye(second.shape[1], dtype=dtype, device=second.device)
        if not torch.allclose(first.T @ first, first_identity, rtol=tolerance, atol=tolerance):
            raise ValueError("subspace_a basis columns must be orthonormal")
        if not torch.allclose(second.T @ second, second_identity, rtol=tolerance, atol=tolerance):
            raise ValueError("subspace_b basis columns must be orthonormal")

    overlap = first.T @ second
    score = _clamp_projection_scores(overlap.square().sum(), min(subspace_a.rank, subspace_b.rank))
    cosines = torch.linalg.svdvals(overlap)
    angles = torch.acos(cosines.clamp(min=0.0, max=1.0))
    denominator = math.sqrt(subspace_a.rank * subspace_b.rank)
    return ProjectionKernelResult(
        score=score,
        normalized=score / denominator,
        cosines=cosines,
        angles=angles,
        rank_a=subspace_a.rank,
        rank_b=subspace_b.rank,
        ambient_dim=subspace_a.ambient_dim,
    )


def random_projection_kernel_moments(ambient_dim: int, rank: int) -> RandomSubspaceReference:
    """Return PK moments for independent Haar-distributed rank-``rank`` planes.

    These idealized descriptive moments are not calibrated p-values for trained
    model weights, whose head subspaces are dependent and anisotropic.
    """
    if isinstance(ambient_dim, bool) or not isinstance(ambient_dim, int) or ambient_dim < 2:
        raise ValueError(f"ambient_dim must be an integer at least 2, got {ambient_dim!r}")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= ambient_dim:
        raise ValueError(f"rank must be an integer between 1 and {ambient_dim}, got {rank!r}")

    mean = rank**2 / ambient_dim
    variance = (
        2
        * rank**2
        * (ambient_dim - rank) ** 2
        / (ambient_dim**2 * (ambient_dim - 1) * (ambient_dim + 2))
    )
    return RandomSubspaceReference(
        ambient_dim=ambient_dim,
        rank=rank,
        mean=float(mean),
        variance=float(variance),
    )
