"""Projection Kernel utilities for comparing linear subspaces.

The Projection Kernel (PK) between subspaces with orthonormal bases ``U`` and
``V`` is ``||U.T @ V||_F^2``. It is invariant to basis choices within either
subspace and equals the sum of squared principal-angle cosines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, List, Literal, Optional, Sequence, Tuple, cast

import torch

AttentionRole = Literal["Q", "K", "V", "O"]
LayerOrder = Literal["forward", "all"]
HeadKind = Literal["query", "kv"]

_PAIRWISE_TEMP_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SubspaceBasis:
    """An explicitly ranked orthonormal basis extracted from a matrix.

    Attributes:
        basis: Orthonormal column-space basis, ``[ambient_dim, rank]``.
        singular_values: All reduced-SVD singular values, in descending order.
        rank: Number of retained basis directions.
        measured_rank: Numerical rank before optional caller truncation.
        rtol: Effective relative rank tolerance.
        threshold: Absolute singular-value threshold used for rank measurement.
        input_shape: Shape of the matrix from which the basis was extracted.
    """

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


@dataclass(frozen=True)
class AttentionHeadRef:
    """Structured identity for one attention-head weight subspace."""

    layer: int
    head: int
    role: AttentionRole
    kind: HeadKind

    @property
    def label(self) -> str:
        """Return the conventional TransformerLens layer/head label."""
        return f"L{self.layer}H{self.head}"


@dataclass(frozen=True)
class HeadAffinityPair:
    """One ranked source-target head pair."""

    source: AttentionHeadRef
    target: AttentionHeadRef
    score: float
    normalized: float


@dataclass(frozen=True)
class HeadAffinityResult:
    """Projection Kernel affinities between two attention-head roles.

    Score tensors have shape
    ``[source_layer, source_head, target_layer, target_head]``. Layer index
    tuples map tensor positions to original model block numbers.
    """

    scores: torch.Tensor
    normalized: torch.Tensor
    valid_mask: torch.Tensor
    source_role: AttentionRole
    target_role: AttentionRole
    source_layer_indices: Tuple[int, ...]
    target_layer_indices: Tuple[int, ...]
    source_head_kind: HeadKind
    target_head_kind: HeadKind
    source_ranks: torch.Tensor
    target_ranks: torch.Tensor
    source_rank: int
    target_rank: int
    rank: Optional[int]
    rtol: float

    def top_pairs(self, k: int = 20, *, normalized: bool = False) -> List[HeadAffinityPair]:
        """Return the highest-scoring valid pairs with deterministic tie order."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")

        selected_scores = self.normalized if normalized else self.scores
        entries: List[HeadAffinityPair] = []
        for source_layer, source_head, target_layer, target_head in (
            torch.nonzero(self.valid_mask, as_tuple=False).cpu().tolist()
        ):
            source = AttentionHeadRef(
                layer=self.source_layer_indices[source_layer],
                head=source_head,
                role=self.source_role,
                kind=self.source_head_kind,
            )
            target = AttentionHeadRef(
                layer=self.target_layer_indices[target_layer],
                head=target_head,
                role=self.target_role,
                kind=self.target_head_kind,
            )
            entries.append(
                HeadAffinityPair(
                    source=source,
                    target=target,
                    score=float(self.scores[source_layer, source_head, target_layer, target_head]),
                    normalized=float(
                        self.normalized[source_layer, source_head, target_layer, target_head]
                    ),
                )
            )

        def sort_key(pair: HeadAffinityPair) -> Tuple[float, int, int, int, int]:
            value = pair.normalized if normalized else pair.score
            return (
                -value,
                pair.source.layer,
                pair.source.head,
                pair.target.layer,
                pair.target.head,
            )

        entries.sort(key=sort_key)
        return entries[:k]


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a dtype supported by stable SVD on common PyTorch backends."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def _rank_tolerance_dtype(dtypes: Sequence[torch.dtype]) -> torch.dtype:
    """Return the least precise storage dtype for a shared rank tolerance."""
    return max(dtypes, key=lambda dtype: torch.finfo(dtype).eps)


def _validate_rtol(rtol: Optional[float], shape: Tuple[int, int], dtype: torch.dtype) -> float:
    if rtol is None:
        compute_epsilon = torch.finfo(_compute_dtype(dtype)).eps
        storage_epsilon = torch.finfo(dtype).eps
        return max(max(shape) * compute_epsilon, storage_epsilon)
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
    explicit ``rtol``, numerical rank uses the larger of the compute-SVD error
    scale and one input-storage epsilon, relative to the largest singular value.
    An explicit ``rank`` truncates the measured subspace but may not exceed its
    measured rank.

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
    effective_rtol = _validate_rtol(rtol, input_shape, matrix.dtype)
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


def _clamp_principal_cosines(cosines: torch.Tensor) -> torch.Tensor:
    """Clamp roundoff-scale cosine violations and reject larger violations."""
    tolerance = 100 * torch.finfo(cosines.dtype).eps * max(1, cosines.numel())
    minimum = float(cosines.detach().min().item())
    maximum = float(cosines.detach().max().item())
    if minimum < -tolerance or maximum > 1 + tolerance:
        raise ValueError(
            "Principal-angle cosine lies outside its theoretical bounds: "
            f"observed [{minimum:.6g}, {maximum:.6g}], expected [0, 1] "
            f"within tolerance {tolerance:.6g}"
        )
    return cosines.clamp(min=0.0, max=1.0)


def _singular_values(matrix: torch.Tensor) -> torch.Tensor:
    """Compute singular values with an explicit MPS CPU fallback."""
    if matrix.device.type == "mps":
        return torch.linalg.svdvals(matrix.cpu()).to(device=matrix.device)
    return torch.linalg.svdvals(matrix)


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
    cosines = _clamp_principal_cosines(_singular_values(overlap))
    angles = torch.acos(cosines)
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


def _pairwise_projection_kernel(
    source_bases: torch.Tensor,
    target_bases: torch.Tensor,
    *,
    max_temp_bytes: int = _PAIRWISE_TEMP_BYTES,
) -> torch.Tensor:
    """Compute pairwise PK scores while bounding the overlap-tensor allocation."""
    if source_bases.ndim != 3 or target_bases.ndim != 3:
        raise ValueError("source_bases and target_bases must be three-dimensional")
    if source_bases.shape[1] != target_bases.shape[1]:
        raise ValueError("source_bases and target_bases must share an ambient dimension")
    if source_bases.device != target_bases.device or source_bases.dtype != target_bases.dtype:
        raise ValueError("source_bases and target_bases must share a device and dtype")
    if (
        isinstance(max_temp_bytes, bool)
        or not isinstance(max_temp_bytes, int)
        or max_temp_bytes < 1
    ):
        raise ValueError("max_temp_bytes must be a positive integer")

    source_count, _, source_rank = source_bases.shape
    target_count, _, target_rank = target_bases.shape
    scores = torch.empty(
        source_count, target_count, dtype=source_bases.dtype, device=source_bases.device
    )
    bytes_per_overlap = source_rank * target_rank * source_bases.element_size()
    pair_capacity = max(1, max_temp_bytes // bytes_per_overlap)
    source_tile = max(1, min(source_count, pair_capacity // max(1, target_count)))
    target_tile = max(1, min(target_count, pair_capacity // source_tile))

    for source_start in range(0, source_count, source_tile):
        source_stop = min(source_start + source_tile, source_count)
        for target_start in range(0, target_count, target_tile):
            target_stop = min(target_start + target_tile, target_count)
            overlap = torch.einsum(
                "adr,bds->abrs",
                source_bases[source_start:source_stop],
                target_bases[target_start:target_stop],
            )
            scores[source_start:source_stop, target_start:target_stop] = overlap.square().sum(
                dim=(-2, -1)
            )
    return scores


def _architecture_name(model: Any) -> str:
    cfg = getattr(model, "cfg", None)
    architecture = getattr(cfg, "original_architecture", None)
    return str(architecture) if architecture is not None else type(model).__name__


def _read_role_matrix(model: Any, block: Any, layer: int, role: AttentionRole) -> torch.Tensor:
    attn = getattr(block, "attn", None)
    if attn is None:
        raise ValueError(f"attention block {layer} does not expose an attn component")
    attribute = f"W_{role}"
    try:
        matrix = getattr(attn, attribute)
    except NotImplementedError as error:
        raise NotImplementedError(
            f"{_architecture_name(model)} cannot expose role {role} at layer {layer}: {error}"
        ) from error
    except (AttributeError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"{_architecture_name(model)} cannot expose role {role} at layer {layer}: {error}"
        ) from error
    if not isinstance(matrix, torch.Tensor):
        raise ValueError(
            f"role {role} at layer {layer} must be a tensor, got {type(matrix).__name__}"
        )
    if matrix.ndim != 3:
        raise ValueError(
            f"role {role} at layer {layer} expected a three-dimensional per-head weight, "
            f"got shape {tuple(matrix.shape)}"
        )
    if role == "O":
        matrix = matrix.transpose(-2, -1)
    if not torch.is_floating_point(matrix):
        raise ValueError(f"role {role} at layer {layer} must have a floating-point dtype")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"role {role} at layer {layer} must contain only finite values")
    return matrix.detach()


def _validate_role_shapes(
    matrices: Sequence[torch.Tensor], layers: Sequence[int], role: AttentionRole
) -> Tuple[int, int, int]:
    expected = (matrices[0].shape[0], matrices[0].shape[1], matrices[0].shape[2])
    if min(expected) < 1:
        raise ValueError(
            f"role {role} at layer {layers[0]} must have non-empty head, d_model, and width "
            f"dimensions, got shape {expected}"
        )
    for matrix, layer in zip(matrices[1:], layers[1:]):
        if tuple(matrix.shape) != expected:
            raise ValueError(
                f"role {role} at layer {layer} has shape {tuple(matrix.shape)}, expected "
                f"the consistent [heads, d_model, width] shape {expected}"
            )
    return expected


def _extract_bases(
    matrices: Sequence[torch.Tensor],
    layers: Sequence[int],
    role: AttentionRole,
    *,
    selected_rank: int,
    rtol: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_bases: List[torch.Tensor] = []
    layer_ranks: List[List[int]] = []
    for matrix, layer in zip(matrices, layers):
        head_bases: List[torch.Tensor] = []
        head_ranks: List[int] = []
        for head in range(matrix.shape[0]):
            try:
                subspace = orthonormal_subspace(
                    matrix[head].to(dtype=dtype), rank=selected_rank, rtol=rtol
                )
            except ValueError as error:
                raise ValueError(
                    f"Could not extract role {role} at layer {layer}, head {head}: {error}"
                ) from error
            head_bases.append(subspace.basis.to(device=device))
            head_ranks.append(subspace.measured_rank)
        layer_bases.append(torch.stack(head_bases))
        layer_ranks.append(head_ranks)
    return torch.stack(layer_bases), torch.tensor(layer_ranks, dtype=torch.long, device=device)


def attention_head_subspace_affinity(
    model: Any,
    *,
    source_role: str = "O",
    target_role: str,
    layer_order: str = "forward",
    rank: Optional[int] = None,
    rtol: Optional[float] = None,
) -> HeadAffinityResult:
    """Compute OQ, OK, or OV Projection Kernel affinities for a TransformerBridge.

    K/V axes preserve native key-value heads on grouped-query attention models;
    they are never expanded to query-head count. Hybrid models include only
    attention blocks and report their original block indices.

    Args:
        model: A TransformerBridge exposing readable per-head attention weights.
        source_role: Source role; v1 supports only ``"O"``.
        target_role: One of ``"Q"``, ``"K"``, or ``"V"``.
        layer_order: ``"forward"`` keeps strict earlier-to-later pairs; ``"all"``
            keeps every pair.
        rank: Optional common truncation rank. By default every head must be full
            column rank.
        rtol: Optional relative numerical-rank tolerance.

    Returns:
        Affinity tensors, validity mask, original layer indices, and rank metadata.
    """
    if source_role != "O":
        raise ValueError(f"source_role must be 'O' in v1, got {source_role!r}")
    if target_role not in ("Q", "K", "V"):
        raise ValueError(f"target_role must be one of ['Q', 'K', 'V'], got {target_role!r}")
    if layer_order not in ("forward", "all"):
        raise ValueError(f"layer_order must be one of ['forward', 'all'], got {layer_order!r}")
    validated_source_role = cast(AttentionRole, source_role)
    validated_target_role = cast(AttentionRole, target_role)
    validated_layer_order = cast(LayerOrder, layer_order)
    blocks_with = getattr(model, "blocks_with", None)
    if not callable(blocks_with):
        raise ValueError("model must be a TransformerBridge exposing blocks_with('attn')")
    attention_blocks = list(blocks_with("attn"))
    if not attention_blocks:
        raise ValueError("No attention layers found — cannot compute head subspace affinity.")

    layer_indices = [int(layer) for layer, _ in attention_blocks]
    source_matrices = [
        _read_role_matrix(model, block, layer, validated_source_role)
        for layer, block in attention_blocks
    ]
    target_matrices = [
        _read_role_matrix(model, block, layer, validated_target_role)
        for layer, block in attention_blocks
    ]
    source_heads, source_ambient, source_width = _validate_role_shapes(
        source_matrices, layer_indices, validated_source_role
    )
    target_heads, target_ambient, target_width = _validate_role_shapes(
        target_matrices, layer_indices, validated_target_role
    )
    if source_ambient != target_ambient:
        raise ValueError(
            f"roles {validated_source_role} and {validated_target_role} must share d_model, got "
            f"{source_ambient} and {target_ambient}"
        )

    matrices = source_matrices + target_matrices
    tolerance_dtype = _rank_tolerance_dtype([matrix.dtype for matrix in matrices])
    dtype = matrices[0].dtype
    for matrix in matrices[1:]:
        dtype = torch.promote_types(dtype, matrix.dtype)
    dtype = _compute_dtype(dtype)
    effective_rtol = _validate_rtol(
        rtol, (source_ambient, max(source_width, target_width)), tolerance_dtype
    )
    requested_rank = _validate_rank(rank, min(source_ambient, source_width, target_width))
    source_rank = source_width if requested_rank is None else requested_rank
    target_rank = target_width if requested_rank is None else requested_rank

    cfg = getattr(model, "cfg", None)
    configured_device = getattr(cfg, "device", None)
    result_device = (
        torch.device(configured_device)
        if configured_device is not None
        else source_matrices[0].device
    )
    source_bases, source_ranks = _extract_bases(
        source_matrices,
        layer_indices,
        validated_source_role,
        selected_rank=source_rank,
        rtol=effective_rtol,
        dtype=dtype,
        device=result_device,
    )
    target_bases, target_ranks = _extract_bases(
        target_matrices,
        layer_indices,
        validated_target_role,
        selected_rank=target_rank,
        rtol=effective_rtol,
        dtype=dtype,
        device=result_device,
    )

    layer_count = len(layer_indices)
    flat_source = source_bases.reshape(layer_count * source_heads, source_ambient, source_rank)
    flat_target = target_bases.reshape(layer_count * target_heads, target_ambient, target_rank)
    scores = _pairwise_projection_kernel(flat_source, flat_target).reshape(
        layer_count, source_heads, layer_count, target_heads
    )
    scores = _clamp_projection_scores(scores, min(source_rank, target_rank))
    normalized = scores / math.sqrt(source_rank * target_rank)
    if validated_layer_order == "forward":
        layer_tensor = torch.tensor(layer_indices, device=result_device)
        layer_mask = layer_tensor[:, None] < layer_tensor[None, :]
        valid_mask = layer_mask[:, None, :, None].expand_as(scores).contiguous()
    else:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
    normalized = torch.where(valid_mask, normalized, torch.zeros_like(normalized))

    target_kind: HeadKind = "query" if validated_target_role == "Q" else "kv"
    return HeadAffinityResult(
        scores=scores,
        normalized=normalized,
        valid_mask=valid_mask,
        source_role=validated_source_role,
        target_role=validated_target_role,
        source_layer_indices=tuple(layer_indices),
        target_layer_indices=tuple(layer_indices),
        source_head_kind="query",
        target_head_kind=target_kind,
        source_ranks=source_ranks,
        target_ranks=target_ranks,
        source_rank=source_rank,
        target_rank=target_rank,
        rank=rank,
        rtol=effective_rtol,
    )
