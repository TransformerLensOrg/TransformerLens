"""Model-free contracts for Backward Lens gradient factors.

The Backward Lens represents a linear weight gradient as a sum of token-position
outer products. Bridge capture and vocabulary projection are added in later
commits; this module starts with independently testable tensor algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

WeightLayout = Literal["in_out", "out_in"]


@dataclass(frozen=True)
class LinearGradientFactors:
    """Detached factors and reconstruction for one linear weight gradient.

    ``forward_inputs`` and ``output_gradients`` have shapes ``[position, in]``
    and ``[position, out]``. Gradient tensors use the requested storage layout.
    All tensors are cloned to CPU in float32 so the result owns no autograd graph.
    """

    forward_inputs: torch.Tensor
    output_gradients: torch.Tensor
    weight_gradient: torch.Tensor
    reconstructed_gradient: torch.Tensor
    absolute_reconstruction_error: float
    relative_reconstruction_error: float
    weight_layout: WeightLayout


@dataclass(frozen=True)
class VocabularyRanking:
    """Owned CPU copies of signed vocabulary rankings with shape ``[..., k]``.

    ``values`` preserves the floating dtype and sign of ``logits``; ``indices``
    has dtype ``torch.int64``. Both tensors are detached.
    """

    values: torch.Tensor
    indices: torch.Tensor


def _validate_floating_matrix(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank 2; got shape {tuple(tensor.shape)}")
    if 0 in tensor.shape:
        raise ValueError(f"{name} must have no empty dimensions; got shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype; got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _reconstruction_errors(
    reference: torch.Tensor, reconstruction: torch.Tensor
) -> tuple[float, float]:
    """Return max absolute and symmetric scale-aware relative errors.

    The relative error is ``||reference - reconstruction||_F`` divided by the
    maximum of the two input Frobenius norms and float32 epsilon. This remains
    finite when one or both gradients are zero.
    """
    difference = reference - reconstruction
    absolute = float(difference.abs().max())
    scale = torch.maximum(reference.norm(), reconstruction.norm()).clamp_min(
        torch.finfo(reference.dtype).eps
    )
    relative = float(difference.norm() / scale)
    return absolute, relative


def _to_detached_float32(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Detach and convert a validated tensor, rejecting float32 overflow."""
    converted = tensor.detach().float()
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(f"{name} must remain finite when converted to float32")
    return converted


def _build_linear_gradient_factors(
    forward_inputs: torch.Tensor,
    output_gradients: torch.Tensor,
    weight_gradient: torch.Tensor,
    *,
    weight_layout: WeightLayout,
) -> LinearGradientFactors:
    """Reconstruct a weight gradient from aligned token-position factors.

    Args:
        forward_inputs: Linear inputs with shape ``[position, in_features]``.
        output_gradients: Loss gradients with respect to linear outputs, shape
            ``[position, out_features]``.
        weight_gradient: Independently computed gradient in ``weight_layout``.
        weight_layout: ``"in_out"`` for GPT-2 ``Conv1D`` storage or
            ``"out_in"`` for ``torch.nn.Linear`` storage.

    Returns:
        Detached factors, the independent gradient, its reconstruction, and
        reconstruction errors, all on CPU with float32 tensor values.
    """
    _validate_floating_matrix("forward_inputs", forward_inputs)
    _validate_floating_matrix("output_gradients", output_gradients)
    _validate_floating_matrix("weight_gradient", weight_gradient)
    if weight_layout not in ("in_out", "out_in"):
        raise ValueError("weight_layout must be 'in_out' or 'out_in'")
    if forward_inputs.shape[0] != output_gradients.shape[0]:
        raise ValueError(
            "forward_inputs and output_gradients must have the same number of positions; "
            f"got {forward_inputs.shape[0]} and {output_gradients.shape[0]}"
        )
    devices = {forward_inputs.device, output_gradients.device, weight_gradient.device}
    if len(devices) != 1:
        raise ValueError(
            "forward_inputs, output_gradients, and weight_gradient must share a device"
        )

    inputs = _to_detached_float32("forward_inputs", forward_inputs)
    gradients = _to_detached_float32("output_gradients", output_gradients)
    canonical = inputs.T @ gradients
    reconstruction = canonical if weight_layout == "in_out" else canonical.T
    reference = _to_detached_float32("weight_gradient", weight_gradient)
    if not bool(torch.isfinite(reconstruction).all()):
        raise ValueError("the float32 outer-product reconstruction must contain only finite values")
    if reference.shape != reconstruction.shape:
        raise ValueError(
            f"weight_gradient shape {tuple(reference.shape)} does not match the "
            f"{weight_layout} reconstruction shape {tuple(reconstruction.shape)}"
        )
    absolute, relative = _reconstruction_errors(reference, reconstruction)
    return LinearGradientFactors(
        forward_inputs=inputs.cpu().clone(),
        output_gradients=gradients.cpu().clone(),
        weight_gradient=reference.cpu().clone(),
        reconstructed_gradient=reconstruction.cpu().clone(),
        absolute_reconstruction_error=absolute,
        relative_reconstruction_error=relative,
        weight_layout=weight_layout,
    )


def _rank_vocabulary_logits(logits: torch.Tensor, *, k: int, largest: bool) -> VocabularyRanking:
    """Return largest or smallest vocabulary logits and token ids per row.

    Ordering among exactly tied logits is intentionally unspecified and follows
    :func:`torch.topk`.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim not in (1, 2) or logits.shape[-1] == 0:
        raise ValueError(
            f"logits must have shape [vocab] or [position, vocab]; got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must have a floating dtype; got {logits.dtype}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain only finite values")
    if not isinstance(largest, bool):
        raise TypeError(f"largest must be a bool; got {type(largest).__name__}")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= logits.shape[-1]:
        raise ValueError(f"k must be in [1, {logits.shape[-1]}]; got {k!r}")
    ranked = torch.topk(logits.detach(), k=k, dim=-1, largest=largest, sorted=True)
    return VocabularyRanking(
        values=ranked.values.cpu().clone(), indices=ranked.indices.cpu().clone()
    )
