"""Backward Lens gradient-factor contracts and GPT-2 Bridge capture.

The Backward Lens represents a linear weight gradient as a sum of token-position
outer products. This module provides independently testable tensor algebra and
the raw GPT-2 capture path; vocabulary projection is added separately.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

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


@dataclass(frozen=True)
class _GPT2LayerGradientFactors:
    """Detached gradient factors for both MLP projections in one GPT-2 layer."""

    layer: int
    input_projection: LinearGradientFactors
    output_projection: LinearGradientFactors


@dataclass(frozen=True)
class _GPT2GradientCapture:
    """Private Commit-2 result for one GPT-2 next-token loss.

    Tensor fields are detached, owned CPU copies. Public vocabulary-facing result
    contracts are introduced with the projection API.
    """

    prompt_token_ids: torch.Tensor
    target_token_id: int
    loss: float
    layers: tuple[_GPT2LayerGradientFactors, ...]


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


def _validate_requested_layers(model: Any, layers: Sequence[int]) -> tuple[int, ...]:
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise TypeError("layers must be a sequence of integer layer indices")
    requested = tuple(layers)
    if not requested:
        raise ValueError("layers must contain at least one layer index")
    for layer in requested:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"each layer must be an integer; got {layer!r}")
    if len(set(requested)) != len(requested):
        raise ValueError("layers must not contain duplicate indices")
    n_layers = int(model.cfg.n_layers)
    invalid = [layer for layer in requested if not 0 <= layer < n_layers]
    if invalid:
        raise ValueError(f"layers must be in [0, {n_layers - 1}]; got {invalid}")
    return requested


def _require_raw_gpt2_bridge(model: Any) -> None:
    """Require the raw GPT-2 Bridge capabilities used by gradient capture."""
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.model_bridge.supported_architectures.gpt2 import (
        GPT2ArchitectureAdapter,
    )

    if not isinstance(model, TransformerBridge):
        raise TypeError(
            "Backward Lens supports TransformerBridge only; load GPT-2 with "
            "TransformerBridge.boot_transformers(...)."
        )
    if getattr(model, "compatibility_mode", False):
        raise ValueError(
            "Backward Lens requires a raw TransformerBridge; compatibility mode is enabled"
        )
    if getattr(model, "_weights_processed", False):
        raise ValueError(
            "Backward Lens requires original GPT-2 weights; this Bridge processed its weights"
        )
    if not isinstance(model.adapter, GPT2ArchitectureAdapter):
        raise ValueError(
            "Backward Lens currently supports the GPT2ArchitectureAdapter only; "
            f"got {type(model.adapter).__name__}"
        )
    if bool(getattr(model.cfg, "gated_mlp", False)):
        raise ValueError("Backward Lens currently requires dense, non-gated GPT-2 MLPs")
    if model.tokenizer is None:
        raise ValueError("Backward Lens requires a GPT-2 Bridge with a tokenizer")
    for component in ("blocks", "ln_final", "unembed"):
        if not hasattr(model, component):
            raise ValueError(f"Backward Lens requires the standard {component} component")


def _get_gpt2_mlp_projections(model: Any, layers: tuple[int, ...]) -> dict[int, tuple[Any, Any]]:
    """Return validated live GPT-2 Conv1D input/output projection bridges."""
    from transformers.pytorch_utils import Conv1D

    from transformer_lens.hook_points import HookPoint
    from transformer_lens.model_bridge.generalized_components import (
        LinearBridge,
        MLPBridge,
    )

    expected_shapes = (
        (int(model.cfg.d_model), int(model.cfg.d_mlp)),
        (int(model.cfg.d_mlp), int(model.cfg.d_model)),
    )
    projections: dict[int, tuple[Any, Any]] = {}
    for layer in layers:
        mlp = model.blocks[layer].mlp
        if not isinstance(mlp, MLPBridge) or getattr(mlp, "gate", None) is not None:
            raise ValueError(f"layer {layer} must have a dense, non-gated MLPBridge")
        pair = (getattr(mlp, "in", None), getattr(mlp, "out", None))
        for name, projection, expected_shape in zip(
            ("input", "output"), pair, expected_shapes, strict=True
        ):
            if not isinstance(projection, LinearBridge):
                raise ValueError(f"layer {layer} {name} projection must be a LinearBridge")
            if not isinstance(projection.original_component, Conv1D):
                raise ValueError(f"layer {layer} {name} projection must wrap GPT-2 Conv1D")
            weight = projection.original_component.weight
            if not isinstance(weight, torch.nn.Parameter) or not weight.requires_grad:
                raise ValueError(
                    f"layer {layer} {name} original weight must be a trainable Parameter"
                )
            if not weight.is_floating_point() or tuple(weight.shape) != expected_shape:
                raise ValueError(
                    f"layer {layer} {name} weight must have shape {expected_shape} "
                    f"and floating dtype; got {tuple(weight.shape)} and {weight.dtype}"
                )
            if not isinstance(projection.hook_in, HookPoint) or not isinstance(
                projection.hook_out, HookPoint
            ):
                raise ValueError(f"layer {layer} {name} projection is missing Bridge hook points")
        projections[layer] = pair
    return projections


def _capture_once(
    captured: dict[tuple[int, str, str], torch.Tensor], key: tuple[int, str, str]
) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
    """Build a non-modifying PyTorch hook that records one tensor."""

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if key in captured:
            raise RuntimeError(f"Backward Lens hook {key} fired more than once")
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Backward Lens hook {key} returned a non-tensor output")
        captured[key] = output

    return capture


@contextmanager
def _capture_projection_tensors(
    projections: dict[int, tuple[Any, Any]],
) -> Iterator[dict[tuple[int, str, str], torch.Tensor]]:
    """Capture exact linear boundaries while preserving every pre-existing hook."""
    captured: dict[tuple[int, str, str], torch.Tensor] = {}
    handles: list[Any] = []
    try:
        for layer, pair in projections.items():
            for name, projection in zip(("input", "output"), pair, strict=True):
                input_key = (layer, name, "forward_input")
                output_key = (layer, name, "output")
                # Existing hook_in edits must run first so this is the actual linear input.
                handles.append(
                    projection.hook_in.register_forward_hook(_capture_once(captured, input_key))
                )
                # Capture the raw linear output before any existing hook_out edits.
                handles.append(
                    projection.hook_out.register_forward_hook(
                        _capture_once(captured, output_key), prepend=True
                    )
                )
        yield captured
    finally:
        for handle in reversed(handles):
            handle.remove()


def _single_batch_matrix(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(
            f"{name} must have shape [1, position, width]; got {tuple(tensor.shape)}"
        )
    return tensor[0]


@contextmanager
def _preserve_model_rng(model: Any) -> Iterator[None]:
    """Preserve CPU and every CUDA/MPS RNG used by the wrapped model."""
    parameter_devices = {parameter.device for parameter in model.original_model.parameters()}
    cuda_devices = sorted(
        {
            device.index
            for device in parameter_devices
            if device.type == "cuda" and device.index is not None
        }
    )
    uses_mps = any(device.type == "mps" for device in parameter_devices)
    mps_state = torch.mps.get_rng_state() if uses_mps else None
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _capture_gpt2_mlp_gradient_factors(
    model: Any,
    prompt: str,
    target_token: str,
    layers: Sequence[int],
) -> _GPT2GradientCapture:
    """Capture exact GPT-2 MLP weight-gradient factors for one next-token loss.

    The analysis performs one grad-enabled forward and exactly one
    :func:`torch.autograd.grad` call. It does not call ``backward``, touch
    parameter ``.grad`` buffers, change training state, or remove caller hooks.
    """
    _require_raw_gpt2_bridge(model)
    requested_layers = _validate_requested_layers(model, layers)
    projections = _get_gpt2_mlp_projections(model, requested_layers)
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if prompt == "":
        raise ValueError("prompt must not be empty")
    if not isinstance(target_token, str):
        raise TypeError("target_token must be a string")

    prompt_tokens = model.to_tokens(prompt)
    target_tokens = model.to_tokens(target_token, prepend_bos=False)
    if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1 or prompt_tokens.shape[1] == 0:
        raise ValueError("prompt must tokenize to one non-empty sequence")
    if target_tokens.ndim != 2 or tuple(target_tokens.shape) != (1, 1):
        count = int(target_tokens.numel())
        raise ValueError(
            "target_token must encode to exactly one token without BOS; " f"got {count} tokens"
        )
    target_token_id = int(target_tokens.item())
    input_device = next(model.original_model.parameters()).device
    prompt_tokens = prompt_tokens.to(input_device)
    weights = [
        projection.original_component.weight
        for layer in requested_layers
        for projection in projections[layer]
    ]
    with _preserve_model_rng(model), torch.enable_grad():
        with _capture_projection_tensors(projections) as captured:
            logits = model(prompt_tokens)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
                raise RuntimeError(
                    "GPT-2 Bridge must return logits with shape [1, position, vocab]"
                )
            target = torch.tensor([target_token_id], device=logits.device)
            loss = F.cross_entropy(logits[:, -1, :], target)
            if not bool(torch.isfinite(loss)):
                raise ValueError("the next-token loss must be finite")
            outputs = [
                captured[(layer, name, "output")]
                for layer in requested_layers
                for name in ("input", "output")
            ]
            gradients = torch.autograd.grad(loss, (*outputs, *weights), allow_unused=False)

    output_gradients = gradients[: len(outputs)]
    weight_gradients = gradients[len(outputs) :]
    layer_results = []
    for index, layer in enumerate(requested_layers):
        input_offset = 2 * index
        input_factors = _build_linear_gradient_factors(
            _single_batch_matrix(
                f"layer {layer} input projection input",
                captured[(layer, "input", "forward_input")],
            ),
            _single_batch_matrix(
                f"layer {layer} input projection gradient", output_gradients[input_offset]
            ),
            weight_gradients[input_offset],
            weight_layout="in_out",
        )
        output_factors = _build_linear_gradient_factors(
            _single_batch_matrix(
                f"layer {layer} output projection input",
                captured[(layer, "output", "forward_input")],
            ),
            _single_batch_matrix(
                f"layer {layer} output projection gradient",
                output_gradients[input_offset + 1],
            ),
            weight_gradients[input_offset + 1],
            weight_layout="in_out",
        )
        layer_results.append(
            _GPT2LayerGradientFactors(
                layer=layer,
                input_projection=input_factors,
                output_projection=output_factors,
            )
        )
    return _GPT2GradientCapture(
        prompt_token_ids=prompt_tokens.detach().cpu().clone(),
        target_token_id=target_token_id,
        loss=float(loss.detach()),
        layers=tuple(layer_results),
    )
