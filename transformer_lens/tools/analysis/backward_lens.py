"""Backward Lens gradient-factor capture and vocabulary projection.

The Backward Lens represents a linear weight gradient as a sum of token-position
outer products and projects residual-width factors into the model vocabulary.
The public API currently supports raw GPT-2 ``TransformerBridge`` models.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

WeightLayout = Literal["in_out", "out_in"]
ProjectedFactor = Literal["forward_inputs", "output_gradients"]
DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class LinearGradientFactors:
    """Detached factors and reconstruction for one linear weight gradient.

    ``forward_inputs`` and ``output_gradients`` have shapes ``[position, in]``
    and ``[position, out]``. ``output_gradients`` and ``weight_gradient`` preserve
    the raw ``d(loss)/d(tensor)`` sign; they are not negated into update directions.
    Gradient tensors use the requested storage layout. All tensors are cloned to
    CPU in float32 so the result owns no autograd graph.
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
class BackwardLensMatrixResult:
    """Factors and vocabulary readouts for one GPT-2 MLP weight matrix.

    ``factors`` contains the full linear factorization. ``projected_factor`` says
    whether its residual-width ``forward_inputs`` or raw-gradient
    ``output_gradients`` were decoded. ``factor_norms`` and ``zero_norm_mask``
    have shape ``[position]`` with float32 and bool dtypes. Largest and smallest
    signed rankings are always retained. Full ``vocabulary_logits`` are present
    only when explicitly requested. Normalized rankings and optional full logits
    are present when the Normalized Logit Lens is requested. Every retained tensor
    is a detached CPU-owned value; gradient descent subtracts raw gradients.
    """

    factors: LinearGradientFactors
    projected_factor: ProjectedFactor
    factor_norms: torch.Tensor
    zero_norm_mask: torch.Tensor
    vocabulary_size: int
    target_token_id: int
    top_ranking: VocabularyRanking
    bottom_ranking: VocabularyRanking
    target_largest_ranks: torch.Tensor
    target_smallest_ranks: torch.Tensor
    normalized_top_ranking: VocabularyRanking | None = None
    normalized_bottom_ranking: VocabularyRanking | None = None
    normalized_target_largest_ranks: torch.Tensor | None = None
    normalized_target_smallest_ranks: torch.Tensor | None = None
    vocabulary_logits: torch.Tensor | None = None
    normalized_vocabulary_logits: torch.Tensor | None = None

    def logits(self, *, normalized: bool = False) -> torch.Tensor:
        """Return opted-in raw or Normalized Logit Lens full logits."""
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a bool")
        if normalized and self.normalized_top_ranking is None:
            raise ValueError("normalized logits were not requested during analysis")
        logits = self.normalized_vocabulary_logits if normalized else self.vocabulary_logits
        if logits is None:
            kind = "normalized " if normalized else ""
            raise ValueError(
                f"full {kind}logits were not retained; call "
                "BackwardLens.analyze(..., return_full_logits=True)"
            )
        return logits

    def _retained_ranking(self, *, normalized: bool, largest: bool) -> VocabularyRanking:
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a bool")
        if normalized:
            ranking = self.normalized_top_ranking if largest else self.normalized_bottom_ranking
            if ranking is None:
                raise ValueError("normalized logits were not requested during analysis")
            return ranking
        return self.top_ranking if largest else self.bottom_ranking

    def top(self, *, k: int, normalized: bool = False) -> VocabularyRanking:
        """Return up to the retained largest signed logits and token ids."""
        return _slice_vocabulary_ranking(
            self._retained_ranking(normalized=normalized, largest=True), k=k
        )

    def bottom(self, *, k: int, normalized: bool = False) -> VocabularyRanking:
        """Return up to the retained smallest signed logits and token ids."""
        return _slice_vocabulary_ranking(
            self._retained_ranking(normalized=normalized, largest=False), k=k
        )

    def top_tokens(self, tokenizer: Any, *, k: int, normalized: bool = False) -> list[list[str]]:
        """Decode the largest-``k`` vocabulary ids for every position."""
        return _decode_vocabulary_ranking(self.top(k=k, normalized=normalized), tokenizer)

    def bottom_tokens(self, tokenizer: Any, *, k: int, normalized: bool = False) -> list[list[str]]:
        """Decode the smallest-``k`` vocabulary ids for every position."""
        return _decode_vocabulary_ranking(self.bottom(k=k, normalized=normalized), tokenizer)

    def target_ranks(
        self,
        target_token_id: int,
        *,
        largest: bool,
        normalized: bool = False,
    ) -> torch.Tensor:
        """Return zero-based target ranks per position in the requested ordering.

        ``largest=True`` gives rank zero to the largest logit. ``largest=False``
        gives rank zero to the smallest, which is the useful raw-gradient
        convention for the second MLP matrix because gradient descent subtracts it.
        Ties receive the same competition rank. The analyzed target's ranks are
        always retained; other token ids require opted-in full logits.
        """
        if isinstance(target_token_id, bool) or not isinstance(target_token_id, int):
            raise TypeError("target_token_id must be an integer")
        if not 0 <= target_token_id < self.vocabulary_size:
            raise ValueError(
                f"target_token_id must be in [0, {self.vocabulary_size - 1}]; "
                f"got {target_token_id}"
            )
        if not isinstance(largest, bool):
            raise TypeError("largest must be a bool")
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a bool")
        if normalized:
            ranks = (
                self.normalized_target_largest_ranks
                if largest
                else self.normalized_target_smallest_ranks
            )
            if ranks is None:
                raise ValueError("normalized logits were not requested during analysis")
        else:
            ranks = self.target_largest_ranks if largest else self.target_smallest_ranks
        if target_token_id == self.target_token_id:
            return ranks.clone()
        return _target_vocabulary_ranks(
            self.logits(normalized=normalized),
            target_token_id=target_token_id,
            largest=largest,
        )

    def gradient_descent_target_ranks(
        self, target_token_id: int, *, normalized: bool = False
    ) -> torch.Tensor:
        """Return ascending raw-gradient target ranks (rank zero is smallest)."""
        return self.target_ranks(target_token_id, largest=False, normalized=normalized)


@dataclass(frozen=True)
class BackwardLensLayerResult:
    """Vocabulary-facing input/output MLP matrix results for one indexed layer."""

    layer: int
    input_projection: BackwardLensMatrixResult
    output_projection: BackwardLensMatrixResult


@dataclass(frozen=True)
class BackwardLensResult:
    """Detached result of one :meth:`BackwardLens.analyze` call.

    ``prompt`` and ``target_token`` echo the analyzed inputs; ``target_token_id``
    is the single vocabulary id the target text encodes to. ``loss`` is the raw
    scalar cross-entropy of the final-position next-token prediction against the
    target; it preserves the ``d(loss)/d(...)`` sign convention and is not negated.
    ``prompt_token_ids`` is an owned CPU int64 tensor with shape ``[position]``;
    position zero is the prepended BOS, and every residual-width factor in
    ``layers`` is aligned to these same positions. ``layers`` preserves requested
    order. Maximum errors summarize both matrices over every requested layer.
    ``includes_normalized_logits`` records whether the Normalized Logit Lens was
    computed. ``includes_full_logits`` records whether full vocabulary tensors
    were retained in addition to bounded rankings. No model or tokenizer reference
    is retained.
    """

    prompt: str
    prompt_token_ids: torch.Tensor
    target_token: str
    target_token_id: int
    loss: float
    layers: tuple[BackwardLensLayerResult, ...]
    max_absolute_reconstruction_error: float
    max_relative_reconstruction_error: float
    includes_normalized_logits: bool
    includes_full_logits: bool

    def layer(self, layer: int) -> BackwardLensLayerResult:
        """Return one requested layer result or raise ``KeyError``."""
        for result in self.layers:
            if result.layer == layer:
                return result
        raise KeyError(f"layer {layer} was not analyzed")


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


def _slice_vocabulary_ranking(ranking: VocabularyRanking, *, k: int) -> VocabularyRanking:
    """Return an owned prefix of an already sorted vocabulary ranking."""
    retained = ranking.indices.shape[-1]
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= retained:
        raise ValueError(f"k must be in [1, retained top_k={retained}]; got {k!r}")
    return VocabularyRanking(
        values=ranking.values[..., :k].clone(),
        indices=ranking.indices[..., :k].clone(),
    )


def _target_vocabulary_ranks(
    logits: torch.Tensor, *, target_token_id: int, largest: bool
) -> torch.Tensor:
    """Return zero-based competition ranks for one vocabulary id per row."""
    target = logits[:, target_token_id].unsqueeze(-1)
    comparisons = logits > target if largest else logits < target
    return comparisons.sum(dim=-1, dtype=torch.int64).cpu().clone()


def _decode_vocabulary_ranking(ranking: VocabularyRanking, tokenizer: Any) -> list[list[str]]:
    """Decode a two-dimensional vocabulary ranking without retaining a tokenizer."""
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer must provide a callable decode method")
    if ranking.indices.ndim != 2:
        raise ValueError("decoded matrix rankings must have shape [position, k]")
    return [[str(decode([token_id])) for token_id in row.tolist()] for row in ranking.indices]


@torch.no_grad()
def _project_residual_factors(model: Any, factors: torch.Tensor) -> torch.Tensor:
    """Apply fresh final normalization and unembedding to residual-width rows."""
    _validate_floating_matrix("factors", factors)
    if factors.shape[-1] != int(model.cfg.d_model):
        raise ValueError(
            f"factors must have width d_model={model.cfg.d_model}; got {factors.shape[-1]}"
        )
    unembed_weight = model.W_U
    if not isinstance(unembed_weight, torch.Tensor) or unembed_weight.ndim != 2:
        raise ValueError("the GPT-2 Bridge must expose a rank-2 unembedding weight")
    batched = (
        factors.detach().to(device=unembed_weight.device, dtype=unembed_weight.dtype).unsqueeze(0)
    )
    logits = model.unembed(model.ln_final(batched)).squeeze(0)
    if logits.ndim != 2 or logits.shape != (factors.shape[0], unembed_weight.shape[1]):
        raise RuntimeError(
            "final normalization and unembedding must return [position, d_vocab]; "
            f"got {tuple(logits.shape)}"
        )
    projected = logits.detach().float()
    if not bool(torch.isfinite(projected).all()):
        raise ValueError("vocabulary projection must remain finite in float32")
    return projected.clone()


def _factor_norms_and_normalized_rows(
    factors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return original L2 norms, exact-zero mask, and safely unit-normalized rows."""
    _validate_floating_matrix("factors", factors)
    rows = _to_detached_float32("factors", factors).cpu().clone()
    norms = rows.norm(dim=-1)
    zero_mask = norms == 0
    denominators = torch.where(zero_mask, torch.ones_like(norms), norms)
    normalized = rows / denominators.unsqueeze(-1)
    return norms.clone(), zero_mask.clone(), normalized


def _build_matrix_result(
    model: Any,
    factors: LinearGradientFactors,
    *,
    projected_factor: ProjectedFactor,
    include_normalized_logits: bool,
    target_token_id: int,
    top_k: int,
    return_full_logits: bool,
) -> BackwardLensMatrixResult:
    if projected_factor not in ("forward_inputs", "output_gradients"):
        raise ValueError("projected_factor must be 'forward_inputs' or 'output_gradients'")
    if not isinstance(include_normalized_logits, bool):
        raise TypeError("include_normalized_logits must be a bool")
    if not isinstance(return_full_logits, bool):
        raise TypeError("return_full_logits must be a bool")
    rows = (
        factors.forward_inputs if projected_factor == "forward_inputs" else factors.output_gradients
    )
    norms, zero_mask, normalized_rows = _factor_norms_and_normalized_rows(rows)
    raw_logits = _project_residual_factors(model, rows)
    normalized_logits = (
        _project_residual_factors(model, normalized_rows) if include_normalized_logits else None
    )
    top_ranking = _rank_vocabulary_logits(raw_logits, k=top_k, largest=True)
    bottom_ranking = _rank_vocabulary_logits(raw_logits, k=top_k, largest=False)
    target_largest_ranks = _target_vocabulary_ranks(
        raw_logits, target_token_id=target_token_id, largest=True
    )
    target_smallest_ranks = _target_vocabulary_ranks(
        raw_logits, target_token_id=target_token_id, largest=False
    )
    normalized_top_ranking = None
    normalized_bottom_ranking = None
    normalized_target_largest_ranks = None
    normalized_target_smallest_ranks = None
    if normalized_logits is not None:
        normalized_top_ranking = _rank_vocabulary_logits(normalized_logits, k=top_k, largest=True)
        normalized_bottom_ranking = _rank_vocabulary_logits(
            normalized_logits, k=top_k, largest=False
        )
        normalized_target_largest_ranks = _target_vocabulary_ranks(
            normalized_logits, target_token_id=target_token_id, largest=True
        )
        normalized_target_smallest_ranks = _target_vocabulary_ranks(
            normalized_logits, target_token_id=target_token_id, largest=False
        )
    return BackwardLensMatrixResult(
        factors=factors,
        projected_factor=projected_factor,
        factor_norms=norms,
        zero_norm_mask=zero_mask,
        vocabulary_size=raw_logits.shape[-1],
        target_token_id=target_token_id,
        top_ranking=top_ranking,
        bottom_ranking=bottom_ranking,
        target_largest_ranks=target_largest_ranks,
        target_smallest_ranks=target_smallest_ranks,
        normalized_top_ranking=normalized_top_ranking,
        normalized_bottom_ranking=normalized_bottom_ranking,
        normalized_target_largest_ranks=normalized_target_largest_ranks,
        normalized_target_smallest_ranks=normalized_target_smallest_ranks,
        vocabulary_logits=raw_logits.cpu().clone() if return_full_logits else None,
        normalized_vocabulary_logits=(
            normalized_logits.cpu().clone()
            if return_full_logits and normalized_logits is not None
            else None
        ),
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
    if int(model.cfg.n_devices) > 1:
        raise ValueError(
            "Backward Lens requires a single-device TransformerBridge because the MLP "
            "projections, final normalization, and unembed must be co-located; "
            f"device-map dispatch with cfg.n_devices={model.cfg.n_devices} is not supported"
        )
    if not isinstance(model.adapter, GPT2ArchitectureAdapter):
        raise NotImplementedError(
            "Backward Lens currently supports the GPT2ArchitectureAdapter only; "
            f"got {type(model.adapter).__name__}"
        )
    if bool(getattr(model.cfg, "gated_mlp", False)):
        raise NotImplementedError("Backward Lens currently requires dense, non-gated GPT-2 MLPs")
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
            if not isinstance(weight, torch.nn.Parameter):
                raise ValueError(
                    f"layer {layer} {name} original weight must be a trainable Parameter"
                )
            if not weight.is_floating_point() or tuple(weight.shape) != expected_shape:
                raise ValueError(
                    f"layer {layer} {name} weight must have shape {expected_shape} "
                    f"and floating dtype; got {tuple(weight.shape)} and {weight.dtype}"
                )
            if not weight.requires_grad:
                raise ValueError(
                    f"layer {layer} {name} original weight must be a trainable Parameter"
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
    if torch.is_inference_mode_enabled():
        raise ValueError(
            "Backward Lens cannot capture gradients inside torch.inference_mode(); "
            "exit inference_mode before running the analysis"
        )
    _require_raw_gpt2_bridge(model)
    requested_layers = _validate_requested_layers(model, layers)
    projections = _get_gpt2_mlp_projections(model, requested_layers)
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if prompt == "":
        raise ValueError("prompt must not be empty")
    if not isinstance(target_token, str):
        raise TypeError("target_token must be a string")

    prompt_tokens = model.to_tokens(prompt, truncate=False)
    if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1 or prompt_tokens.shape[1] == 0:
        raise ValueError("prompt must tokenize to one non-empty sequence")
    prompt_token_count = int(prompt_tokens.shape[1])
    context_size = int(model.cfg.n_ctx)
    if prompt_token_count > context_size:
        raise ValueError(
            f"prompt token count {prompt_token_count} exceeds model context limit "
            f"n_ctx={context_size}"
        )
    target_tokens = model.to_tokens(target_token, prepend_bos=False)
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


class BackwardLens:
    """Analyze GPT-2 MLP weight gradients in the output vocabulary basis.

    The analyzer accepts a fresh, raw GPT-2 :class:`TransformerBridge`. Results
    retain no model or tokenizer reference and contain detached CPU-owned tensors.
    Raw backward signals are loss gradients; gradient descent subtracts them.
    """

    def __init__(self, model: Any):
        """Validate and retain the raw GPT-2 Bridge used for analyses."""
        _require_raw_gpt2_bridge(model)
        self._model = model

    def analyze(
        self,
        prompt: str,
        target_token: str,
        layers: Sequence[int],
        *,
        normalized: bool = False,
        top_k: int = DEFAULT_TOP_K,
        return_full_logits: bool = False,
    ) -> BackwardLensResult:
        """Analyze one final-position, one-token target loss.

        Args:
            prompt: Non-empty unbatched prompt text.
            target_token: Text encoding to exactly one token without BOS.
            layers: Unique GPT-2 layer indices in desired result order.
            normalized: Also project unit-normalized nonzero factors using the
                Normalized Logit Lens. Raw projections are always returned.
            top_k: Number of largest and smallest values and token ids retained
                per matrix and position. Defaults to 10.
            return_full_logits: Also retain full vocabulary tensors on CPU.
                Defaults to ``False`` to keep result size bounded.

        Returns:
            Detached gradient factors, bounded vocabulary rankings, norms,
            reconstruction errors, target metadata, and optional full logits.
        """
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a bool")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        vocabulary_size = int(self._model.cfg.d_vocab)
        if not 1 <= top_k <= vocabulary_size:
            raise ValueError(f"top_k must be in [1, {vocabulary_size}]; got {top_k!r}")
        if not isinstance(return_full_logits, bool):
            raise TypeError("return_full_logits must be a bool")
        capture = _capture_gpt2_mlp_gradient_factors(self._model, prompt, target_token, layers)
        layer_results: list[BackwardLensLayerResult] = []
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        for layer in capture.layers:
            input_result = _build_matrix_result(
                self._model,
                layer.input_projection,
                projected_factor="forward_inputs",
                include_normalized_logits=normalized,
                target_token_id=capture.target_token_id,
                top_k=top_k,
                return_full_logits=return_full_logits,
            )
            output_result = _build_matrix_result(
                self._model,
                layer.output_projection,
                projected_factor="output_gradients",
                include_normalized_logits=normalized,
                target_token_id=capture.target_token_id,
                top_k=top_k,
                return_full_logits=return_full_logits,
            )
            layer_results.append(
                BackwardLensLayerResult(
                    layer=layer.layer,
                    input_projection=input_result,
                    output_projection=output_result,
                )
            )
            absolute_errors.extend(
                (
                    layer.input_projection.absolute_reconstruction_error,
                    layer.output_projection.absolute_reconstruction_error,
                )
            )
            relative_errors.extend(
                (
                    layer.input_projection.relative_reconstruction_error,
                    layer.output_projection.relative_reconstruction_error,
                )
            )
        return BackwardLensResult(
            prompt=prompt,
            prompt_token_ids=capture.prompt_token_ids[0].clone(),
            target_token=target_token,
            target_token_id=capture.target_token_id,
            loss=capture.loss,
            layers=tuple(layer_results),
            max_absolute_reconstruction_error=max(absolute_errors),
            max_relative_reconstruction_error=max(relative_errors),
            includes_normalized_logits=normalized,
            includes_full_logits=return_full_logits,
        )
