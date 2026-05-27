"""Worker extension exposed to collective_rpc for capture reads and
intervention writes.

Hook *installation* lives in :mod:`plugin` (must happen pre-compile). This class
only exposes the read/write surface. State is per-Worker so concurrent
``boot_vllm`` calls don't collide. All methods prefixed ``tl_`` to avoid
colliding with vLLM ``Worker`` attributes.

Two capture modes, selected at boot:
  * Compiled (default): per-hook GPU buffers + affine scale/bias swap. Single
    prompt. ``tl_read_captures`` / ``tl_set_interventions``.
  * Batched (``enable_batching=True``, eager): per-(req_id, hook) CPU
    accumulators filled in the hook via query_start_loc segmentation; arbitrary
    batch + chunked prefill. ``tl_read_batched_captures`` /
    ``tl_set_batched_interventions`` / ``tl_reset_accumulators``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from .intervention_specs import SUPPORTED_OPS


class TLWorkerExtension:
    """Mixed into vLLM's ``Worker`` via ``worker_extension_cls``."""

    _tl_hook_handles: list
    _tl_buffers: Dict[str, torch.Tensor]
    _tl_scale_buffers: Dict[str, torch.Tensor]
    _tl_bias_buffers: Dict[str, torch.Tensor]
    _tl_fire_counter: torch.Tensor
    # Batched-mode state (eager).
    _tl_accum: Dict[tuple, List[torch.Tensor]]
    _tl_intervention_specs: Dict[str, Dict[str, Any]]

    def tl_read_captures(self, prompt_lens: List[int]) -> Dict[str, torch.Tensor]:
        """Slice each capture buffer back to ``sum(prompt_lens)`` rows; CPU copies.

        Caller (VLLMDriver.forward) gates ``sum(prompt_lens) <= max_num_batched_tokens``
        before the RPC, so ``total`` is always within buffer bounds.
        """
        total = sum(prompt_lens)
        buffers: Dict[str, torch.Tensor] = getattr(self, "_tl_buffers", {})
        return {name: buf[:total].detach().cpu().clone() for name, buf in buffers.items()}

    def tl_set_interventions(self, specs: Dict[str, Dict[str, Any]]) -> None:
        """Reset all affine buffers to identity, then apply each spec.

        Driver pushes the full spec set every forward (or ``{}`` to reset).
        Spec format: ``{hook_name: {"op": <op>, ...op-specific params>}}``.
        Supported ops: suppress, scale (``factor``: float), add and set
        (``value``: scalar broadcast across width, or 1-D shape ``(width,)``).
        """
        scale_bufs: Dict[str, torch.Tensor] = getattr(self, "_tl_scale_buffers", {})
        bias_bufs: Dict[str, torch.Tensor] = getattr(self, "_tl_bias_buffers", {})
        # Reset every hook to identity first — clears any stale state from
        # the previous forward.
        for name, sb in scale_bufs.items():
            sb.fill_(1.0)
            bias_bufs[name].zero_()
        for hook_name, spec in specs.items():
            if hook_name not in scale_bufs:
                raise KeyError(f"Unknown hook for intervention: {hook_name!r}")
            _apply_intervention(scale_bufs[hook_name], bias_bufs[hook_name], spec)

    def tl_reset_counter(self) -> None:
        """Zero the shared hook-fire counter before a forward."""
        counter = getattr(self, "_tl_fire_counter", None)
        if counter is not None:
            counter.zero_()

    def tl_read_counter(self) -> int:
        """Total hook fires since the last reset."""
        counter = getattr(self, "_tl_fire_counter", None)
        return int(counter.item()) if counter is not None else 0

    def tl_reset_accumulators(self) -> None:
        """Clear capture chunks before each generate, else prior chunks leak into the cat."""
        self._tl_accum = {}

    def tl_read_batched_captures(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Cat per-request chunks into ``{req_id: {hook: (seq, width)}}`` (chunks are token-order)."""
        accum: Dict[tuple, List[torch.Tensor]] = getattr(self, "_tl_accum", {})
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for (req_id, name), chunks in accum.items():
            out.setdefault(req_id, {})[name] = torch.cat(chunks, dim=0)
        return out

    def tl_set_batched_interventions(self, specs: Dict[str, Dict[str, Any]]) -> None:
        """Store the global spec dict the eager hook reads; ``{}`` clears."""
        for spec in specs.values():
            op = spec.get("op")
            if op not in SUPPORTED_OPS:
                raise ValueError(
                    f"Unsupported intervention op: {op!r}. Supported: {sorted(SUPPORTED_OPS)}"
                )
        self._tl_intervention_specs = dict(specs)

    def tl_remove_hooks(self) -> None:
        """Detach all capture hooks and drop buffer references. Idempotent."""
        for handle in getattr(self, "_tl_hook_handles", []):
            handle.remove()
        self._tl_hook_handles = []
        self._tl_buffers = {}
        self._tl_scale_buffers = {}
        self._tl_bias_buffers = {}
        self._tl_accum = {}
        self._tl_intervention_specs = {}


def _apply_op(t: torch.Tensor, spec: Dict[str, Any]) -> torch.Tensor:
    """Apply a spec to a tensor in-line (eager path; no GPU buffer to swap)."""
    op = spec.get("op")
    if op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported intervention op: {op!r}. Supported: {sorted(SUPPORTED_OPS)}")
    if op == "suppress":
        return torch.zeros_like(t)
    if op == "scale":
        return t * float(spec["factor"])
    value = torch.as_tensor(spec["value"], device=t.device, dtype=t.dtype)
    if value.ndim != 0 and value.shape != (t.shape[-1],):
        raise ValueError(
            f"Intervention 'value' must be a scalar or shape {(t.shape[-1],)}; "
            f"got shape {tuple(value.shape)}"
        )
    if op == "add":
        return t + value
    return torch.zeros_like(t) + value  # set


def _apply_intervention(
    scale_buf: torch.Tensor, bias_buf: torch.Tensor, spec: Dict[str, Any]
) -> None:
    """Translate a spec dict to in-place buffer writes."""
    op = spec.get("op")
    if op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported intervention op: {op!r}. Supported: {sorted(SUPPORTED_OPS)}")
    if op == "suppress":
        scale_buf.zero_()
        bias_buf.zero_()
        return
    if op == "scale":
        scale_buf.fill_(float(spec["factor"]))
        bias_buf.zero_()
        return
    value = torch.as_tensor(spec["value"], device=bias_buf.device, dtype=bias_buf.dtype)
    # 0-d broadcasts across width (e.g. "shift all dims by 0.5"); width-shaped
    # writes element-wise (e.g. SAE steering vector). Anything else is an error.
    if value.ndim == 0:
        bias_buf.fill_(value.item())
    elif value.shape == bias_buf.shape:
        bias_buf.copy_(value)
    else:
        raise ValueError(
            f"Intervention 'value' must be a scalar or shape {tuple(bias_buf.shape)}; "
            f"got shape {tuple(value.shape)}"
        )
    if op == "add":
        scale_buf.fill_(1.0)
    elif op == "set":
        scale_buf.zero_()
    else:
        raise RuntimeError(
            f"op {op!r} is in SUPPORTED_OPS but _apply_intervention has no branch for it."
        )
