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

from typing import Any, Dict, List, Optional, Set

import torch

from .intervention_specs import SUPPORTED_OPS, validate_spec

# Explicit tensor wire format for collective_rpc returns. vLLM's multiproc RPC
# cannot round-trip raw tensors from extension methods — its tensor-aware msgpack
# encoder's header (['float32', [rows, cols], buf_idx]) leaks through undecoded on
# the response path (GPU-verified on vllm 0.20.2). dtype/shape/bytes are all
# msgpack-native, so this survives any executor topology; the in-process executor
# pays one extra CPU copy.
_TL_TENSOR_KEY = "__tl_tensor__"


def encode_tensor(t: torch.Tensor) -> Dict[str, Any]:
    """Encode a CPU tensor for the RPC wire. bf16 rides as fp32 bytes (exact)."""
    t = t.detach().cpu().contiguous()
    dtype_name = str(t.dtype).removeprefix("torch.")
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return {
        _TL_TENSOR_KEY: True,
        "dtype": dtype_name,
        "shape": list(t.shape),
        "data": t.numpy().tobytes(),
    }


def decode_tensor(payload: Dict[str, Any]) -> torch.Tensor:
    """Inverse of :func:`encode_tensor`; used driver-side."""
    import numpy as np

    dtype = getattr(torch, payload["dtype"])
    wire_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    np_dtype = torch.empty(0, dtype=wire_dtype).numpy().dtype
    array = np.frombuffer(payload["data"], dtype=np_dtype).copy()
    tensor = torch.from_numpy(array).reshape(payload["shape"])
    return tensor.to(torch.bfloat16) if dtype == torch.bfloat16 else tensor


class TLWorkerExtension:
    """Mixed into vLLM's ``Worker`` via ``worker_extension_cls``."""

    _tl_hook_handles: list
    _tl_buffers: Dict[str, torch.Tensor]
    _tl_scale_buffers: Dict[str, torch.Tensor]
    _tl_bias_buffers: Dict[str, torch.Tensor]
    # Per-hook first-write-wins gates (compiled mode); see plugin._gated_capture.
    _tl_capture_flags: Dict[str, torch.Tensor]
    _tl_fire_counter: torch.Tensor
    # Batched-mode state (eager).
    _tl_accum: Dict[tuple, List[torch.Tensor]]
    _tl_intervention_specs: Dict[str, Dict[str, Any]]
    # Specs whose module doesn't exist on this rank (PP layer shards).
    _tl_absent_hooks: set

    def tl_absent_hooks(self) -> Optional[List[str]]:
        """Spec names that installed no hook on this rank — the boot site verifies
        their union across ranks covers every spec. ``None`` means installation
        never ran at all (plugin patch absent or spec channel empty), which the
        coverage check must treat as fatal rather than vacuously complete."""
        if not hasattr(self, "_tl_absent_hooks"):
            return None
        return sorted(self._tl_absent_hooks)

    def tl_read_captures(
        self, prompt_lens: List[int], names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Slice each capture buffer to ``sum(prompt_lens)`` rows; wire-encoded CPU copies.

        ``names`` restricts the read (``None`` = all) — this is the only GPU→CPU
        crossing, so it's where a names_filtered run saves bandwidth. Caller gates
        ``sum(prompt_lens) <= max_num_batched_tokens``, so ``total`` is in bounds.
        """
        total = sum(prompt_lens)
        buffers: Dict[str, torch.Tensor] = getattr(self, "_tl_buffers", {})
        # Ownership is dynamic, not structural: a hook can be installed on a module
        # this rank holds but never runs (PP stages share tied-embedding aliases and
        # some archs instantiate norm on every rank), so a still-open first-write
        # flag means the buffer holds no data from this forward — don't serve it.
        flags: Dict[str, torch.Tensor] = getattr(self, "_tl_capture_flags", {})
        wanted = buffers.keys() if names is None else [n for n in names if n in buffers]
        return {
            name: encode_tensor(buffers[name][:total])
            for name in wanted
            if name not in flags or bool(flags[name].item())
        }

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
        absent: Set[str] = getattr(self, "_tl_absent_hooks", set())
        for hook_name, spec in specs.items():
            if hook_name in absent:
                # Another PP stage owns this hook; its rank applies the spec. The
                # driver's supported_hook_points check already caught real typos.
                continue
            if hook_name not in scale_bufs:
                raise KeyError(f"Unknown hook for intervention: {hook_name!r}")
            # Authoritative validation: producers that bypass VLLMDriver (the Inspect
            # vLLM provider pushes specs over this same RPC) get identical rejection.
            validate_spec(hook_name, spec, width=scale_bufs[hook_name].shape[-1])
            _apply_intervention(scale_bufs[hook_name], bias_bufs[hook_name], spec)

    def tl_get_param(self, dotted_name: str) -> Optional[Dict[str, Any]]:
        """Read a named model tensor (e.g. ``model.norm.weight``) as a wire-encoded
        CPU clone.

        ``None`` if the path doesn't resolve to a tensor. The bridge has no general
        weight surface, so this is how callers reach e.g. the ln_final weight.
        """
        target = getattr(self, "model_runner").model
        for seg in dotted_name.split("."):
            target = target[int(seg)] if seg.isdigit() else getattr(target, seg, None)
            if target is None:
                return None
        return encode_tensor(target) if isinstance(target, torch.Tensor) else None

    def tl_reset_counter(self) -> None:
        """Zero the shared hook-fire counter before a forward."""
        counter = getattr(self, "_tl_fire_counter", None)
        if counter is not None:
            counter.zero_()

    def tl_reset_capture_flags(self) -> None:
        """Open every per-hook capture gate so the next forward writes to the buffers.

        First-write-wins gating means decode-step forwards self-copy and never overwrite
        prefill activations — the driver calls this once before any capture-needing
        generate (single-forward or multi-token eval) and ``tl_read_captures`` afterward.
        """
        flags: Dict[str, torch.Tensor] = getattr(self, "_tl_capture_flags", {})
        for flag in flags.values():
            flag.zero_()

    def tl_read_counter(self) -> int:
        """Total hook fires since the last reset."""
        counter = getattr(self, "_tl_fire_counter", None)
        return int(counter.item()) if counter is not None else 0

    def tl_reset_accumulators(self) -> None:
        """Clear capture chunks before each generate, else prior chunks leak into the cat."""
        self._tl_accum = {}

    def tl_read_batched_captures(
        self, names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Cat per-request chunks into ``{req_id: {hook: (seq, width)}}`` (token-order).

        ``names`` restricts to those hooks (``None`` = all). Note the per-chunk
        GPU→CPU copy already happened in the hook, so this only saves the cat.
        """
        accum: Dict[tuple, List[torch.Tensor]] = getattr(self, "_tl_accum", {})
        nameset = None if names is None else set(names)
        out: Dict[str, Dict[str, Any]] = {}
        for (req_id, name), chunks in accum.items():
            if nameset is not None and name not in nameset:
                continue
            out.setdefault(req_id, {})[name] = encode_tensor(torch.cat(chunks, dim=0))
        return out

    def tl_set_batched_interventions(self, specs: Dict[str, Dict[str, Any]]) -> None:
        """Store the global spec dict the eager hook reads; ``{}`` clears."""
        for hook_name, spec in specs.items():
            validate_spec(hook_name, spec)
        self._tl_intervention_specs = dict(specs)

    def tl_remove_hooks(self) -> None:
        """Detach all capture hooks and drop buffer references. Idempotent."""
        for handle in getattr(self, "_tl_hook_handles", []):
            handle.remove()
        self._tl_hook_handles = []
        self._tl_buffers = {}
        self._tl_scale_buffers = {}
        self._tl_bias_buffers = {}
        self._tl_capture_flags = {}
        self._tl_accum = {}
        self._tl_intervention_specs = {}
        self._tl_absent_hooks = set()


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


def _write_rows(buf: torch.Tensor, idx: Optional[List[int]], value: Any) -> None:
    """In-place write of ``value`` (Python scalar or ``(width,)`` tensor) into ``buf``.

    ``idx is None`` writes the whole buffer — for a 2-D ``(max_n, width)`` buffer that
    broadcasts a ``(width,)`` value across every row. ``idx`` (a list of row indices)
    writes only those rows of a 2-D buffer.
    """
    if idx is None:
        if isinstance(value, torch.Tensor):
            buf[...] = value  # (width,) broadcasts across rows for a 2-D buffer
        else:
            buf.fill_(value)
    else:
        buf[idx] = value  # advanced-index rows; scalar or (width,) broadcasts


def _apply_intervention(
    scale_buf: torch.Tensor, bias_buf: torch.Tensor, spec: Dict[str, Any]
) -> None:
    """Translate a spec dict to in-place buffer writes.

    ``spec['pos']`` (int or list[int]) scopes the edit to those sequence rows and
    requires 2-D ``(max_n, width)`` affine buffers (position interventions enabled at
    boot). Absent ``pos`` writes the whole buffer, applying to every position. The
    caller resets both buffers to identity first, so a ``pos`` edit leaves other rows
    untouched.
    """
    op = spec.get("op")
    if op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported intervention op: {op!r}. Supported: {sorted(SUPPORTED_OPS)}")
    pos = spec.get("pos")
    idx: Optional[List[int]] = None
    if pos is not None:
        if scale_buf.ndim != 2:
            raise ValueError(
                "Per-position 'pos' requires 2-D affine buffers; boot with "
                "enable_position_interventions=True."
            )
        idx = [pos] if isinstance(pos, int) else list(pos)
        max_n = scale_buf.shape[0]
        bad = [p for p in idx if p < 0 or p >= max_n]
        if bad:
            raise ValueError(f"Intervention 'pos' {bad} out of range [0, {max_n}).")

    if op == "suppress":
        _write_rows(scale_buf, idx, 0.0)
        _write_rows(bias_buf, idx, 0.0)
        return
    if op == "scale":
        _write_rows(scale_buf, idx, float(spec["factor"]))
        _write_rows(bias_buf, idx, 0.0)
        return
    value = torch.as_tensor(spec["value"], device=bias_buf.device, dtype=bias_buf.dtype)
    width = bias_buf.shape[-1]
    # 0-d broadcasts across width (e.g. "shift all dims by 0.5"); width-shaped
    # writes element-wise (e.g. SAE steering vector). Anything else is an error.
    if value.ndim != 0 and value.shape != (width,):
        raise ValueError(
            f"Intervention 'value' must be a scalar or shape {(width,)}; "
            f"got shape {tuple(value.shape)}"
        )
    _write_rows(bias_buf, idx, value.item() if value.ndim == 0 else value)
    if op == "add":
        _write_rows(scale_buf, idx, 1.0)
    elif op == "set":
        _write_rows(scale_buf, idx, 0.0)
    else:
        raise RuntimeError(
            f"op {op!r} is in SUPPORTED_OPS but _apply_intervention has no branch for it."
        )
