"""Worker-side ``tl_*`` methods invoked via SGLang's ``RpcReqInput`` channel.
Monkey-patched onto ``ModelRunner`` by :meth:`plugin.register` (which is where
the ``_tl_*`` state lives). All methods ``tl_``-prefixed to avoid colliding with
SGLang attributes. Single-prompt compiled mode for v1; batched-mode capture
follows the vLLM design and is a later increment."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .intervention_specs import SUPPORTED_OPS

# Methods plugin.register() monkey-patches onto ModelRunner.
TL_METHOD_NAMES = (
    "tl_read_captures",
    "tl_set_interventions",
    "tl_get_param",
    "tl_reset_counter",
    "tl_read_counter",
    "tl_reset_capture_flags",
    "tl_remove_hooks",
)


def tl_read_captures(
    self: Any,
    prompt_lens: List[int],
    names: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Slice each capture buffer to ``sum(prompt_lens)`` rows; CPU clones.
    Single GPU→CPU crossing — pass ``names`` to filter for bandwidth."""
    total = sum(prompt_lens)
    buffers: Dict[str, torch.Tensor] = getattr(self, "_tl_buffers", {})
    wanted = buffers.keys() if names is None else [n for n in names if n in buffers]
    return {name: buffers[name][:total].detach().cpu().clone() for name in wanted}


def tl_set_interventions(self: Any, specs: Dict[str, Dict[str, Any]]) -> None:
    """Reset every affine buffer to identity then apply each spec. Driver pushes
    the full spec dict (or ``{}`` to clear). Ops: suppress / scale (``factor``) /
    add / set (``value``: scalar or width-shaped)."""
    scale_bufs: Dict[str, torch.Tensor] = getattr(self, "_tl_scale_buffers", {})
    bias_bufs: Dict[str, torch.Tensor] = getattr(self, "_tl_bias_buffers", {})
    # Identity first; clears any stale state from the prior forward.
    for name, sb in scale_bufs.items():
        sb.fill_(1.0)
        bias_bufs[name].zero_()
    for hook_name, spec in specs.items():
        if hook_name not in scale_bufs:
            raise KeyError(f"Unknown hook for intervention: {hook_name!r}")
        _apply_intervention(scale_bufs[hook_name], bias_bufs[hook_name], spec)


def tl_get_param(self: Any, dotted_name: str) -> Optional[torch.Tensor]:
    """Named model tensor as a CPU clone (e.g. ``model.norm.weight`` to invert
    the post-weight ``ln_final.hook_normalized`` convention). ``None`` if missing."""
    target = self.model
    for seg in dotted_name.split("."):
        target = target[int(seg)] if seg.isdigit() else getattr(target, seg, None)
        if target is None:
            return None
    return target.detach().cpu().clone() if isinstance(target, torch.Tensor) else None


def tl_reset_counter(self: Any) -> None:
    """Zero the shared hook-fire counter before a forward."""
    counter = getattr(self, "_tl_fire_counter", None)
    if counter is not None:
        counter.zero_()


def tl_read_counter(self: Any) -> int:
    """Total hook fires since the last reset."""
    counter = getattr(self, "_tl_fire_counter", None)
    return int(counter.item()) if counter is not None else 0


def tl_reset_capture_flags(self: Any) -> None:
    """Open every capture gate; decode-step forwards then self-copy until the
    next reset, so a multi-token generate only captures the prefill."""
    flags: Dict[str, torch.Tensor] = getattr(self, "_tl_capture_flags", {})
    for flag in flags.values():
        flag.zero_()


def tl_remove_hooks(self: Any) -> None:
    """Detach all capture hooks and drop buffer references. Idempotent."""
    for handle in getattr(self, "_tl_hook_handles", []):
        handle.remove()
    self._tl_hook_handles = []
    self._tl_buffers = {}
    self._tl_scale_buffers = {}
    self._tl_bias_buffers = {}
    self._tl_capture_flags = {}


def _apply_intervention(
    scale_buf: torch.Tensor, bias_buf: torch.Tensor, spec: Dict[str, Any]
) -> None:
    """In-place affine-buffer writes; same semantics as the vLLM driver."""
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
    # 0-d broadcasts across width; width-shaped writes element-wise (e.g. SAE steering).
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
