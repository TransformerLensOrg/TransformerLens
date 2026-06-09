"""SGLang plugin: monkey-patches ``ModelRunner.load_model`` to install capture
hooks pre-compile. Hook semantics mirror the vLLM compiled-mode hook (pre-allocated
GPU buffers, affine intervention via scale/bias buffer swap, first-write-wins
gating); see :mod:`sources.vllm.plugin` for the design rationale."""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import torch

from .worker_extension import _apply_intervention  # noqa: F401 — re-exported for tests

# Decoder layers return the fused-residual ``(mlp_delta, residual)`` tuple;
# their hooks materialize the sum.
_DECODER_LAYER_PATH = re.compile(r"^model\.layers\.\d+$")

# Transient driver→worker signal during Engine construction.
_config: Dict[str, Any] = {}
_install_patched = False
_orig_load_model = None


def configure(
    capture_specs: Dict[str, Tuple[str, int]],
    max_num_batched_tokens: int,
    dtype: torch.dtype,
) -> None:
    """Set capture specs, buffer length, and dtype before ``Engine(...)``."""
    _config["capture_specs"] = capture_specs
    _config["max_num_batched_tokens"] = max_num_batched_tokens
    _config["dtype"] = dtype


def register() -> None:
    """Idempotently monkey-patch ``ModelRunner.load_model``. ``boot_sglang`` clears
    ``_config`` after each Engine, so a non-TL Engine in the same process no-ops."""
    global _install_patched, _orig_load_model
    if _install_patched:
        return
    from . import worker_extension
    from .internals import model_runner_class

    ModelRunner = model_runner_class()
    _orig_load_model = ModelRunner.load_model

    # The tl_* methods read state that patched_load_model writes onto each
    # ModelRunner instance; they're invoked from the driver via RpcReqInput.
    for method_name in worker_extension.TL_METHOD_NAMES:
        setattr(ModelRunner, method_name, getattr(worker_extension, method_name))

    def patched_load_model(self):
        _orig_load_model(self)
        specs = _config.get("capture_specs")
        if not specs:
            return  # not a TL-driven Engine
        max_n = _config["max_num_batched_tokens"]
        dtype = _config["dtype"]
        device = next(self.model.parameters()).device

        # Detach prior handles in case of unexpected reload.
        for handle in getattr(self, "_tl_hook_handles", []):
            handle.remove()
        self._tl_buffers = {}
        self._tl_scale_buffers = {}
        self._tl_bias_buffers = {}
        # First-write-wins flags default to closed; driver opens via tl_reset_capture_flags.
        self._tl_capture_flags = {}
        self._tl_hook_handles = []
        self._tl_fire_counter = torch.zeros(1, device=device, dtype=torch.int64)

        for canonical_name, (dot_path, width) in specs.items():
            target = self.model
            for seg in dot_path.split("."):
                target = target[int(seg)] if seg.isdigit() else getattr(target, seg)
            materialize = bool(_DECODER_LAYER_PATH.match(dot_path))

            capture_buf = torch.zeros(max_n, width, device=device, dtype=dtype)
            scale_buf = torch.ones(width, device=device, dtype=dtype)
            bias_buf = torch.zeros(width, device=device, dtype=dtype)
            capture_flag = torch.ones(1, device=device, dtype=torch.int64)
            self._tl_buffers[canonical_name] = capture_buf
            self._tl_scale_buffers[canonical_name] = scale_buf
            self._tl_bias_buffers[canonical_name] = bias_buf
            self._tl_capture_flags[canonical_name] = capture_flag

            handle = target.register_forward_hook(
                _make_capture_hook(
                    capture_buf,
                    scale_buf,
                    bias_buf,
                    self._tl_fire_counter,
                    capture_flag,
                    materialize=materialize,
                )
            )
            self._tl_hook_handles.append(handle)

    ModelRunner.load_model = patched_load_model
    _install_patched = True


def _make_capture_hook(
    capture_buf: torch.Tensor,
    scale_buf: torch.Tensor,
    bias_buf: torch.Tensor,
    fire_counter: torch.Tensor,
    capture_flag: torch.Tensor,
    *,
    materialize: bool = False,
):
    """Affine + first-write-wins capture into pre-allocated buffers; mirrors the
    vLLM compiled hook. ``materialize=True`` (decoder layers) sums the fused
    ``(mlp_delta, residual)`` tuple so the capture matches HF's ``hook_out``."""

    @torch.no_grad()
    def hook(_module, _inputs, output):
        fire_counter.add_(1)
        if materialize and isinstance(output, tuple) and len(output) == 2:
            hidden, residual = output
            if isinstance(hidden, torch.Tensor) and isinstance(residual, torch.Tensor):
                t = hidden + residual
                modified = t * scale_buf + bias_buf
                n = t.shape[0]
                _gated_capture(capture_buf, n, modified, capture_flag)
                return (modified - residual, residual)

        tuple_tail: tuple = ()
        if isinstance(output, tuple):
            t = output[0]
            tuple_tail = output[1:]
        else:
            t = output
        if not isinstance(t, torch.Tensor):
            return None
        modified = t * scale_buf + bias_buf
        n = t.shape[0]
        _gated_capture(capture_buf, n, modified, capture_flag)
        if isinstance(output, tuple):
            return (modified,) + tuple_tail
        return modified

    return hook


def _gated_capture(
    capture_buf: torch.Tensor, n: Any, modified: torch.Tensor, capture_flag: torch.Tensor
) -> None:
    """First-write-wins via torch.where (compile-safe — no Python branching).
    Always closes the flag; driver re-opens it before the next capture forward."""
    existing = capture_buf.narrow(0, 0, n)
    to_write = torch.where(capture_flag.bool(), existing, modified)
    capture_buf.narrow(0, 0, n).copy_(to_write)
    capture_flag.fill_(1)
