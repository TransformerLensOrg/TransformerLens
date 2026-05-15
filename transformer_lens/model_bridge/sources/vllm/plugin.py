"""vLLM plugin entry point.

Monkey-patches ``Worker.load_model`` to install capture hooks after weights
load and before ``compile_or_warm_up_model`` — the only window where hooks
make it into the compiled FX graph (PyTorch #117758).

Hook body invariants under ``torch.compile``: in-place writes to a
pre-allocated GPU tensor; no ``.cpu()`` (illegal during CUDA-graph capture);
SymInt-indexed slicing only (Python ``.shape`` access forces specialization).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

# Transient signal driver → worker during LLM construction. Per-Worker buffers
# live on Worker instances so concurrent boot_vllm calls don't collide.
_config: Dict[str, Any] = {}
_install_patched = False
_orig_load_model = None


def configure(
    capture_specs: Dict[str, Tuple[str, int]],
    max_num_batched_tokens: int,
    dtype: torch.dtype,
) -> None:
    """Set capture specs, buffer length, and dtype before ``LLM(...)``."""
    _config["capture_specs"] = capture_specs
    _config["max_num_batched_tokens"] = max_num_batched_tokens
    _config["dtype"] = dtype


def register() -> None:
    """Idempotent monkey-patch of ``Worker.load_model``.

    vLLM calls ``register()`` once per process at entry-points discovery. Idempotent
    so re-imports (notebook restarts, repeated ``boot_vllm`` in the same process)
    don't double-wrap.
    """
    global _install_patched, _orig_load_model
    if _install_patched:
        return
    from vllm.v1.worker.gpu_worker import Worker

    _orig_load_model = Worker.load_model

    def patched_load_model(self):
        _orig_load_model(self)
        specs = _config.get("capture_specs")
        if not specs:
            return  # not a TL-driven LLM; no hooks to install
        max_n = _config["max_num_batched_tokens"]
        dtype = _config["dtype"]
        device = next(self.model_runner.model.parameters()).device

        # Detach prior handles before reassigning — vLLM doesn't double-load
        # today, but unconditional reassignment would orphan hooks if it ever did.
        for handle in getattr(self, "_tl_hook_handles", []):
            handle.remove()
        self._tl_buffers = {}
        self._tl_hook_handles = []
        for canonical_name, (dot_path, width) in specs.items():
            target = self.model_runner.model
            for seg in dot_path.split("."):
                target = target[int(seg)] if seg.isdigit() else getattr(target, seg)
            buf = torch.zeros(max_n, width, device=device, dtype=dtype)
            self._tl_buffers[canonical_name] = buf
            handle = target.register_forward_hook(_make_capture_hook(buf))
            self._tl_hook_handles.append(handle)

    Worker.load_model = patched_load_model
    _install_patched = True


def _make_capture_hook(buffer: torch.Tensor):
    """GPU-only, dynamic-shape-safe capture into a pre-allocated buffer."""

    @torch.no_grad()
    def hook(_module, _inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        if not isinstance(t, torch.Tensor):
            return
        # SymInt-indexed slice traces under dynamic shapes without specialization.
        n = t.shape[0]
        buffer[:n].copy_(t)

    return hook
