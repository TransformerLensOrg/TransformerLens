"""vLLM plugin entry point.

Monkey-patches ``Worker.load_model`` to install capture hooks after weights
load and before ``compile_or_warm_up_model`` — the only window where hooks
make it into the compiled FX graph (PyTorch #117758).

Hook body invariants under ``torch.compile``: in-place writes to a
pre-allocated GPU tensor; no ``.cpu()`` (illegal during CUDA-graph capture);
SymInt-indexed slicing only (Python ``.shape`` access forces specialization).

Interventions ride the same hook. Each hook applies an affine transform
``output = output * scale_buf + bias_buf`` before capturing. Defaults are
``scale=ones`` / ``bias=zeros`` (identity). The driver swaps buffer contents
between forwards via ``tl_set_interventions`` — the FX graph references the
buffers, so swaps take effect on the next dispatch without recompiling.

Memory cost: the affine transform allocates a transient output-shape tensor
per hook per forward, even in identity mode. Peak forward memory is roughly
1.5× the prior capture-only design — the caching allocator reuses the slot
but the peak pressure rises. Branching the hook to skip the affine in
identity mode would defeat the swap-via-buffer trick and break the FX graph.
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

    No ``unregister()`` symmetry: the patch stays for process lifetime. Benign
    because ``patched_load_model`` no-ops when ``_config["capture_specs"]`` is
    absent — and ``boot_vllm`` clears ``_config`` after each ``LLM(...)``, so
    any subsequent non-TL ``LLM(...)`` in the same process hits the no-op path.
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
        self._tl_scale_buffers = {}
        self._tl_bias_buffers = {}
        self._tl_hook_handles = []
        for canonical_name, (dot_path, width) in specs.items():
            target = self.model_runner.model
            for seg in dot_path.split("."):
                target = target[int(seg)] if seg.isdigit() else getattr(target, seg)
            capture_buf = torch.zeros(max_n, width, device=device, dtype=dtype)
            # Affine identity at install. Driver swaps via tl_set_interventions
            # to enable suppress/scale/add/set ops between forwards.
            scale_buf = torch.ones(width, device=device, dtype=dtype)
            bias_buf = torch.zeros(width, device=device, dtype=dtype)
            self._tl_buffers[canonical_name] = capture_buf
            self._tl_scale_buffers[canonical_name] = scale_buf
            self._tl_bias_buffers[canonical_name] = bias_buf
            handle = target.register_forward_hook(
                _make_capture_hook(capture_buf, scale_buf, bias_buf)
            )
            self._tl_hook_handles.append(handle)

    Worker.load_model = patched_load_model
    _install_patched = True


def _make_capture_hook(
    capture_buf: torch.Tensor,
    scale_buf: torch.Tensor,
    bias_buf: torch.Tensor,
):
    """GPU-only, dynamic-shape-safe affine + capture into pre-allocated buffers."""

    @torch.no_grad()
    def hook(_module, _inputs, output):
        tuple_tail: tuple = ()
        if isinstance(output, tuple):
            t = output[0]
            tuple_tail = output[1:]
        else:
            t = output
        if not isinstance(t, torch.Tensor):
            return None
        # Affine transform; default scale=1 / bias=0 means identity. Driver
        # swaps buffer contents to enable interventions.
        modified = t * scale_buf + bias_buf
        # SymInt-indexed slice traces under dynamic shapes without specialization.
        n = modified.shape[0]
        capture_buf[:n].copy_(modified)
        # Preserve the input wrapping — a 1-tuple input must come back as a 1-tuple
        # (``tuple_tail`` is falsy in that case, so don't gate on it).
        if isinstance(output, tuple):
            return (modified,) + tuple_tail
        return modified

    return hook
