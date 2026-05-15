"""vLLM plugin entry point.

The plugin's ``register()`` monkey-patches ``Worker.load_model`` so capture hooks
are installed AFTER weights load and BEFORE ``compile_or_warm_up_model``. This is
the only mechanism on current vLLM (validated by Phase 0b spike) where hooks
make it into the compiled FX graph — hooks installed via ``collective_rpc`` after
``LLM(...)`` returns are silently ignored (PyTorch #117758).

The hook body is GPU-only and shape-agnostic:
   * In-place ``buf[:n].copy_(t)`` where ``buf`` is a pre-allocated GPU tensor.
   * No ``.cpu()`` (illegal during CUDA-graph capture).
   * No Python access to ``t.shape`` other than via SymInt slice indices (forces
     shape specialization otherwise; breaks dynamic-shape compilation).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

# Module-level singletons read by both the driver (boot_vllm calls configure())
# and the worker (the monkey-patched load_model reads _config). entry_points
# discovery imports this module in every vLLM process so the same dict is visible.
_config: Dict[str, Any] = {}
_buffers: Dict[str, torch.Tensor] = {}
_install_patched = False
_orig_load_model = None


def configure(
    capture_specs: Dict[str, Tuple[str, int]],
    max_num_batched_tokens: int,
    dtype: torch.dtype,
) -> None:
    """Called from the driver by :func:`boot_vllm` before ``LLM(...)``.

    Args:
        capture_specs: ``{canonical_hook_name: (dot_path_in_model, output_width)}``.
        max_num_batched_tokens: Bucket length for each pre-allocated buffer.
        dtype: Model dtype; buffers match.
    """
    _config["capture_specs"] = capture_specs
    _config["max_num_batched_tokens"] = max_num_batched_tokens
    _config["dtype"] = dtype
    _buffers.clear()


def reset() -> None:
    """Clear plugin state. Useful between back-to-back ``boot_vllm`` calls in the same process."""
    _config.clear()
    _buffers.clear()


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

        for canonical_name, (dot_path, width) in specs.items():
            target = self.model_runner.model
            for seg in dot_path.split("."):
                target = target[int(seg)] if seg.isdigit() else getattr(target, seg)
            buf = torch.zeros(max_n, width, device=device, dtype=dtype)
            _buffers[canonical_name] = buf
            target.register_forward_hook(_make_capture_hook(buf))

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
