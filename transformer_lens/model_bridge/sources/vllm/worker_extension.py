"""Worker extension exposed to collective_rpc for capture-buffer reads.

Hook *installation* lives in :mod:`plugin` (must happen pre-compile). This class
only exposes the post-compile read/clear surface. All methods prefixed ``tl_``
to avoid colliding with vLLM ``Worker`` attributes; vLLM asserts no-conflict
on injection.
"""
from __future__ import annotations

from typing import Dict, List

import torch


class TLWorkerExtension:
    """Mixed into vLLM's ``Worker`` via ``worker_extension_cls``."""

    def tl_read_captures(self, prompt_lens: List[int]) -> Dict[str, torch.Tensor]:
        """Slice each capture buffer back to ``sum(prompt_lens)`` rows and return CPU copies.

        ``.cpu()`` is legal here — we're outside any CUDA-graph capture region.
        """
        from . import plugin

        total = sum(prompt_lens)
        out: Dict[str, torch.Tensor] = {}
        for name, buf in plugin._buffers.items():
            n = min(total, buf.shape[0])
            out[name] = buf[:n].detach().cpu().clone()
        return out

    def tl_capture_buffer_shapes(self) -> Dict[str, tuple]:
        """Diagnostic: report the shape of each pre-allocated capture buffer."""
        from . import plugin

        return {name: tuple(buf.shape) for name, buf in plugin._buffers.items()}

    def tl_zero_captures(self) -> None:
        """Zero all capture buffers (for between-call sanity checks)."""
        from . import plugin

        for buf in plugin._buffers.values():
            buf.zero_()
