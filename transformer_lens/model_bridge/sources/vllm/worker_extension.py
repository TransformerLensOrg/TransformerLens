"""Worker extension exposed to collective_rpc for capture-buffer reads.

Hook *installation* lives in :mod:`plugin` (must happen pre-compile). This class
only exposes the post-compile read/clear surface. Buffers are per-Worker
(``self._tl_buffers``) so concurrent ``boot_vllm`` calls don't collide. All
methods prefixed ``tl_`` to avoid colliding with vLLM ``Worker`` attributes.
"""
from __future__ import annotations

from typing import Dict, List

import torch


class TLWorkerExtension:
    """Mixed into vLLM's ``Worker`` via ``worker_extension_cls``."""

    _tl_hook_handles: list
    _tl_buffers: Dict[str, torch.Tensor]

    def tl_read_captures(self, prompt_lens: List[int]) -> Dict[str, torch.Tensor]:
        """Slice each capture buffer back to ``sum(prompt_lens)`` rows; CPU copies.

        Caller (VLLMDriver.forward) gates ``sum(prompt_lens) <= max_num_batched_tokens``
        before the RPC, so ``total`` is always within buffer bounds.
        """
        total = sum(prompt_lens)
        buffers: Dict[str, torch.Tensor] = getattr(self, "_tl_buffers", {})
        return {name: buf[:total].detach().cpu().clone() for name, buf in buffers.items()}

    def tl_remove_hooks(self) -> None:
        """Detach all capture hooks and drop buffer references. Idempotent."""
        for handle in getattr(self, "_tl_hook_handles", []):
            handle.remove()
        self._tl_hook_handles = []
        self._tl_buffers = {}
