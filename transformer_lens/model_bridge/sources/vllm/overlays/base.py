"""Base class for vLLM overlays."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class AdapterOverlay:
    """A vLLM overlay for a single architecture family.

    Two responsibilities, each phased on a different point in the boot lifecycle:

    1. :meth:`capture_specs` — called BEFORE ``LLM(...)`` to register the dot-paths
       and output widths the plugin should pre-allocate GPU buffers for. The plugin
       reads these during ``Worker.load_model`` so capture hooks are present when
       ``torch.compile`` traces the model.
    2. :meth:`apply` — called AFTER ``LLM(...)`` on the bridge's adapter to swap
       canonical components (e.g. split q/k/v) for fused-projection variants that
       match vLLM's module tree (``qkv_proj``, ``gate_up_proj``).

    :meth:`nonfiring_hooks` enumerates hooks that vLLM's fused kernels prevent
    from firing; surfaced as a single boot-time warning.
    """

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        """Return ``{canonical_hook_name: (dot_path_in_vllm_model, output_width)}``.

        Each entry produces one pre-allocated GPU buffer of shape
        ``(max_num_batched_tokens, output_width)`` that the corresponding
        forward hook copies into via ``buf[:n].copy_(t)``.
        """
        raise NotImplementedError

    def apply(self, adapter: Any) -> None:
        """Mutate ``adapter.component_mapping`` in place to match vLLM's module tree."""
        raise NotImplementedError

    def nonfiring_hooks(self) -> List[str]:
        """Canonical hook names that vLLM's fused kernels cannot expose."""
        return []
