"""Base class for vLLM overlays."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class AdapterOverlay:
    """A vLLM overlay for a single architecture family.

    :meth:`capture_specs` is called BEFORE ``LLM(...)`` to register the dot-paths
    and output widths the plugin should pre-allocate GPU buffers for. The plugin
    reads these during ``Worker.load_model`` so capture hooks are present when
    ``torch.compile`` traces the model.

    :meth:`nonfiring_hooks` enumerates hooks that vLLM's fused kernels prevent
    from firing; surfaced as a single boot-time warning.
    """

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        """Return ``{canonical_hook_name: (dot_path_in_vllm_model, output_width)}``."""
        raise NotImplementedError

    def nonfiring_hooks(self) -> List[str]:
        """Canonical hook names that vLLM's fused kernels cannot expose."""
        return []
