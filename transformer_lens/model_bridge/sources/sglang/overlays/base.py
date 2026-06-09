"""Base class for SGLang overlays — mirrors the vLLM overlay base verbatim."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class AdapterOverlay:
    """SGLang overlay for one architecture family."""

    def capture_specs(self, hf_config: Any) -> Dict[str, Tuple[str, int]]:
        """``{canonical_hook_name: (dot_path_in_sglang_model, output_width)}``."""
        raise NotImplementedError

    def nonfiring_hooks(self) -> List[str]:
        """Canonical hook names SGLang's fused kernels can't expose."""
        return []
