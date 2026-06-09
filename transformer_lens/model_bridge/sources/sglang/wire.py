"""Overlay capture specs → ``ServerArgs.forward_hooks`` JSON entries.

Each spec becomes one dict with an fnmatch ``target_modules`` pattern + our factory
dotted-path. Decoder layers are flagged ``materialize=True`` so the hook sums the
``(mlp_delta, residual)`` tuple — matches HF's ``hook_out`` semantics."""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Tuple

_DECODER_LAYER_PATH = re.compile(r"^model\.layers\.\d+$")
_HOOK_FACTORY = "transformer_lens.model_bridge.sources.sglang.hooks:make_capture_hook"


def fresh_channel() -> str:
    """Unique ``ipc://`` address for one driver↔worker session."""
    return f"ipc:///tmp/tl_sglang_{uuid.uuid4().hex}.sock"


def build_forward_hooks(
    capture_specs: Dict[str, Tuple[str, int]],
    channel: str,
) -> List[Dict[str, Any]]:
    """``{canonical_name: (dot_path, width)}`` → ``ServerArgs.forward_hooks`` list."""
    return [
        {
            "name": f"tl_{canonical_name}",
            "target_modules": [dot_path],
            "hook_factory": _HOOK_FACTORY,
            "config": {
                "canonical_name": canonical_name,
                "channel": channel,
                "materialize": bool(_DECODER_LAYER_PATH.match(dot_path)),
            },
        }
        for canonical_name, (dot_path, _width) in capture_specs.items()
    ]
