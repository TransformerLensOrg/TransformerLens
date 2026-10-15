"""Deprecated location of direct path patching, now in :mod:`transformer_lens.patching`."""

import warnings
from typing import Any

from transformer_lens import patching

_MOVED = ("get_act_patch_direct_path", "get_act_patch_direct_path_all_sources")

# Star-import resolves these through __getattr__, so it binds the moved names and warns.
__all__ = list(_MOVED)


def __getattr__(name: str) -> Any:
    if name not in _MOVED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # FutureWarning is shown by default: the results changed, not only the import path.
    warnings.warn(
        f"{__name__}.{name} is deprecated; import it from transformer_lens.patching. It now "
        "computes the exact edge patch instead of a frozen-LayerNorm-scale approximation, so "
        "its results differ from earlier releases.",
        FutureWarning,
        stacklevel=2,
    )
    return getattr(patching, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_MOVED))
