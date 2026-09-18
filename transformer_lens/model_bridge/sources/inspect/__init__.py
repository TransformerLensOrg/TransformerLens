"""Inspect driver: turn an ``inspect_ai``-served model into a TransformerLens bridge.

We ship our own HF-backed provider (``provider.py``) and a torch-free consumer
(``driver.py``).

Maintainer note: this package is named ``inspect``, shadowing the stdlib module.
Only ever import the stdlib ``inspect`` via an absolute import from outside this
package; never write a bare ``import inspect`` inside these modules.
"""
from __future__ import annotations

from typing import Any

from .source import boot_inspect

__all__ = ["activations_column", "boot_inspect", "capture_activations", "turn_activations"]


def __getattr__(name: str) -> Any:
    # Lazy so importing the package stays inspect_ai-free; eval.py imports inspect_ai.
    if name in ("capture_activations", "activations_column", "turn_activations"):
        from . import eval as _eval

        return getattr(_eval, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
