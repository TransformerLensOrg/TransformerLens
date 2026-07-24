"""TransformerLens tools package.

This package contains utilities and tools for working with TransformerLens,
including the model registry for discovering compatible HuggingFace models.

Subpackages:
    - analysis: High-level interpretability analyses (e.g. Direct Logit Attribution)
    - model_registry: Tools for discovering and documenting supported models

Subpackages load lazily (PEP 562): ``analysis`` imports HookedTransformer, so an
eager import here would create a cycle for core modules that consume
``model_registry`` data at import time.
"""

import importlib
from typing import Any

_SUBMODULES = ("analysis", "model_registry")

__all__ = ["analysis", "model_registry"]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULES))
