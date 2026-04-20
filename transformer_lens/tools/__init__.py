"""TransformerLens tools package.

This package contains utilities and tools for working with TransformerLens,
including the model registry for discovering compatible HuggingFace models.

Subpackages:
    - model_registry: Tools for discovering and documenting supported models
"""

from . import model_registry

__all__ = ["model_registry"]
