"""TransformerLens tools package.

This package contains utilities and tools for working with TransformerLens,
including the model registry for discovering compatible HuggingFace models.

Subpackages:
    - model_registry: Tools for discovering and documenting supported models
    - analysis: High-level interpretability analyses (e.g. Direct Logit Attribution)
"""

from . import analysis, model_registry

__all__ = ["analysis", "model_registry"]
