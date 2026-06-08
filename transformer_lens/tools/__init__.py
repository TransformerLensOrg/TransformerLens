"""TransformerLens tools package.

This package contains utilities and tools for working with TransformerLens,
including the model registry for discovering compatible HuggingFace models.

Subpackages:
    - analysis: Interpretability analyses such as Direct Logit Attribution
    - model_registry: Tools for discovering and documenting supported models
"""

from . import analysis, model_registry

__all__ = ["analysis", "model_registry"]
