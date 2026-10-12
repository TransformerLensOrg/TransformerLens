"""Bridge-native Qwen natural-language autoencoder inference.

Use raw fp32 models with eager attention. Reconstruction similarity measures
vector preservation, not human-readable faithfulness. Models load lazily; the
returned target is caller-owned and AV/AR resources are context-owned.
"""

from transformer_lens.tools.analysis._nla import NaturalLanguageAutoencoder
from transformer_lens.tools.analysis._nla_contracts import (
    ActivationProvenance,
    ExplanationStatus,
    NLAActivation,
    NLAExplanation,
    NLAReconstruction,
    NLAReconstructionScore,
    NLASamplingConfig,
    SourceProvenance,
)

__all__ = [
    "ActivationProvenance",
    "ExplanationStatus",
    "NLAActivation",
    "NLAExplanation",
    "NLAReconstruction",
    "NLAReconstructionScore",
    "NLASamplingConfig",
    "NaturalLanguageAutoencoder",
    "SourceProvenance",
]
