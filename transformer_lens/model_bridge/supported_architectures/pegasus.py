"""Pegasus architecture adapter.

Google's PEGASUS summarization family (``PegasusForConditionalGeneration``):
pre-norm encoder-decoder with per-stack final LayerNorms (M2M100-style
layout), Marian-style deterministic sinusoidal position embeddings, and the
sqrt(d_model) embedding scale applied in the stack forward — so, as with
Marian and unlike M2M100/MBart, hooks on ``embed`` observe unscaled output.
"""

from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class PegasusArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for PegasusForConditionalGeneration models."""

    has_final_stack_norm = True
