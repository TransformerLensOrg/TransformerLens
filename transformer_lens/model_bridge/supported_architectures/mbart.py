"""MBart architecture adapter.

Multilingual BART (``MBartForConditionalGeneration``): Bart's learned
positional embeddings (offset 2) and per-stack ``layernorm_embedding``, but
PRE-norm layers and an extra final LayerNorm after each stack (M2M100-style).
The sqrt(d_model) embedding scale is baked into ``MBartScaledWordEmbedding``,
so embed hooks observe the scaled output.
"""

from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class MBartArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for MBartForConditionalGeneration models."""

    has_layernorm_embedding = True
    has_final_stack_norm = True
