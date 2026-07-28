"""M2M100 architecture adapter.

Covers Meta's M2M100 and NLLB-200 translation families
(``M2M100ForConditionalGeneration``). Same q/k/v/out_proj + fc1/fc2 layout as
Bart/Marian, but PRE-norm (LayerNorm before attention and MLP) with an extra
final LayerNorm after each stack, deterministic sinusoidal positions with a
padding-aware offset, and the sqrt(d_model) embedding scale baked into
``M2M100ScaledWordEmbedding`` itself — so unlike Marian, hooks on ``embed``
observe the already-scaled output.
"""

from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class M2M100ArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for M2M100ForConditionalGeneration (M2M100 / NLLB) models."""

    has_final_stack_norm = True
