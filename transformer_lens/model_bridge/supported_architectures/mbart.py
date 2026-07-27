"""MBart (``MBartForConditionalGeneration``) adapter: Bart layout but PRE-norm with
an extra final LayerNorm per stack; embedding scale is baked in (embed hooks see it)."""

from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class MBartArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for MBartForConditionalGeneration models."""

    has_layernorm_embedding = True
    has_final_stack_norm = True
