"""Blenderbot (``BlenderbotForConditionalGeneration``) adapter: Pegasus-style
pre-norm encoder-decoder with learned pos-embeds and baked-in embedding scale (so
embed hooks see already-scaled output); asymmetric stacks, n_layers follows the decoder."""

from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class BlenderbotArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for BlenderbotForConditionalGeneration models."""

    require_symmetric_layers = False
    n_layers_from = "decoder"
    has_final_stack_norm = True
