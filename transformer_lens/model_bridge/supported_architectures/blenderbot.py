"""Blenderbot architecture adapter.

Meta's Blenderbot dialogue family (``BlenderbotForConditionalGeneration``):
Pegasus-style pre-norm encoder-decoder with per-stack final LayerNorms but
LEARNED positional embeddings (no offset). Unlike Marian/Pegasus, the
sqrt(d_model) embedding scale is baked into BlenderbotScaledWordEmbedding
(MBart/M2M100-style), so embed hooks observe ALREADY-SCALED output. Public checkpoints are asymmetric (small
encoder, large decoder: 2/12 on 400M-distill, 2/24 on 3B), so unlike the
Bart/Marian adapters only heads and FFN width are required to match;
``cfg.n_layers`` follows the decoder stack.
"""

from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class BlenderbotArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for BlenderbotForConditionalGeneration models."""

    require_symmetric_layers = False
    n_layers_from = "decoder"
    has_final_stack_norm = True
