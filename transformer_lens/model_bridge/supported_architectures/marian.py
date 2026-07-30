"""Marian architecture adapter.

Helsinki-NLP's opus-mt family (``MarianMTModel``): post-norm Bart layout with
no layernorm_embedding and no per-stack final norms, deterministic sinusoidal
positions, and the sqrt(d_model) embedding scale applied in the stack forward
(embed hooks observe unscaled output). A trained ``final_logits_bias`` is
added after lm_head inside HF's forward.
"""

from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class MarianArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for MarianMTModel models (Helsinki-NLP opus-mt family)."""
