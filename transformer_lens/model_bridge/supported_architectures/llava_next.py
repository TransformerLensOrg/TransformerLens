"""LLava-NeXT architecture adapter.

LlavaNextForConditionalGeneration shares the same module hierarchy as
LlavaForConditionalGeneration (vision_tower, multi_modal_projector,
language_model, lm_head).  The differences — dynamic high-res image
tiling and an image_newline parameter — are handled internally by the
HuggingFace forward() and are transparent to the bridge.
"""

from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)


class LlavaNextArchitectureAdapter(LlavaArchitectureAdapter):
    """Architecture adapter for LLaVA-NeXT (1.6) models."""

    pass
