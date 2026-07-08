"""Mistral 3 (Mistral-Small-3.x VLM) architecture adapter.

Mistral AI's Mistral3 (``Mistral3ForConditionalGeneration``,
Mistral-Small-3.1/3.2): a Pixtral vision tower and a patch-merging
projector feeding a Mistral text decoder — the same Llava layout, so the
Llava adapter applies with the vision tower swapped to an opaque
delegated component (Pixtral's 2D-RoPE block-diagonal attention has no
CLIP/SigLIP-shaped bridge).
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)


class Mistral3ArchitectureAdapter(LlavaArchitectureAdapter):
    """Architecture adapter for Mistral3ForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Mistral 3 architecture adapter."""
        super().__init__(cfg)

        # Pixtral's 2D-RoPE block-diagonal attention has no CLIP-shaped bridge.
        # The projector inherits Llava's VisionProjectionBridge; its extra
        # (image_features, image_sizes) positional flows through the *args
        # passthrough.
        self.components["vision_encoder"] = GeneralizedComponent(name="model.vision_tower")
