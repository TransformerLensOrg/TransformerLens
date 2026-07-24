"""Mistral 3 VLM (``Mistral3ForConditionalGeneration``) adapter: Llava layout with
the Pixtral vision tower delegated opaquely."""

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

        # Pixtral's 2D-RoPE block-diagonal attention has no CLIP-shaped bridge, so
        # the vision tower is delegated opaquely.
        self.components["vision_encoder"] = GeneralizedComponent(name="model.vision_tower")
