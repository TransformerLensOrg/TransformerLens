"""Florence-2 architecture adapter.

Microsoft's Florence-2 (``Florence2ForConditionalGeneration``): a DaViT
vision backbone at ``model.vision_tower`` whose projected features are
scattered into image-placeholder tokens, feeding a BART encoder-decoder at
``model.language_model``. The text stack reuses the BART mapping with
re-prefixed paths; the vision tower and projector are opaque delegated
components (window/channel-attention DaViT has no TL analogue).
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.bart import (
    BartArchitectureAdapter,
)

_TEXT_PREFIX = "model.language_model."


class Florence2ArchitectureAdapter(BartArchitectureAdapter):
    """Architecture adapter for Florence2ForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Florence-2 architecture adapter."""
        super().__init__(cfg)
        assert self.component_mapping is not None

        self.cfg.is_multimodal = True
        if hasattr(cfg, "vision_config"):
            self.cfg.vision_hidden_size = getattr(cfg.vision_config, "hidden_size", None)

        # Re-prefix the BART text stack under model.language_model.
        for component in self.component_mapping.values():
            if component.name and component.name.startswith("model."):
                component.name = _TEXT_PREFIX + component.name[len("model.") :]

        self.component_mapping["vision_encoder"] = GeneralizedComponent(name="model.vision_tower")
        self.component_mapping["vision_projector"] = GeneralizedComponent(
            name="model.multi_modal_projector"
        )
