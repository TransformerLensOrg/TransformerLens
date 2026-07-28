"""Llama 4 multimodal architecture adapter.

Meta's Llama 4 composite (``Llama4ForConditionalGeneration``): a Llama4
vision transformer at ``vision_model`` and a projector feeding the full
Llama4ForCausalLM at ``language_model`` (text stack at
``language_model.model.*`` with ``language_model.lm_head``). The vision
side is delegated opaquely; the text mapping is the Llama4 text adapter's,
re-prefixed.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.llama4 import (
    Llama4ArchitectureAdapter,
)


class Llama4MultimodalArchitectureAdapter(Llama4ArchitectureAdapter):
    """Architecture adapter for Llama4ForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Llama 4 multimodal architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        if hasattr(cfg, "vision_config"):
            self.cfg.vision_hidden_size = getattr(cfg.vision_config, "hidden_size", None)

        self._reprefix_components("model.", "language_model.model.")
        self._reprefix_components("lm_head", "language_model.lm_head")

        self.components["vision_encoder"] = GeneralizedComponent(name="vision_model")
        self.components["vision_projector"] = GeneralizedComponent(name="multi_modal_projector")
