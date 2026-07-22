"""Gemma 4 text-only architecture adapter.

``Gemma4ForCausalLM``: the bare Gemma4TextModel with an LM head — the same
decoder the multimodal adapter maps, but living at ``model.*`` directly
(no ``language_model`` nesting) and with no vision components.
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.gemma4 import (
    Gemma4ArchitectureAdapter,
)

_MM_PREFIX = "model.language_model."


class Gemma4TextArchitectureAdapter(Gemma4ArchitectureAdapter):
    """Architecture adapter for Gemma4ForCausalLM models."""

    _testing_wire_rotary = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma 4 text-only architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = False
        self.components.pop("vision_encoder", None)
        self.components.pop("vision_projector", None)
        self._reprefix_components(_MM_PREFIX, "model.")
