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

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma 4 text-only architecture adapter."""
        super().__init__(cfg)
        assert self.component_mapping is not None

        self.cfg.is_multimodal = False
        self.component_mapping.pop("vision_encoder", None)
        self.component_mapping.pop("vision_projector", None)
        for component in self.component_mapping.values():
            if component.name and component.name.startswith(_MM_PREFIX):
                component.name = "model." + component.name[len(_MM_PREFIX) :]

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention so bridge and HF match (sliding/full layer mix)."""
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"
        model = getattr(hf_model, "model", None)
        if model is not None and hasattr(model, "layers"):
            for layer in model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"
