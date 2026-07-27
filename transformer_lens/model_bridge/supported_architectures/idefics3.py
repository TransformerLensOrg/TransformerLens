"""Idefics3 architecture adapter.

Supports ``Idefics3ForConditionalGeneration`` (SmolVLM lineage — e.g.
ibm-granite/granite-docling-258M): a SigLIP-style vision transformer at
``model.vision_model``, a pixel-shuffle connector at ``model.connector``,
and a llama-shaped text model at ``model.text_model`` with a top-level
``lm_head``.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components import (
    SiglipVisionEncoderBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)


class Idefics3ArchitectureAdapter(LlamaArchitectureAdapter):
    """Architecture adapter for Idefics3ForConditionalGeneration models."""

    _testing_lm_attr = "model.text_model"
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Idefics3 architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        # Text model is llama-shaped (SmolLM2 in public checkpoints).
        self.cfg.attn_implementation = "eager"
        self._extract_vision_dims(cfg)
        self._reprefix_components("model.", "model.text_model.")

        self.components["vision_encoder"] = SiglipVisionEncoderBridge(
            name="model.vision_model", config=self.cfg
        )
        self.components["vision_projector"] = VisionProjectionBridge(name="model.connector")
