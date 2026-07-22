"""Qwen3.5 multimodal (vision-language) adapter for ``Qwen3_5ForConditionalGeneration``.

Reuses the text-only Qwen3.5 hybrid backbone nested under ``model.language_model`` and adds
the vision tower (``model.visual``) + merger. The HF model runs the vision computation during
forward; this adapter only supplies the component mapping (hooks + weights).
"""

from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components import VisionProjectionBridge
from transformer_lens.model_bridge.generalized_components.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoderBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


class Qwen3_5MultimodalArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Full vision-language adapter for Qwen3_5ForConditionalGeneration."""

    _testing_lm_attr = "model.language_model"
    _testing_hybrid = True

    # Qwen3.5's image/video processor (Qwen3VLProcessor) requires torchvision.
    required_libraries: list[str] = ["torchvision"]
    required_libraries_group: str = "multimodal"

    def __init__(self, cfg: Any) -> None:
        setattr(cfg, "gated_q_proj", True)
        super().__init__(cfg, hybrid=True, lm_prefix="model.language_model")

        self.cfg.is_multimodal = True

        self._extract_vision_dims(cfg)
        self.components["vision_encoder"] = Qwen3_5VisionEncoderBridge(
            name="model.visual", config=self.cfg
        )
        self.components["vision_projector"] = VisionProjectionBridge(name="model.visual.merger")

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice query half from gated q_proj.weight (matcher is path-prefix-agnostic)."""
        return self._preprocess_gated_q_proj(state_dict, self.cfg.n_heads, self.cfg.d_head)
