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

    # Qwen3.5's image/video processor (Qwen3VLProcessor) requires torchvision.
    required_libraries: list[str] = ["torchvision"]
    required_libraries_group: str = "multimodal"

    def __init__(self, cfg: Any) -> None:
        setattr(cfg, "gated_q_proj", True)
        super().__init__(cfg, hybrid=True, lm_prefix="model.language_model")

        self.cfg.is_multimodal = True

        self._extract_vision_dims(cfg)

        assert self.component_mapping is not None  # built by super().__init__
        self.component_mapping["vision_encoder"] = Qwen3_5VisionEncoderBridge(
            name="model.visual", config=self.cfg
        )
        self.component_mapping["vision_projector"] = VisionProjectionBridge(
            name="model.visual.merger"
        )

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slice query half from gated q_proj.weight (matcher is path-prefix-agnostic)."""
        return self._preprocess_gated_q_proj(state_dict, self.cfg.n_heads, self.cfg.d_head)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set eager attn and rotary_emb refs for the nested language model.

        Hybrid: only full-attention layers have ``self_attn``/``attn``; linear-attention
        layers are skipped.
        """
        language_model = hf_model.model.language_model
        rotary_emb = language_model.rotary_emb

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"
        if hasattr(hf_model.config, "text_config"):
            hf_model.config.text_config._attn_implementation = "eager"

        if hasattr(language_model, "layers"):
            for layer in language_model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if "attn" in block._modules:
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls.
        try:
            attn_template = self.get_generalized_component("blocks.0.attn")
            attn_template.set_rotary_emb(rotary_emb)
        except (ValueError, AttributeError, KeyError):
            pass
