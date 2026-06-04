"""Qwen3.5 multimodal (vision-language) architecture adapter.

Supports ``Qwen3_5ForConditionalGeneration`` — the full vision-language Qwen3.5 model:
- model.visual: Qwen3.5 vision tower (patch_embed, pos_embed, blocks, merger)
- model.language_model: hybrid linear-attention/full-attention text backbone
- lm_head: output projection

The text backbone is identical to the text-only ``Qwen3_5ArchitectureAdapter`` (hybrid
GatedDeltaNet + optional full attention with gated q_proj), so this adapter reuses the
Qwen3 block construction and nests it under ``model.language_model``. The HF model performs
the actual vision computation during forward; this adapter supplies the component mapping
(hooks + weights) for both the vision and language paths.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components import (
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.qwen3_5_vision_encoder import (
    Qwen3_5VisionEncoderBridge,
)
from transformer_lens.model_bridge.supported_architectures.qwen3 import (
    Qwen3ArchitectureAdapter,
)


class Qwen3_5MultimodalArchitectureAdapter(Qwen3ArchitectureAdapter):
    """Architecture adapter for Qwen3.5 multimodal models (Qwen3_5ForConditionalGeneration).

    Reuses the Qwen3 hybrid text backbone (nested under model.language_model) and adds a
    decomposed vision tower (model.visual) plus its merger as the vision projector.
    """

    def __init__(self, cfg: Any) -> None:
        setattr(cfg, "gated_q_proj", True)
        # Build the hybrid text backbone nested under model.language_model directly.
        super().__init__(cfg, hybrid=True, lm_prefix="model.language_model")

        self.cfg.is_multimodal = True

        # Store vision-related config (Qwen vision config uses depth/num_heads).
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            self.cfg.vision_hidden_size = getattr(vision_cfg, "hidden_size", None)
            self.cfg.vision_num_layers = getattr(vision_cfg, "depth", None) or getattr(
                vision_cfg, "num_hidden_layers", None
            )
            self.cfg.vision_num_heads = getattr(vision_cfg, "num_heads", None) or getattr(
                vision_cfg, "num_attention_heads", None
            )

        # Add the vision tower and its merger (the vision->text projector).
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
