"""Muse Glimmer (MuseGlimmerForConditionalGeneration) architecture adapter."""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MuseGlimmerArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for MuseGlimmerForConditionalGeneration models."""

    _testing_lm_attr = "model.language_model"
    _testing_wire_rotary = False

    # Sandwich norms rescale sublayer outputs, so folding them is not function-preserving.
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        self._extract_vision_dims(cfg)

        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"
        self.weight_processing_conversions: dict = {}

        self.component_mapping = {
            "vision_encoder": GeneralizedComponent(name="model.vision_tower"),
            "vision_projector": GeneralizedComponent(name="model.vision_projection"),
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.language_model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.language_model.layers",
                submodules={
                    "ln1": GeneralizedComponent(name="input_layernorm"),
                    "ln1_post": GeneralizedComponent(name="post_attention_layernorm"),
                    "ln2": GeneralizedComponent(name="pre_feedforward_layernorm"),
                    "ln2_post": GeneralizedComponent(name="post_feedforward_layernorm"),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "gate": LinearBridge(name="gate_proj"),
                        },
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                    ),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": GeneralizedComponent(name="model.language_model.norm"),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def apply_output_logits_transform(self, logits: torch.Tensor) -> torch.Tensor:
        multiplier = float(getattr(self.cfg, "output_multiplier", 1.0))
        return super().apply_output_logits_transform(logits * multiplier)
