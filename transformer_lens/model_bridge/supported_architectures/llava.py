"""LLava architecture adapter.

This adapter supports LlavaForConditionalGeneration, the vision-language
model combining a CLIP vision encoder with a LLaMA language model.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CLIPVisionEncoderBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SiglipVisionEncoderBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)


class LlavaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for LLava multimodal models (LlavaForConditionalGeneration).

    This adapter handles vision-language models like LLava 1.5.
    The model structure is:
    - model.vision_tower: CLIP vision encoder
    - model.multi_modal_projector: 2-layer MLP (Linear -> GELU -> Linear)
    - model.language_model: LlamaForCausalLM
      - model.language_model.model.embed_tokens
      - model.language_model.model.layers[]: LLaMA transformer blocks
      - model.language_model.model.norm
      - model.language_model.lm_head

    The language model component follows the same patterns as LlamaArchitectureAdapter.
    """

    _testing_lm_attr = "model.language_model"

    def __init__(self, cfg: Any) -> None:
        """Initialize the LLava architecture adapter."""
        super().__init__(cfg)

        # Mark this as a multimodal model
        self.cfg.is_multimodal = True

        # Language model configuration (same as LLaMA)
        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"

        # Store vision-related config
        self._extract_vision_dims(cfg)

        # Weight processing conversions (same as LLaMA - Q/K/V/O rearrangements)
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        # Select vision encoder bridge based on vision model type
        vision_cfg = getattr(cfg, "vision_config", None)
        vision_type = getattr(vision_cfg, "model_type", "clip_vision_model")
        vision_bridge: GeneralizedComponent
        if vision_type in ("siglip_vision_model", "siglip"):
            vision_bridge = SiglipVisionEncoderBridge(name="model.vision_tower", config=self.cfg)
        else:
            vision_bridge = CLIPVisionEncoderBridge(name="model.vision_tower", config=self.cfg)

        # Component mapping for the full multimodal model
        # LlavaForConditionalGeneration wraps:
        #   model.vision_tower, model.multi_modal_projector, model.language_model
        # The language_model is a *Model (LlamaModel, Qwen2Model, MistralModel)
        # with embed_tokens, layers, norm, rotary_emb directly (no nested .model).
        # lm_head sits at the top level of LlavaForConditionalGeneration.
        self.component_mapping = {
            # Vision components
            "vision_encoder": vision_bridge,
            "vision_projector": VisionProjectionBridge(name="model.multi_modal_projector"),
            # Language model components
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.language_model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.language_model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.language_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
