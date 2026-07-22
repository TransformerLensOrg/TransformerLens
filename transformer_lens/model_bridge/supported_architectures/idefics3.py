"""Idefics3 architecture adapter.

Supports ``Idefics3ForConditionalGeneration`` (SmolVLM lineage — e.g.
ibm-granite/granite-docling-258M): a SigLIP-style vision transformer at
``model.vision_model``, a pixel-shuffle connector at ``model.connector``,
and a llama-shaped text model at ``model.text_model`` with a top-level
``lm_head``.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SiglipVisionEncoderBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)


class Idefics3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Idefics3ForConditionalGeneration models."""

    _testing_lm_attr = "model.text_model"
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Idefics3 architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True

        # Text model is llama-shaped (SmolLM2 in public checkpoints).
        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"

        self._extract_vision_dims(cfg)

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "vision_encoder": SiglipVisionEncoderBridge(name="model.vision_model", config=self.cfg),
            "vision_projector": VisionProjectionBridge(name="model.connector"),
            "embed": EmbeddingBridge(name="model.text_model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(
                name="model.text_model.rotary_emb", config=self.cfg
            ),
            "blocks": BlockBridge(
                name="model.text_model.layers",
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
            "ln_final": RMSNormalizationBridge(name="model.text_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
