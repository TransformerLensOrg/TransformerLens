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
    GatedMLPBridge,
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

    def __init__(self, cfg: Any) -> None:
        """Initialize the Idefics3 architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True

        # Text model is llama-shaped (SmolLM2 in public checkpoints).
        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.attn_implementation = "eager"
        self.cfg.final_rms = True
        self.cfg.attn_only = False

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        if hasattr(cfg, "vision_config"):
            self.cfg.vision_hidden_size = getattr(cfg.vision_config, "hidden_size", None)
            self.cfg.vision_num_layers = getattr(cfg.vision_config, "num_hidden_layers", None)
            self.cfg.vision_num_heads = getattr(cfg.vision_config, "num_attention_heads", None)

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
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.text_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire the text model's rotary embedding through to attention bridges."""
        rotary_emb = hf_model.model.text_model.rotary_emb

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"
        if hasattr(hf_model.config, "text_config"):
            hf_model.config.text_config._attn_implementation = "eager"

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
