"""Qwen2-Audio architecture adapter.

``Qwen2AudioForConditionalGeneration``: a Whisper-style audio encoder at
``model.audio_tower``, a linear projector at ``model.multi_modal_projector``,
and a Qwen2 text decoder at ``model.language_model`` with a top-level
``lm_head`` (the transformers >= 5.13 layout). Text-only forwards work without
audio; audio features enter via ``input_features``.
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
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class Qwen2AudioArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen2AudioForConditionalGeneration models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen2-Audio architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True

        # Text model is Qwen2 (RMS + RoPE + GQA + gated MLP, attention biases).
        self._set_rms_rotary_defaults()
        self.cfg.default_prepend_bos = False

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            # The Whisper-style encoder and projector are wrapped opaquely
            # (hook_in/hook_out only) — audio features are injected by the HF
            # forward when input_features is passed.
            "audio_encoder": GeneralizedComponent(name="model.audio_tower"),
            "audio_projector": GeneralizedComponent(name="model.multi_modal_projector"),
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(
                name="model.language_model.rotary_emb", config=self.cfg
            ),
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

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire the text model's rotary embedding through to attention bridges."""
        rotary_emb = hf_model.model.language_model.rotary_emb

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
