"""Qwen2-Audio architecture adapter.

``Qwen2AudioForConditionalGeneration``: a Whisper-style audio encoder at
``audio_tower``, a linear projector at ``multi_modal_projector``, and a full
Qwen2ForCausalLM at ``language_model`` (so text paths live under
``language_model.model.*`` with ``language_model.lm_head``). Text-only
forwards work without audio; audio features enter via ``input_features``.
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
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.default_prepend_bos = False

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            # The Whisper-style encoder and projector are wrapped opaquely
            # (hook_in/hook_out only) — audio features are injected by the HF
            # forward when input_features is passed.
            "audio_encoder": GeneralizedComponent(name="audio_tower"),
            "audio_projector": GeneralizedComponent(name="multi_modal_projector"),
            "embed": EmbeddingBridge(name="language_model.model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(
                name="language_model.model.rotary_emb", config=self.cfg
            ),
            "blocks": BlockBridge(
                name="language_model.model.layers",
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
            "ln_final": RMSNormalizationBridge(name="language_model.model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="language_model.lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire the text model's rotary embedding through to attention bridges."""
        rotary_emb = hf_model.language_model.model.rotary_emb

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
