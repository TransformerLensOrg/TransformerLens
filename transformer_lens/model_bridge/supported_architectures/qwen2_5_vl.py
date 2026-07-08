"""Qwen2.5-VL architecture adapter.

Alibaba's Qwen2.5-VL (``Qwen2_5_VLForConditionalGeneration``): a windowed
ViT at ``model.visual`` (window attention with a few full-attention
blocks, RMS block norms, gated vision MLP, 2D rotary) whose patch merger
feeds a Qwen2-layout text decoder at ``model.language_model``. Text
attention uses mRoPE — three position streams (temporal/height/width)
split across rotary channels — so the generic RoPE reconstruction would
be text-only-correct but wrong for image runs; attention stays HF-native.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.qwen3_5_vision_encoder import (
    Qwen3_5VisionBlockBridge,
    Qwen3_5VisionEncoderBridge,
)


class Qwen2_5_VLArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen2_5_VLForConditionalGeneration models."""

    required_libraries: list[str] = ["torchvision"]
    required_libraries_group: str = "multimodal"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen2.5-VL architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"
        # Qwen tokenizers have no BOS; the prepend fallback would inject
        # <|im_end|>, which reads as an ended turn.
        self.cfg.default_prepend_bos = False

        self._extract_vision_dims(cfg)

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            # Qwen2.5-VL's tower has a rotary embedding where Qwen3.5 has a
            # learned pos_embed, and a gated vision MLP instead of fc1/fc2.
            "vision_encoder": Qwen3_5VisionEncoderBridge(
                name="model.visual",
                config=self.cfg,
                submodules={
                    "pos_embed": GeneralizedComponent(name="rotary_pos_emb"),
                    "blocks": Qwen3_5VisionBlockBridge(
                        name="blocks",
                        submodules={
                            "mlp": GeneralizedComponent(
                                name="mlp",
                                submodules={
                                    "gate": LinearBridge(name="gate_proj"),
                                    "in": LinearBridge(name="up_proj"),
                                    "out": LinearBridge(name="down_proj"),
                                },
                            ),
                        },
                    ),
                },
            ),
            "vision_projector": VisionProjectionBridge(name="model.visual.merger"),
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.language_model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # mRoPE (3-section multimodal rotary) lives in HF's forward.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                    ),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.language_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
