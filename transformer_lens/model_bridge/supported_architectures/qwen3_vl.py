"""Qwen3-VL architecture adapter.

Alibaba's Qwen3-VL (``Qwen3VLForConditionalGeneration``): a ViT tower at
``model.visual`` matching the Qwen3.5 vision layout (learned pos_embed +
2D rotary, qkv/proj attention, fc1/fc2 MLP) plus DeepStack — extra patch
mergers on early vision blocks whose features the text model injects
into the residual stream at visual token positions during the first
decoder layers. The injection itself is a tensor add inside the HF text
loop (not a module call); the per-level DeepStack mergers are wrapped so
their features are hookable at the source. Because the injection happens
BETWEEN block calls, blocks.{i}.hook_out omits it while blocks.{i+1}'s
input contains it on image runs — resid_post[i] != resid_pre[i+1] at
visual positions for the first DeepStack layers, and patching resid_post
there drops the injection (prefer blocks.{i+1}.hook_in). Text attention
is Qwen3-style (per-head QK RMS-norm) with interleaved mRoPE, HF-native.
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
    Qwen3_5VisionEncoderBridge,
)


class _DeepStackMergerBridge(GeneralizedComponent):
    """Per-level DeepStack patch merger (multi-scale visual features)."""

    is_list_item: bool = True


class Qwen3VLArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen3VLForConditionalGeneration models."""

    required_libraries: list[str] = ["torchvision"]
    required_libraries_group: str = "multimodal"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen3-VL architecture adapter."""
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
            # The Qwen3.5 vision bridge defaults (patch_embed, learned
            # pos_embed, qkv/proj + fc1/fc2 blocks) match this tower exactly;
            # Qwen3-VL adds the DeepStack merger stack on top.
            "vision_encoder": Qwen3_5VisionEncoderBridge(
                name="model.visual",
                config=self.cfg,
                submodules={
                    "deepstack_mergers": _DeepStackMergerBridge(name="deepstack_merger_list"),
                },
            ),
            "vision_projector": VisionProjectionBridge(name="model.visual.merger"),
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.language_model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # Interleaved mRoPE + per-head QK-norm live in HF's forward.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                    ),
                    "mlp": self._build_mlp_bridge(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.language_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def _build_mlp_bridge(self) -> Any:
        """Dense gated MLP; the MoE variant overrides this."""
        return self._gated_mlp()
