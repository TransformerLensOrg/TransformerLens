"""Gemma3 Multimodal architecture adapter.

This adapter supports Gemma3ForConditionalGeneration, the vision-language
variant of Gemma 3 used by models like MedGemma.
"""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    ArithmeticTensorConversion,
    TransposeTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.arithmetic_tensor_conversion import (
    OperationTypes,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SiglipVisionEncoderBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)


class Gemma3MultimodalArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma3 multimodal models (Gemma3ForConditionalGeneration).

    This adapter handles vision-language models like Gemma 3 4B/12B/27B and MedGemma.
    The model structure is:
    - model.vision_tower: SigLIP vision encoder
    - model.multi_modal_projector: Projects vision embeddings to language space
    - model.language_model: Gemma3TextModel (same as text-only Gemma 3)
    - lm_head: Output projection

    The language model component follows the same patterns as Gemma3ArchitectureAdapter.
    """

    _testing_lm_attr = "model.language_model"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma3 multimodal architecture adapter."""
        super().__init__(cfg)

        # Mark this as a multimodal model
        self.cfg.is_multimodal = True

        # Language model configuration (same as text-only Gemma 3)
        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        # Gemma models use (1.0 + weight) in RMSNorm instead of just weight.
        # Without this, fold_ln sets identity to 1.0 instead of 0.0, causing 2x scaling.
        self.cfg.rmsnorm_uses_offset = True
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.attn_implementation = "eager"

        # Store vision-related config
        self._extract_vision_dims(cfg)

        # Store multimodal projection config
        self.cfg.mm_tokens_per_image = getattr(cfg, "mm_tokens_per_image", 256)

        # Weight processing conversions for the language model
        # Note: The language model weights are under "model.language_model.*"
        self.weight_processing_conversions = {
            # Q/K/V weight conversions for language model
            **self._qkvo_weight_conversions(),
            # RMSNorm weight conversions - Gemma adds 1.0 to weights
            "blocks.{i}.ln1.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.ln1_post.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.ln2.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.ln2_post.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "ln_final.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # Gemma-3 q_norm and k_norm in attention
            "blocks.{i}.attn.q_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.attn.k_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            # MLP weight conversions
            "blocks.{i}.mlp.gate.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "blocks.{i}.mlp.in.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "blocks.{i}.mlp.out.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            # Unembed weight conversion
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
        }

        # Component mapping for the full multimodal model
        # Note: We use distinct TL names (vision_encoder, vision_projector) to avoid
        # conflicting with HF model attribute names (vision_tower, multi_modal_projector)
        self.component_mapping = {
            # Vision components
            "vision_encoder": SiglipVisionEncoderBridge(name="model.vision_tower", config=self.cfg),
            "vision_projector": VisionProjectionBridge(name="model.multi_modal_projector"),
            # Language model components (under model.language_model)
            "embed": EmbeddingBridge(name="model.language_model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.language_model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.language_model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_attention_layernorm", config=self.cfg
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="pre_feedforward_layernorm", config=self.cfg
                    ),
                    "ln2_post": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
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
                    ),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.language_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire rotary + eager, then enable native autograd on the Q/K norms."""
        super().setup_component_testing(hf_model, bridge_model)
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                hf_attn = getattr(getattr(block, "attn", None), "original_component", None)
                if hf_attn is None:
                    continue
                if hasattr(hf_attn, "q_norm"):
                    hf_attn.q_norm.use_native_layernorm_autograd = True
                if hasattr(hf_attn, "k_norm"):
                    hf_attn.k_norm.use_native_layernorm_autograd = True
