"""Gemma3 Multimodal architecture adapter.

This adapter supports Gemma3ForConditionalGeneration, the vision-language
variant of Gemma 3 used by models like MedGemma.
"""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    ArithmeticTensorConversion,
    RearrangeTensorConversion,
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
    GatedMLPBridge,
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

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma3 multimodal architecture adapter."""
        super().__init__(cfg)

        # Mark this as a multimodal model
        self.cfg.is_multimodal = True

        # Language model configuration (same as text-only Gemma 3)
        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.attn_implementation = "eager"

        # Store vision-related config
        if hasattr(cfg, "vision_config"):
            self.cfg.vision_hidden_size = getattr(cfg.vision_config, "hidden_size", None)
            self.cfg.vision_num_layers = getattr(cfg.vision_config, "num_hidden_layers", None)
            self.cfg.vision_num_heads = getattr(cfg.vision_config, "num_attention_heads", None)

        # Store multimodal projection config
        self.cfg.mm_tokens_per_image = getattr(cfg, "mm_tokens_per_image", 256)

        # Weight processing conversions for the language model
        # Note: The language model weights are under "model.language_model.*"
        self.weight_processing_conversions = {
            # Q/K/V weight conversions for language model
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
                ),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
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
            "rotary_emb_local": RotaryEmbeddingBridge(name="model.language_model.rotary_emb_local"),
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
            "ln_final": RMSNormalizationBridge(name="model.language_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Setup hook compatibility for Gemma3 multimodal models.

        Applies embedding scaling like text-only Gemma 3.

        Args:
            bridge: The TransformerBridge instance
        """
        from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
            BaseTensorConversion,
        )

        class EmbeddingScaleConversion(BaseTensorConversion):
            """Scale embeddings by sqrt(d_model) for Gemma models."""

            def __init__(self, scale: float):
                super().__init__()
                self.scale = scale

            def handle_conversion(self, input_value: Any, *full_context: Any) -> Any:
                """Scale the embedding output."""
                return input_value * self.scale

            def revert(self, input_value: Any, *full_context: Any) -> Any:
                """Unscale the embedding output (for user modifications)."""
                return input_value / self.scale

        # Apply scaling to embed.hook_out
        if hasattr(bridge, "embed") and hasattr(bridge.embed, "hook_out"):
            scale_factor = self.cfg.d_model**0.5
            bridge.embed.hook_out.hook_conversion = EmbeddingScaleConversion(scale_factor)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Gemma-3 multimodal component testing.

        The language model uses dual RoPE (global + local) like text-only Gemma 3.

        Args:
            hf_model: The HuggingFace Gemma-3 multimodal model instance
            bridge_model: The TransformerBridge model (if available)
        """
        # Get rotary embedding instances from the language model
        language_model = hf_model.model.language_model
        rotary_emb_local = language_model.rotary_emb_local

        # Force HF model to use "eager" attention
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        # Also set on text config
        if hasattr(hf_model.config, "text_config"):
            hf_model.config.text_config._attn_implementation = "eager"

        # Set on all language model attention layers
        if hasattr(language_model, "layers"):
            for layer in language_model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary_emb on actual bridge instances if available
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb_local)

                    # Enable native autograd for q_norm/k_norm
                    if hasattr(block.attn, "original_component"):
                        hf_attn = block.attn.original_component
                        if hasattr(hf_attn, "q_norm"):
                            hf_attn.q_norm.use_native_layernorm_autograd = True
                        if hasattr(hf_attn, "k_norm"):
                            hf_attn.k_norm.use_native_layernorm_autograd = True

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb_local)
