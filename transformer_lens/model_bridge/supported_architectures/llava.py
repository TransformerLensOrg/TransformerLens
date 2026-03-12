"""LLava architecture adapter.

This adapter supports LlavaForConditionalGeneration, the vision-language
model combining a CLIP vision encoder with a LLaMA language model.
"""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    CLIPVisionEncoderBridge,
    EmbeddingBridge,
    GatedMLPBridge,
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

    def __init__(self, cfg: Any) -> None:
        """Initialize the LLava architecture adapter."""
        super().__init__(cfg)

        # Mark this as a multimodal model
        self.cfg.is_multimodal = True

        # Language model configuration (same as LLaMA)
        self.cfg.gated_mlp = True
        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.attn_implementation = "eager"
        self.cfg.final_rms = True
        self.cfg.attn_only = False
        self.cfg.eps_attr = "variance_epsilon"

        # GQA support
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        # Store vision-related config
        if hasattr(cfg, "vision_config"):
            self.cfg.vision_hidden_size = getattr(cfg.vision_config, "hidden_size", None)
            self.cfg.vision_num_layers = getattr(cfg.vision_config, "num_hidden_layers", None)
            self.cfg.vision_num_heads = getattr(cfg.vision_config, "num_attention_heads", None)

        # Weight processing conversions (same as LLaMA - Q/K/V/O rearrangements)
        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads,
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads,
                ),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
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
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for LLava component testing.

        LLava uses a LLaMA language backbone with RoPE. We set the rotary_emb
        reference on all attention bridge instances for component testing.

        Args:
            hf_model: The HuggingFace LLava model instance
            bridge_model: The TransformerBridge model (if available)
        """
        # Get rotary embedding instance from the language model
        language_model = hf_model.model.language_model
        rotary_emb = language_model.rotary_emb

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
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
