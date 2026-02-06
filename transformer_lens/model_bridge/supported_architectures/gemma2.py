"""Gemma2 architecture adapter."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

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
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Gemma2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma2 architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False

        # Gemma models were not trained with BOS tokens
        # self.cfg.default_prepend_bos = False
        self.cfg.uses_rms_norm = True
        # Gemma models use (1.0 + weight) in RMSNorm instead of just weight
        # See: https://github.com/huggingface/transformers/pull/29402
        self.cfg.rmsnorm_uses_offset = True

        # Gemma2 uses logit softcapping
        if hasattr(self.cfg, "final_logit_softcapping"):
            self.cfg.output_logits_soft_cap = self.cfg.final_logit_softcapping
        if hasattr(self.cfg, "attn_logit_softcapping"):
            self.cfg.attn_scores_soft_cap = self.cfg.attn_logit_softcapping

        # Note: n_key_value_heads is now automatically mapped from num_key_value_heads
        # by map_default_transformer_lens_config() in sources/transformers.py

        self.weight_processing_conversions = {
            # Embedding weight scaling - Gemma models scale embeddings by sqrt(d_model)
            "embed.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(
                    OperationTypes.MULTIPLICATION, self.cfg.d_model**0.5
                ),
            ),
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
            # RMSNorm weight conversions - Gemma adds 1.0 to weights before applying
            # See: https://github.com/huggingface/transformers/pull/29402
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
            # MLP weight conversions - transpose from [out, in] to [in, out]
            "blocks.{i}.mlp.gate.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "blocks.{i}.mlp.in.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            "blocks.{i}.mlp.out.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            # # Unembed weight conversion - transpose from [vocab, d_model] to [d_model, vocab]
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    # Gemma 2 uses RMSNorm for all normalization layers
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
                    # Gemma 2 uses PositionEmbeddingsAttentionBridge like Gemma 3
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
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Setup hook compatibility for Gemma2 models.

        Gemma2 scales embeddings by sqrt(d_model). The weights are pre-scaled via
        preprocess_weights(), but we still need to apply the scaling conversion to
        the hook output for proper hook functionality (so user modifications are
        correctly scaled/unscaled).

        Args:
            bridge: The TransformerBridge instance
        """
        # Apply embedding scaling conversion to hook output
        if hasattr(bridge, "embed") and hasattr(bridge.embed, "hook_out"):
            scale_factor = self.cfg.d_model**0.5
            bridge.embed.hook_out.hook_conversion = ArithmeticTensorConversion(
                OperationTypes.MULTIPLICATION, scale_factor
            )

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references and attention implementation for Gemma-2 component testing.

        Gemma-2 uses RoPE (Rotary Position Embeddings). We set the rotary_emb reference
        on all attention bridge instances for component testing.

        We also force the HF model to use "eager" attention to match the bridge's implementation.
        The bridge uses "eager" to support output_attentions for hooks, while HF defaults
        to "sdpa". These produce mathematically equivalent results but with small numerical
        differences due to different implementations.

        Args:
            hf_model: The HuggingFace Gemma-2 model instance
            bridge_model: The TransformerBridge model (if available, set rotary_emb on actual instances)
        """
        # Get rotary embedding instance from the model
        rotary_emb = hf_model.model.rotary_emb

        # Force HF model to use "eager" attention to match bridge implementation
        # Bridge uses "eager" to support output_attentions for hook compatibility
        # SDPA and eager are mathematically equivalent but have numerical differences
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        # Also set on all attention layers
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary_emb on actual bridge instances in bridge_model if available
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            # Set on each layer's actual attention bridge instance
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
