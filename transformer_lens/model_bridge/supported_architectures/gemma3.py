"""Gemma3 architecture adapter."""


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
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)


class Gemma3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Gemma3 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Gemma3 architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        self.cfg.normalization_type = "RMS"

        # Gemma 3 uses rotary positional embeddings (dual RoPE)
        self.cfg.positional_embedding_type = "rotary"

        # Use eager attention to support output_attentions for hook_attn_scores and hook_pattern
        # SDPA doesn't support output_attentions, which is required for HookedTransformer compatibility
        self.cfg.attn_implementation = "eager"

        self.weight_processing_conversions = {
            # Note: Gemma3 scales embeddings by sqrt(d_model) in the forward pass.
            # This is handled in setup_hook_compatibility() which applies the scaling
            # to hook_embed output at runtime, matching HuggingFace's behavior.
            # We do NOT scale the stored weights here.
            #
            # Q/K/V weight conversions
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(
                        self.cfg,
                        "n_key_value_heads",
                        self.cfg.n_heads,
                    ),
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(
                        self.cfg,
                        "n_key_value_heads",
                        self.cfg.n_heads,
                    ),
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
            # Gemma-3 also has q_norm and k_norm in attention
            "blocks.{i}.attn.q_norm.weight": ParamProcessingConversion(
                tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0),
            ),
            "blocks.{i}.attn.k_norm.weight": ParamProcessingConversion(
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
            # Unembed weight conversion - transpose from [vocab, d_model] to [d_model, vocab]
            "unembed.weight": ParamProcessingConversion(
                tensor_conversion=TransposeTensorConversion(),
            ),
            # Note: Gemma-3 does NOT have biases on attention projections (q/k/v/o_proj.bias are all None)
            # No bias conversions needed
        }

        # Set up component mapping with actual bridge instances
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "rotary_emb_local": RotaryEmbeddingBridge(name="model.rotary_emb_local"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    # All Gemma-3 normalizations use simple RMSNorm pass-through
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
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Setup hook compatibility for Gemma3 models.

        Gemma3 scales embeddings by sqrt(d_model) in its forward pass,
        but the HuggingFace embed_tokens layer doesn't include this scaling.
        We need to apply it to hook_embed to match HookedTransformer behavior.

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
        """Set up rotary embedding references and native autograd for Gemma-3 component testing.

        Gemma-3 uses dual RoPE (global + local). We set local RoPE (used by 85% of layers)
        on all attention bridge instances for component testing.

        We also enable use_native_layernorm_autograd on all normalization bridges to ensure
        they delegate to HuggingFace's exact implementation instead of using manual computation.

        Additionally, we force the HF model to use "eager" attention to match the bridge's
        implementation. The bridge uses "eager" to support output_attentions for hooks, while
        HF defaults to "sdpa". These produce mathematically equivalent results but with small
        numerical differences due to different implementations.

        Note: Layers 5, 11, 17, 23 use global RoPE but will use local in component tests.
        This is an acceptable tradeoff given the shared-instance constraint.

        Args:
            hf_model: The HuggingFace Gemma-3 model instance
            bridge_model: The TransformerBridge model (if available, set rotary_emb on actual instances)
        """
        # Get rotary embedding instances from the model
        rotary_emb_local = hf_model.model.rotary_emb_local  # Used by 22/26 layers

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
                    block.attn.set_rotary_emb(rotary_emb_local)

                    # Enable native autograd for q_norm/k_norm to match HF exactly
                    if hasattr(block.attn, "original_component"):
                        hf_attn = block.attn.original_component
                        if hasattr(hf_attn, "q_norm"):
                            hf_attn.q_norm.use_native_layernorm_autograd = True
                        if hasattr(hf_attn, "k_norm"):
                            hf_attn.k_norm.use_native_layernorm_autograd = True

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb_local)
