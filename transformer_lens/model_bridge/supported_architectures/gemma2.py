"""Gemma2 architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
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

        # Note: n_key_value_heads is now automatically mapped from num_key_value_heads
        # by map_default_transformer_lens_config() in sources/transformers.py

        self.weight_processing_conversions = {
            # Gemma2 scales embeddings by sqrt(d_model)
            "embed.e": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_vocab d_model -> d_vocab d_model",
                    scale=self.cfg.d_model**0.5,
                ),
                source_key="model.embed_tokens.weight",
            ),
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.q_proj.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
                ),
                source_key="model.layers.{i}.self_attn.k_proj.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h",
                    n=getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads),
                ),
                source_key="model.layers.{i}.self_attn.v_proj.weight",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.o_proj.weight",
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
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
        """Setup hook compatibility for Gemma2 models.

        Gemma2 scales embeddings by sqrt(d_model) in its forward pass,
        but the HuggingFace embed_tokens layer doesn't include this scaling.
        We need to apply it to hook_embed to match HookedTransformer behavior.

        Args:
            bridge: The TransformerBridge instance
        """
        from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
            BaseTensorConversion,
        )

        class EmbeddingScaleConversion(BaseTensorConversion):
            """Scale embeddings by sqrt(d_model) for Gemma models.

            This only applies when NOT using processed weights, since processed
            weights have the scaling baked into the embedding matrix itself.
            """

            def __init__(self, scale: float, embed_component):
                super().__init__()
                self.scale = scale
                self.embed_component = embed_component

            def handle_conversion(self, input_value, *full_context):
                """Scale the embedding output if not using processed weights."""
                # Skip scaling if using processed weights (they're already scaled)
                if (
                    hasattr(self.embed_component, "_use_processed_weights")
                    and self.embed_component._use_processed_weights
                ):
                    return input_value
                return input_value * self.scale

            def revert(self, input_value, *full_context):
                """Unscale the embedding output (for user modifications)."""
                # Skip unscaling if using processed weights
                if (
                    hasattr(self.embed_component, "_use_processed_weights")
                    and self.embed_component._use_processed_weights
                ):
                    return input_value
                return input_value / self.scale

        # Apply scaling to embed.hook_out
        if hasattr(bridge, "embed") and hasattr(bridge.embed, "hook_out"):
            scale_factor = self.cfg.d_model**0.5
            bridge.embed.hook_out.hook_conversion = EmbeddingScaleConversion(
                scale_factor, bridge.embed
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

    def preprocess_weights(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply Gemma2-specific weight transformations before ProcessWeights.

        Gemma2 models scale embeddings by sqrt(d_model) in their forward pass.
        We bake this scaling into the embedding weights to avoid applying it every forward pass.

        Args:
            state_dict: The state dictionary with HuggingFace format keys

        Returns:
            The modified state dictionary with scaled embedding weights
        """
        from transformer_lens.weight_processing import ProcessWeights

        # Get the HF key for the embedding weight
        embed_key = ProcessWeights._get_param_key("embed.W_E", self)

        if embed_key in state_dict:
            # Scale embeddings by sqrt(d_model) to match Gemma2's forward pass behavior
            scale_factor = self.cfg.d_model**0.5
            state_dict[embed_key] = state_dict[embed_key] * scale_factor

        return state_dict
