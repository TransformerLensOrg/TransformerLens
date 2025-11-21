"""GPT-OSS architecture adapter."""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class GPTOSSArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GPT-OSS model."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the GPT-OSS architecture adapter."""
        super().__init__(cfg)

        self.cfg.gated_mlp = True

        self.cfg.uses_rms_norm = True
        # GPT-OSS uses 'variance_epsilon' instead of 'eps' for RMSNorm
        self.cfg.eps_attr = "variance_epsilon"
        # GPT-OSS uses rotary position embeddings, not learned embeddings
        self.cfg.positional_embedding_type = "rotary"
        # GPT-OSS attention returns (output, attn_weights), not a 3-tuple
        # Note: attention_output_format is not a standard config attribute, handled in architecture code

        # Conversion rules for weight processing/folding
        # GPT-OSS uses MoE with batched experts, so we need special handling
        self.weight_processing_conversions = {
            "blocks.{i}.attn.q": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.q_proj.weight",
            ),
            "blocks.{i}.attn.k": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.k_proj.weight",
            ),
            "blocks.{i}.attn.v": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.v_proj.weight",
            ),
            "blocks.{i}.attn.o": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                source_key="model.layers.{i}.self_attn.o_proj.weight",
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(
                        name="input_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,  # Use HF's RMSNorm for correct dtype handling
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        requires_position_embeddings=True,  # GPT-OSS requires position_embeddings (rotary)
                        requires_attention_mask=True,  # GPT-OSS requires attention_mask
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="post_attention_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,  # Use HF's RMSNorm for correct dtype handling
                    ),
                    # GPT-OSS uses batched MoE experts with router scores
                    # MoEBridge handles the (hidden_states, router_scores) tuple returns
                    "mlp": MoEBridge(name="mlp", config=self.cfg),
                },
            ),
            "ln_final": RMSNormalizationBridge(
                name="model.norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,  # Use HF's RMSNorm for correct dtype handling
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_hook_compatibility(self, bridge_model: Any) -> None:
        """Setup hook compatibility transformations for GPT-OSS models.

        This configures rotary embedding references for attention layers, which is
        needed for models using RoPE (Rotary Position Embeddings).

        This is called during Bridge.__init__ and should always be run.

        Args:
            bridge_model: The TransformerBridge instance
        """
        # Get the rotary_emb component from the actual bridge model
        if bridge_model is None or not hasattr(bridge_model, "rotary_emb"):
            return

        # Get the actual HF rotary_emb from the bridge's rotary_emb component
        rotary_emb = bridge_model.rotary_emb.original_component

        # Set rotary_emb on all attention bridge instances
        if hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

    def setup_no_processing_hooks(self, bridge_model: Any) -> None:
        """Backward compatibility alias for setup_hook_compatibility."""
        self.setup_hook_compatibility(bridge_model)
