"""<MODEL_NAME> architecture adapter.

TODO: Replace <MODEL_NAME> with the actual model name throughout this file.
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


class ModelNameArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for <MODEL_NAME> models.

    TODO: Document which parameters are optional (missing biases, etc.)

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    TODO: List parameters that may not exist. Example for models without biases:

    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP input
    - blocks.{i}.mlp.b_gate - No bias on MLP gate projection
    - blocks.{i}.mlp.b_out - No bias on MLP output
    - blocks.{i}.ln1.b - RMSNorm has no bias
    - blocks.{i}.ln2.b - RMSNorm has no bias
    - ln_final.b - RMSNorm has no bias
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the <MODEL_NAME> architecture adapter."""
        super().__init__(cfg)

        # =====================================================================
        # 1. CONFIG ATTRIBUTES
        # Set these based on the HuggingFace model's architecture.
        # =====================================================================

        # TODO: Set normalization type
        # "RMS" for RMSNorm (Llama, Qwen, Gemma, etc.)
        # "LN" for LayerNorm (GPT-2, GPT-J, etc.)
        self.cfg.normalization_type = "RMS"

        # TODO: Set positional embedding type
        # "rotary" for RoPE (Llama, Qwen, Mistral, etc.)
        # "standard" for learned positional embeddings (GPT-2)
        self.cfg.positional_embedding_type = "rotary"

        # TODO: Set these flags
        self.cfg.final_rms = True       # True if final layer norm is RMSNorm
        self.cfg.gated_mlp = True       # True if MLP has gate projection (SwiGLU)
        self.cfg.attn_only = False      # True only for attention-only models (rare)
        self.cfg.uses_rms_norm = True   # Should match normalization_type

        # TODO: Set the epsilon attribute name used by this model's normalization
        # Check the HF model's norm layer to find the correct attribute name
        self.cfg.eps_attr = "variance_epsilon"  # or "layer_norm_eps", "rms_norm_eps", etc.

        # TODO: Handle GQA if applicable
        # If the model uses Grouped Query Attention (n_key_value_heads < n_heads):
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        # =====================================================================
        # 2. WEIGHT PROCESSING CONVERSIONS
        # Defines how to reshape weights from HF format to TL format.
        # For most models with separate Q/K/V/O, use the built-in helper.
        # =====================================================================

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
            # TODO: Add any model-specific weight conversions here
        }

        # =====================================================================
        # 3. COMPONENT MAPPING
        # Maps TransformerLens canonical names to HuggingFace module paths.
        # The `name=` parameter is the HF path relative to the model root
        # (for top-level) or relative to the block (for block submodules).
        # =====================================================================

        # TODO: Replace all HF paths (name="...") with actual paths from the model.
        # Inspect the HF model's named_modules() or config to find the correct paths.
        self.component_mapping = {
            # Token embedding
            "embed": EmbeddingBridge(name="model.embed_tokens"),

            # Rotary position embeddings (remove if model uses standard pos embeddings)
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),

            # Transformer blocks
            "blocks": BlockBridge(
                name="model.layers",  # TODO: HF path to the layer list
                submodules={
                    # Pre-attention layer norm
                    "ln1": RMSNormalizationBridge(
                        name="input_layernorm",  # TODO: HF name within block
                        config=self.cfg,
                    ),
                    # Post-attention layer norm
                    "ln2": RMSNormalizationBridge(
                        name="post_attention_layernorm",  # TODO: HF name within block
                        config=self.cfg,
                    ),
                    # Self-attention
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",  # TODO: HF name within block
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),   # TODO: HF projection names
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # MLP (gated)
                    "mlp": GatedMLPBridge(
                        name="mlp",  # TODO: HF name within block
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),  # TODO: HF projection names
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),

            # Final layer norm
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),

            # Output head (unembedding)
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up model-specific references for component testing.

        TODO: Required for RoPE models. Remove if model uses standard positional embeddings.
        """
        # Get rotary embedding instance from the HF model
        rotary_emb = hf_model.model.rotary_emb  # TODO: Adjust path if different

        # Set rotary_emb on actual bridge instances
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Set on template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
