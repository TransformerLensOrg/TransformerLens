"""XGLM architecture adapter.

Supports XGLMForCausalLM (facebook/xglm-*).
Assumes add_cross_attention=False (all published XGLM checkpoints).
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    SymbolicBridge,
    UnembeddingBridge,
)


class XGLMArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for XGLM models.

    XGLM uses pre-norm LayerNorm, sinusoidal positional embeddings (no
    learnable weights), standard MHA with separate q/k/v/out_proj, and a
    2-layer MLP (fc1/fc2) that lives directly on the decoder block rather
    than inside an mlp sub-module.

    All attention projections and fc1/fc2 carry biases. lm_head has no bias.
    Embeddings are scaled by sqrt(d_model) at runtime in XGLMScaledWordEmbedding.

    Optional Parameters (may not exist in state_dict):
    --------------------------------------------------
    None — all published XGLM checkpoints include all parameters listed above.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the XGLM architecture adapter."""
        super().__init__(cfg)

        # LayerNorm throughout (not RMSNorm)
        self.cfg.normalization_type = "LN"
        # Sinusoidal positional embeddings — added to token embeddings before blocks,
        # no learnable weights, no RoPE
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        # Standard 2-layer MLP (fc1 -> gelu -> fc2), no gate projection
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False

        # Sinusoidal positional embeddings have no weights in the state_dict, so
        # center_writing_weights cannot center pos_embed.  Disable it for XGLM.
        self.supports_center_writing_weights = False

        # Standard MHA: n_heads == n_kv_heads for all XGLM sizes
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            # No "pos_embed": sinusoidal embeddings are a non-persistent buffer with
            # no learnable weights — embed_positions does not appear in state_dict.
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(
                        name="self_attn_layer_norm",  # pre-attn norm on XGLMDecoderLayer
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        requires_attention_mask=True,
                        attention_mask_4d=True,  # (batch, 1, tgt_len, src_len)
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),  # out_proj, not o_proj
                        },
                    ),
                    "ln2": NormalizationBridge(
                        name="final_layer_norm",  # pre-MLP norm on XGLMDecoderLayer
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    # fc1/fc2 live directly on XGLMDecoderLayer — no "mlp" container.
                    # SymbolicBridge preserves TL structure without a real HF submodule.
                    "mlp": SymbolicBridge(
                        submodules={
                            "in": LinearBridge(name="fc1"),
                            "out": LinearBridge(name="fc2"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(
                name="model.layer_norm",  # note: layer_norm, not norm
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_hook_compatibility(self, bridge: Any) -> None:
        """Scale hook_embed by sqrt(d_model) to match XGLMScaledWordEmbedding.forward().

        XGLMScaledWordEmbedding multiplies the embedding lookup by embed_scale =
        sqrt(d_model) at runtime.  Without this override, hook_embed would capture
        the raw (unscaled) table output, diverging from actual model activations.
        """
        from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
            BaseTensorConversion,
        )

        class EmbeddingScaleConversion(BaseTensorConversion):
            """Scale embeddings by sqrt(d_model) for XGLM models."""

            def __init__(self, scale: float) -> None:
                super().__init__()
                self.scale = scale

            def handle_conversion(self, input_value: Any, *full_context: Any) -> Any:
                return input_value * self.scale

            def revert(self, input_value: Any, *full_context: Any) -> Any:
                return input_value / self.scale

        if hasattr(bridge, "embed") and hasattr(bridge.embed, "hook_out"):
            bridge.embed.hook_out.hook_conversion = EmbeddingScaleConversion(
                self.cfg.d_model**0.5
            )
