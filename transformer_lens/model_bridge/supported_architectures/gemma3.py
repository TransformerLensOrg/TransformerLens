"""Gemma3 architecture adapter."""


from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
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

        # Gemma 3 uses rotary positional embeddings (dual RoPE)
        self.cfg.positional_embedding_type = "rotary"

        # Use SDPA for numerical consistency with HuggingFace
        # Only set if not already configured
        if self.cfg.attn_implementation is None:
            self.cfg.attn_implementation = "sdpa"

        self.conversion_rules = HookConversionSet(
            {
                # Gemma3 scales embeddings by sqrt(d_model)
                "embed.e": (
                    "model.embed_tokens.weight",
                    RearrangeHookConversion(
                        "d_vocab d_model -> d_vocab d_model",
                        scale=self.cfg.d_model**0.5,
                    ),
                ),
                "blocks.{i}.ln1.w": "model.layers.{i}.input_layernorm.weight",
                "blocks.{i}.ln2.w": "model.layers.{i}.post_attention_layernorm.weight",
                "blocks.{i}.attn.q": (
                    "model.layers.{i}.self_attn.q_proj.weight",
                    RearrangeHookConversion("(n h) m -> n m h", n=self.cfg.n_heads),
                ),
                "blocks.{i}.attn.k": (
                    "model.layers.{i}.self_attn.k_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(
                            self.cfg,
                            "n_key_value_heads",
                            self.cfg.n_heads,
                        ),
                    ),
                ),
                "blocks.{i}.attn.v": (
                    "model.layers.{i}.self_attn.v_proj.weight",
                    RearrangeHookConversion(
                        "(n h) m -> n m h",
                        n=getattr(
                            self.cfg,
                            "n_key_value_heads",
                            self.cfg.n_heads,
                        ),
                    ),
                ),
                "blocks.{i}.attn.o": (
                    "model.layers.{i}.self_attn.o_proj.weight",
                    RearrangeHookConversion("m (n h) -> n h m", n=self.cfg.n_heads),
                ),
                "blocks.{i}.mlp.in": "model.layers.{i}.mlp.up_proj.weight.T",
                "blocks.{i}.mlp.gate": "model.layers.{i}.mlp.gate_proj.weight.T",
                "blocks.{i}.mlp.out": "model.layers.{i}.mlp.down_proj.weight.T",
                "ln_final.w": "model.norm.weight",
                "unembed.u": "lm_head.weight.T",  # Not shared with embedding
            }
        )

        # Set up component mapping with actual bridge instances
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb_local": RotaryEmbeddingBridge(
                name="model.rotary_emb_local", config=self.cfg
            ),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
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
                    "mlp": MLPBridge(
                        name="mlp",
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

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references and native autograd for Gemma-3 component testing.

        Gemma-3 uses dual RoPE (global + local). We set local RoPE (used by 85% of layers)
        on all attention bridge instances for component testing.

        We also enable use_native_layernorm_autograd on all normalization bridges to ensure
        they delegate to HuggingFace's exact implementation instead of using manual computation.

        Note: Layers 5, 11, 17, 23 use global RoPE but will use local in component tests.
        This is an acceptable tradeoff given the shared-instance constraint.

        Args:
            hf_model: The HuggingFace Gemma-3 model instance
            bridge_model: The TransformerBridge model (if available, set rotary_emb on actual instances)
        """
        # Get rotary embedding instances from the model
        rotary_emb_local = hf_model.model.rotary_emb_local  # Used by 22/26 layers

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
