"""Mixtral architecture adapter."""

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


class MixtralArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Mixtral models.

    Mixtral uses a pre-norm architecture with RMSNorm, rotary position embeddings
    (RoPE), and a Sparse Mixture of Experts MLP. Key features:

    - Pre-norm: RMSNorm applied BEFORE attention and BEFORE MLP.
    - Rotary embeddings: stored at model.rotary_emb and passed per-forward-call.
    - Sparse MoE: batched expert parameters (gate_up_proj, down_proj as 3D tensors).
    - MixtralAttention.forward() requires position_embeddings and attention_mask args.
    - Optional GQA (n_key_value_heads may differ from n_heads).
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Mixtral architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True

        n_kv_heads = (
            self.cfg.n_key_value_heads
            if hasattr(self.cfg, "n_key_value_heads") and self.cfg.n_key_value_heads is not None
            else self.cfg.n_heads
        )

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h", n=self.cfg.n_heads
                ),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h", n=n_kv_heads
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(n h) m -> n m h", n=n_kv_heads
                ),
            ),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) -> h d_head", h=self.cfg.n_heads
                ),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) -> h d_head", h=n_kv_heads
                ),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) -> h d_head", h=n_kv_heads
                ),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "m (n h) -> n h m", n=self.cfg.n_heads
                ),
            ),
        }

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # MixtralAttention.forward() requires position_embeddings and
                    # attention_mask as positional arguments (not optional kwargs).
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
                    # Mixtral uses batched expert parameters (gate_up_proj, down_proj
                    # as 3D tensors) rather than a ModuleList of individual experts.
                    # MoEBridge wraps the entire MLP module and delegates to HF's
                    # native forward pass. The gate (router) is mapped as a submodule
                    # for hook access.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Mixtral component testing.

        Mixtral uses RoPE (Rotary Position Embeddings). We set the rotary_emb
        reference on all attention bridge instances for component testing.

        Args:
            hf_model: The HuggingFace Mixtral model instance
            bridge_model: The TransformerBridge model (if available)
        """
        rotary_emb = hf_model.model.rotary_emb

        # Force HF model to use "eager" attention to match bridge implementation
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary_emb on actual bridge instances in bridge_model if available
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
