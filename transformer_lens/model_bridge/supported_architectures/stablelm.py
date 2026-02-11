"""StableLM architecture adapter."""

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
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class StableLmArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for StableLM models.

    StableLM uses a Llama-like architecture with separate Q/K/V projections and
    gated MLP, but differs in using standard LayerNorm (not RMSNorm) and partial
    rotary embeddings (25% of head dimensions by default).

    Supports optional features:
    - Grouped Query Attention (num_key_value_heads != num_attention_heads)
    - QKV bias (use_qkv_bias=True on some models like stable-code-3b)
    - Parallel residual connections (use_parallel_residual=True)
    - Per-head QK LayerNorm (qk_layernorm=True)

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_K - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_V - Only present when use_qkv_bias=True
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP up_proj
    - blocks.{i}.mlp.b_gate - No bias on MLP gate_proj
    - blocks.{i}.mlp.b_out - No bias on MLP down_proj
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the StableLM architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False
        # Force eager attention for numerical consistency with benchmark reference
        # PositionEmbeddingsAttentionBridge delegates to native HF attention, so
        # both bridge and reference must use the same implementation
        self.cfg.attn_implementation = "eager"

        self.default_config = {
            "d_model": cfg.d_model,
            "d_head": cfg.d_model // cfg.n_heads,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_vocab": cfg.d_vocab,
        }

        # GQA support
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.default_config["n_key_value_heads"] = cfg.n_key_value_heads
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        n_kv_heads = getattr(self.cfg, "n_key_value_heads", self.cfg.n_heads)

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=self.cfg.n_heads),
            ),
            # Bias conversions for models with use_qkv_bias=True
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=self.cfg.n_heads),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) -> n h", n=n_kv_heads),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(
                        name="input_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="post_attention_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
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
            "ln_final": NormalizationBridge(
                name="model.norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for StableLM component testing.

        StableLM uses RoPE (Rotary Position Embeddings) with partial rotation.
        We set the rotary_emb reference on all attention bridge instances and
        force eager attention for numerical consistency.

        Args:
            hf_model: The HuggingFace StableLM model instance
            bridge_model: The TransformerBridge model (if available)
        """
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

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
