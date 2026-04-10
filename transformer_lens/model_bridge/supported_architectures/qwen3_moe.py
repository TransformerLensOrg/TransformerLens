"""Qwen3MoE (Mixture of Experts) architecture adapter."""

from typing import Any

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


class Qwen3MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen3MoE (Mixture of Experts) models.

    Qwen3MoE is a sparse Mixture-of-Experts decoder-only Transformer that closely
    mirrors OLMoE in structure.  Key architectural features:

    - Pre-norm: RMSNorm applied BEFORE attention (input_layernorm) and BEFORE MLP
      (post_attention_layernorm).
    - Q/K normalization: RMSNorm applied to queries and keys after projection and
      before rotary embedding application.
    - Sparse MoE: 128 experts with top-8 routing (in the public 30B-A3B checkpoints).
    - Batched expert parameters: gate_up_proj [num_experts, 2*moe_intermediate_size,
      hidden_size] and down_proj [num_experts, hidden_size, moe_intermediate_size] are
      stored as single 3D tensors rather than a ModuleList.
    - final_rms=True (Qwen3-style; differs from OLMoE which uses False).
    - No biases on any projections (attention_bias=False in all public checkpoints).
    - GQA: num_key_value_heads < num_attention_heads in all public models.

    Limitation — all-MoE configuration only:
        All public Qwen3MoE models have decoder_sparse_step=1 and mlp_only_layers=[]
        (every decoder layer is a sparse MoE block).  This adapter supports only that
        all-MoE configuration.  Models with a non-empty mlp_only_layers list are NOT
        supported because MoEBridge cannot handle the dense Qwen3MoeMLP fallback layers.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q  - No bias on query projection  (attention_bias=False)
    - blocks.{i}.attn.b_K  - No bias on key projection    (attention_bias=False)
    - blocks.{i}.attn.b_V  - No bias on value projection  (attention_bias=False)
    - blocks.{i}.attn.b_O  - No bias on output projection (attention_bias=False)
    - blocks.{i}.ln1.b    - RMSNorm has no additive bias
    - blocks.{i}.ln2.b    - RMSNorm has no additive bias
    - ln_final.b          - RMSNorm has no additive bias
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen3MoE architecture adapter."""
        super().__init__(cfg)

        # ------------------------------------------------------------------ #
        # Config attributes
        # ------------------------------------------------------------------ #
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True  # Qwen3-style; OLMoE uses False
        self.cfg.gated_mlp = True  # SwiGLU-style gate in every MoE expert
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        # Force eager attention for output_attentions hook support
        self.cfg.attn_implementation = "eager"
        self.cfg.default_prepend_bos = False  # Qwen3 family convention

        # GQA: propagate n_key_value_heads when provided by the loaded config.
        # map_default_transformer_lens_config() sets this from num_key_value_heads
        # in the HF checkpoint config; we do not hard-code a fallback value.
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        # ------------------------------------------------------------------ #
        # Weight processing conversions
        # ------------------------------------------------------------------ #
        # Standard QKVO rearrangements; _qkvo_weight_conversions() resolves
        # n_kv_heads from self.cfg.n_key_value_heads automatically.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }
        # MoE expert weights (gate_up_proj, down_proj) and gate router weights
        # (gate.weight) pass through unchanged — HF's native forward handles them.

        # ------------------------------------------------------------------ #
        # Component mapping — pre-norm architecture
        # ------------------------------------------------------------------ #
        # ln1 = input_layernorm  (applied BEFORE attention)
        # ln2 = post_attention_layernorm (applied BEFORE MLP)
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
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
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # Qwen3MoeSparseMoeBlock uses batched expert parameters
                    # (gate_up_proj / down_proj as 3D tensors) rather than a
                    # ModuleList.  MoEBridge wraps the entire block and delegates
                    # to HF's native forward.  The gate (Qwen3MoeTopKRouter) is
                    # mapped as a submodule via LinearBridge for hook access —
                    # same pattern as OLMoE.
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
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Qwen3MoE component testing.

        Qwen3MoE uses RoPE (Rotary Position Embeddings) stored at model.rotary_emb.
        We retrieve the rotary_emb instance from the HF model and attach it to all
        attention bridge instances so that component-level tests can run the full
        attention forward pass correctly.

        Args:
            hf_model: The HuggingFace Qwen3MoeForCausalLM model instance.
            bridge_model: The TransformerBridge model (if available).
        """
        rotary_emb = hf_model.model.rotary_emb

        # Force eager attention on the HF model to match bridge implementation
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Attach rotary_emb to each block's attention bridge
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on the template bridge for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
