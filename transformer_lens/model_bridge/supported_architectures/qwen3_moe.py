"""Qwen3MoE (Mixture of Experts) architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Qwen3MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen3MoE (Mixture of Experts) models.

    Qwen3MoE is a sparse MoE decoder-only Transformer, structurally close to OLMoE.
    Key features:

    - Pre-norm: RMSNorm applied BEFORE attention and BEFORE MLP.
    - Q/K normalization: RMSNorm applied to queries and keys after projection.
    - Sparse MoE: 128 experts with top-8 routing (public 30B-A3B checkpoints).
    - Batched expert parameters: gate_up_proj and down_proj as single 3D tensors,
      not a ModuleList.
    - final_rms=True (Qwen3-style; OLMoE uses False).
    - No biases on any projections.
    - GQA: n_key_value_heads < n_heads in all public checkpoints.

    Only the all-MoE configuration is supported (decoder_sparse_step=1,
    mlp_only_layers=[]). Models with dense fallback layers cannot be wrapped
    because MoEBridge does not handle the dense Qwen3MoeMLP path.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.ln1.b - RMSNorm has no bias
    - blocks.{i}.ln2.b - RMSNorm has no bias
    - ln_final.b - RMSNorm has no bias
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen3MoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Force eager attention for output_attentions hook support
        self.cfg.attn_implementation = "eager"
        self.cfg.default_prepend_bos = False  # Qwen3 family convention

        # QKVO rearrangements; MoE expert and gate weights pass through unchanged
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        # Component mapping — PRE-NORM architecture:
        # ln1 = input_layernorm (applied BEFORE attention)
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
                    # Qwen3MoeSparseMoeBlock stores experts as batched 3D tensors
                    # rather than a ModuleList. MoEBridge wraps the entire block and
                    # delegates to HF's native forward — same pattern as OLMoE.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": MoERouterBridge(name="gate"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model)
