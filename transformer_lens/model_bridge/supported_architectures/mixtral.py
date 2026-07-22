"""Mixtral architecture adapter."""

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

        self._set_rms_rotary_defaults(final_rms=False)

        n_kv_heads = (
            self.cfg.n_key_value_heads
            if hasattr(self.cfg, "n_key_value_heads") and self.cfg.n_key_value_heads is not None
            else self.cfg.n_heads
        )

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
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
                    # native forward pass. 5.13 renamed the decoder-layer attr
                    # block_sparse_moe -> mlp.
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
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model)
