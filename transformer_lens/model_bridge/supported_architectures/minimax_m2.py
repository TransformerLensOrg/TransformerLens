"""MiniMax-M2 architecture adapter."""

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


class MiniMaxM2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for MiniMaxM2ForCausalLM models.

    Pre-norm RMSNorm decoder with RoPE (partial rotary factor 0.5), GQA, and a
    sparse MoE MLP. MiniMax-M2 quirks vs the structurally-closest Qwen3-MoE:

    - Q/K normalization is applied over the FULL projection width (all heads
      concatenated) before the head reshape, not per-head.
    - The router scores experts with sigmoid + a trained e_score_correction_bias
      buffer (DeepSeek-V3 style) instead of softmax, and is a custom module
      holding a raw weight parameter — not an nn.Linear — so the MoE block is
      fully delegated with no gate submodule mapping.
    - Explicit head_dim (128) larger than hidden_size / num_heads.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the MiniMax-M2 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Verified against MiniMaxAI/MiniMax-M2: tokenizer does not prepend BOS.
        self.cfg.default_prepend_bos = False

        # QKVO rearrangements; MoE expert and router weights pass through unchanged
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

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
                            # Full-width (all-heads) RMSNorm, applied pre-reshape.
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mlp": MoEBridge(name="mlp", config=self.cfg),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model)
