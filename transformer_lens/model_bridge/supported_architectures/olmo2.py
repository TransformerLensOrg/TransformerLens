"""OLMo 2 architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Olmo2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OLMo 2 models.

    OLMo 2 uses a post-norm architecture with RMSNorm, Q/K normalization in attention,
    rotary position embeddings (RoPE), and gated MLP (SwiGLU). Key differences from
    pre-norm models like Llama:

    - Post-norm: RMSNorm is applied AFTER attention and AFTER MLP, not before.
      ln1_post maps to post_attention_layernorm, ln2_post to post_feedforward_layernorm;
      there is no ln1/ln2 (no pre-sublayer norm).
    - Q/K normalization: Per-head RMSNorm applied to queries and keys after projection.
    - No biases on any projections.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP up_proj
    - blocks.{i}.mlp.b_gate - No bias on MLP gate_proj
    - blocks.{i}.mlp.b_out - No bias on MLP down_proj
    - blocks.{i}.ln1_post.b - RMSNorm has no bias
    - blocks.{i}.ln2_post.b - RMSNorm has no bias
    - ln_final.b - RMSNorm has no bias
    """

    # Attention bridge seam; EXAONE-4 swaps in its NoPE-gating variant.
    _attention_bridge_cls = PositionEmbeddingsAttentionBridge

    def __init__(self, cfg: Any) -> None:
        """Initialize the OLMo 2 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # OLMo-2 uses post-norm (RMSNorm AFTER attention/MLP), so layer norm
        # folding into QKV/MLP weights is incorrect — the norms apply to the
        # output, not the input. Same pattern as BERT and Phi-3.
        self.supports_fold_ln = False
        # Force eager attention for numerical consistency with benchmark reference.
        # PositionEmbeddingsAttentionBridge delegates to native HF attention, so
        # both bridge and reference must use the same implementation.
        self.cfg.attn_implementation = "eager"

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    # Post-norm only: BlockBridge derives hook_attn_out / hook_mlp_out
                    # (norm outputs) and hook_resid_mid (mlp.hook_in) from these keys.
                    "ln1_post": RMSNormalizationBridge(
                        name="post_attention_layernorm", config=self.cfg
                    ),
                    "ln2_post": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
                    ),
                    "attn": self._attention_bridge_cls(
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
                    "mlp": self._build_mlp_bridge(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def _build_mlp_bridge(self):
        """MLP bridge seam; FlexOlmo swaps in the MoE variant."""
        return self._gated_mlp()
