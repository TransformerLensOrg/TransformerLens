"""OLMo architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class OlmoArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OLMo (v1) models.

    OLMo v1 uses a pre-norm architecture with a custom non-learnable LayerNorm
    (fixed weight=1, bias=0), rotary position embeddings (RoPE), and gated MLP
    (SwiGLU). Key differences from later OLMo variants:

    - Pre-norm: LayerNorm is applied BEFORE attention and BEFORE MLP.
    - Non-learnable LayerNorm: Weight and bias are not trainable parameters.
      Delegating to HF's native forward via NormalizationBridge handles this correctly.
    - No Q/K normalization in attention.
    - Optional QKV clipping (applied out-of-place by the reconstructed
      attention forward when config.clip_qkv is set).

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP up_proj
    - blocks.{i}.mlp.b_gate - No bias on MLP gate_proj
    - blocks.{i}.mlp.b_out - No bias on MLP down_proj
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the OLMo architecture adapter."""
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = False
        # Force eager attention for numerical consistency with benchmark reference
        self.cfg.attn_implementation = "eager"

        n_kv_heads = (
            self.cfg.n_key_value_heads
            if self.cfg.n_key_value_heads is not None
            else self.cfg.n_heads
        )

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
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": NormalizationBridge(
                name="model.norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
