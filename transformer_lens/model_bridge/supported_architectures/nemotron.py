"""Nemotron architecture adapter.

NVIDIA's dense Nemotron-3/4 and Minitron families (``NemotronForCausalLM``):
llama-shaped GQA decoder with three quirks — LayerNorm1P normalization
(zero-centered gamma, applied as ``weight + 1``), a non-gated squared-ReLU
MLP (``up_proj``/``down_proj`` only), and partial rotary embeddings
(rotary factor 0.5).
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PositionEmbeddingsAttentionBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class NemotronArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NemotronForCausalLM models."""

    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Nemotron architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # LayerNorm1P applies gamma as (weight + 1); standard LN folding would
        # fold the stored weight without the offset. Keep raw weights.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    # LayerNorm1P applies gamma as (weight + 1); delegate to the
                    # native module rather than reimplementing LN without the offset.
                    "ln1": NormalizationBridge(
                        name="input_layernorm",
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
                    "ln2": NormalizationBridge(
                        name="post_attention_layernorm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        submodules={
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(
                name="model.norm", config=self.cfg, use_native_layernorm_autograd=True
            ),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
