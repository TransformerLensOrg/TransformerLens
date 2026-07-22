"""Starcoder2 architecture adapter.

BigCode's StarCoder2 (``Starcoder2ForCausalLM``): pre-norm decoder with
plain LayerNorm (not RMS), separate biased q/k/v/o projections, GQA, RoPE,
and a non-gated ``c_fc``/``c_proj`` MLP.
"""

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


class Starcoder2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Starcoder2ForCausalLM models."""

    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Starcoder2 architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        # StarCoder2 biases every q/k/v/o projection; the bias reshapes must use
        # the kv-head count or compat mode mis-shapes (silent) or crashes (GQA).
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(include_biases=True),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": NormalizationBridge(name="input_layernorm", config=self.cfg),
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
                    "ln2": NormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "mlp": self._ungated_mlp(up="c_fc", down="c_proj"),
                },
            ),
            "ln_final": NormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
