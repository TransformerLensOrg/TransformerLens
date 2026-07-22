"""Llama architecture adapter."""

from typing import Any, Optional

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


class LlamaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Llama models.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    LLaMA models do NOT have biases on attention and MLP projections:

    - blocks.{i}.attn.b_Q - No bias on query projection
    - blocks.{i}.attn.b_K - No bias on key projection
    - blocks.{i}.attn.b_V - No bias on value projection
    - blocks.{i}.attn.b_O - No bias on output projection
    - blocks.{i}.mlp.b_in - No bias on MLP input (up_proj)
    - blocks.{i}.mlp.b_gate - No bias on MLP gate projection
    - blocks.{i}.mlp.b_out - No bias on MLP output (down_proj)
    - blocks.{i}.ln1.b - RMSNorm has no bias
    - blocks.{i}.ln2.b - RMSNorm has no bias
    - ln_final.b - RMSNorm has no bias

    Weight processing must handle these missing biases gracefully using
    ProcessWeights._safe_get_tensor() or by checking for None values.
    """

    _testing_eager: Optional[str] = None

    _attention_bridge_cls = PositionEmbeddingsAttentionBridge

    def __init__(self, cfg: Any) -> None:
        """Initialize the Llama architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": self._build_attention_bridge(),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def _build_attention_bridge(self):
        """Attention bridge seam; subclasses swap the class or the construction."""
        return self._attention_bridge_cls(
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
        )
