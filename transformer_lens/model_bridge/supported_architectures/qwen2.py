"""Qwen2 architecture adapter."""

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


class Qwen2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen2 models.

    Qwen2 hardcodes q/k/v biases (o_proj, MLP, and norms are bias-free); the
    include_biases conversions keep GQA K/V biases in the per-head
    (n_kv_heads, d_head) layout weight processing expects.
    """

    _testing_eager: Optional[str] = None

    _attention_bridge_cls = PositionEmbeddingsAttentionBridge

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen2 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.cfg.default_prepend_bos = False

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(include_biases=True),
        }
        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": self._build_attention_bridge(),
                    # GatedMLPBridge: hook_pre = gate pre-activation (HT GatedMLP
                    # semantics) + compat reconstruction; plain MLPBridge pointed
                    # hook_pre at the up-projection.
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
