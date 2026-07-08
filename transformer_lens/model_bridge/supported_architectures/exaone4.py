"""EXAONE 4.0 architecture adapter.

LG AI Research's EXAONE-4.0 (``Exaone4ForCausalLM``, native transformers —
distinct from the remote-code EXAONE-3.x family in exaone.py): llama-shaped
GQA decoder with per-head Q/K RMSNorms, hybrid sliding/global attention
(``layer_types``), and POST-norms applied inside the residual branch
(``post_attention_layernorm`` after attention, ``post_feedforward_layernorm``
after the MLP) with no pre-norms.
"""

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


class Exaone4ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Exaone4ForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the EXAONE 4.0 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Same tokenizer family as EXAONE-3.x: no BOS prepending.
        self.cfg.default_prepend_bos = False

        layer_types = getattr(cfg, "layer_types", None)
        if layer_types:
            setattr(self.cfg, "layer_types", list(layer_types))

        # Norms are applied after each sublayer inside the residual branch;
        # pre-LN folding does not apply.
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
                    # Post-norms: ln1 follows attention, ln2 follows the MLP.
                    "ln1": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "mlp": self._gated_mlp(),
                    "ln2": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model, eager="config")
