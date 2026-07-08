"""HunYuanDenseV1 architecture adapter."""

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


class HunYuanDenseV1ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for HunYuanDenseV1 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the HunYuanDenseV1 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.cfg.attn_implementation = "eager"

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(
                        name="input_layernorm",
                        config=self.cfg,
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="post_attention_layernorm",
                        config=self.cfg,
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "q_norm": RMSNormalizationBridge(
                                name="query_layernorm", config=self.cfg
                            ),
                            "k_norm": RMSNormalizationBridge(name="key_layernorm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up model-specific references for component testing."""
        # Get rotary embedding instance from the HF model
        rotary_emb = hf_model.model.rotary_emb

        # Set attention implementation on HF model to eager (vs sdpa default)
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary_emb on actual bridge instances
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Set on template for get_generalized_component() calls
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
