"""Lfm2 architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    Lfm2ShortConvBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Lfm2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Lfm2 models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the Lfm2 architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True

        self.cfg.attn_implementation = "eager"

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

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
                        name="operator_norm",
                        config=self.cfg,
                    ),
                    "ln2": RMSNormalizationBridge(
                        name="ffn_norm",
                        config=self.cfg,
                    ),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        optional=True,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="out_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_layernorm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_layernorm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "conv": Lfm2ShortConvBridge(
                        name="conv",
                        config=self.cfg,
                        optional=True,
                        submodules={
                            "in": LinearBridge(name="in_proj"),
                            "conv": DepthwiseConv1DBridge(name="conv"),
                            "out": LinearBridge(name="out_proj"),
                        },
                    ),
                    "mlp": GatedMLPBridge(
                        name="feed_forward",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="w1"),
                            "in": LinearBridge(name="w3"),
                            "out": LinearBridge(name="w2"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.embedding_norm", config=self.cfg),
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
        # Find the first attention layer (LFM2 layer 0 is conv, not attn)
        layer_types = getattr(self.cfg, "layer_types", None)
        if layer_types is not None and "full_attention" in layer_types:
            first_attn_idx = layer_types.index("full_attention")
            attn_bridge = self.get_generalized_component(f"blocks.{first_attn_idx}.attn")
            attn_bridge.set_rotary_emb(rotary_emb)
