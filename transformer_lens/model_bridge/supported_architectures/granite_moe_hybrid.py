"""Granite MoE Hybrid architecture adapter.

Hybrid Mamba2 + Attention with Sparse MoE. Most layers are Mamba SSM blocks;
a few are standard attention (determined by config.layer_types). Every layer
has a shared MLP and optional sparse MoE.

Both attention and Mamba are mapped as optional — each present only on its
respective layer type. Mamba hooks expose in_proj, conv1d, and inner_norm.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SSM2MixerBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.depthwise_conv1d import (
    DepthwiseConv1DBridge,
)
from transformer_lens.model_bridge.supported_architectures.granite import (
    GraniteArchitectureAdapter,
)


class GraniteMoeHybridArchitectureAdapter(GraniteArchitectureAdapter):
    """Hybrid Mamba2 + Attention with Sparse MoE.

    Attention is optional (absent on Mamba layers). shared_mlp and MoE are
    universal. Inherits Granite config and attention bridge construction.
    """

    def __init__(self, cfg: Any) -> None:
        ArchitectureAdapter.__init__(self, cfg)
        self._setup_common_config(cfg)

        pos_emb_type = getattr(cfg, "position_embedding_type", "rope")
        if pos_emb_type != "rope":
            self.cfg.positional_embedding_type = "none"

        self.supports_fold_ln = False
        self.weight_processing_conversions = {}
        self.component_mapping = self._build_component_mapping()

    def _build_mamba_bridge(self) -> SSM2MixerBridge:
        """Mamba-2 mixer bridge with in_proj, conv1d, inner_norm hooks."""
        return SSM2MixerBridge(
            name="mamba",
            config=self.cfg,
            optional=True,
            submodules={
                "in_proj": LinearBridge(name="in_proj"),
                "conv1d": DepthwiseConv1DBridge(name="conv1d"),
                "inner_norm": LinearBridge(name="norm"),
            },
        )

    def _build_component_mapping(self) -> dict:
        block_submodules: dict = {
            "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
            "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
            "attn": self._build_attention_bridge(optional=True),
            "mamba": self._build_mamba_bridge(),
            "shared_mlp": MLPBridge(
                name="shared_mlp",
                config=self.cfg,
                submodules={
                    "in": LinearBridge(name="input_linear"),
                    "out": LinearBridge(name="output_linear"),
                },
            ),
        }

        num_experts = getattr(self.cfg, "num_experts", None) or getattr(
            self.cfg, "num_local_experts", 0
        )
        if num_experts and num_experts > 0:
            block_submodules["moe"] = MoEBridge(
                name="block_sparse_moe",
                config=self.cfg,
            )

        mapping: dict = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(name="model.layers", submodules=block_submodules),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

        if self.cfg.positional_embedding_type == "rotary":
            mapping["rotary_emb"] = RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg)

        return mapping
