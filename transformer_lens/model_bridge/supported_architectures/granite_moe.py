"""Granite MoE architecture adapter."""

from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    ScaledResidualBlockBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.granite import (
    GraniteArchitectureAdapter,
)


class GraniteMoeArchitectureAdapter(GraniteArchitectureAdapter):
    """Architecture adapter for IBM Granite MoE models.

    Identical to dense Granite but replaces the gated MLP with a Sparse Mixture
    of Experts block (block_sparse_moe) using batched expert parameters and
    top-k routing.
    """

    def _build_component_mapping(self) -> dict:
        """Build component mapping with MoE instead of dense MLP."""
        return {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            # HF multiplies each sublayer output by residual_multiplier before the
            # residual add; hook_attn_out / hook_mlp_out expose the scaled contribution.
            "blocks": ScaledResidualBlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": self._build_attention_bridge(),
                    "mlp": MoEBridge(
                        name="block_sparse_moe",
                        config=self.cfg,
                    ),
                },
                residual_contribution_scale=getattr(self.cfg, "residual_multiplier", 1.0),
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
