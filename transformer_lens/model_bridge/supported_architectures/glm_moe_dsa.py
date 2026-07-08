"""GLM-MoE-DSA architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    LinearBridge,
    MLABlockBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.glm_moe_dsa_attention import (
    GlmMoeDsaAttentionBridge,
)


class GlmMoeDsaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Z.ai GLM-5 / GLM-5.1 DSA models.

    GLM-MoE-DSA combines MLA-style latent attention, a learned sparse-attention
    indexer, dense early MLP layers, and sparse MoE later layers.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.supports_fold_ln = False
        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"
        self.cfg.default_prepend_bos = False

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": MLABlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": GlmMoeDsaAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q_a_proj": LinearBridge(name="q_a_proj"),
                            "q_a_layernorm": RMSNormalizationBridge(
                                name="q_a_layernorm", config=self.cfg
                            ),
                            "q_b_proj": LinearBridge(name="q_b_proj"),
                            "kv_a_proj_with_mqa": LinearBridge(name="kv_a_proj_with_mqa"),
                            "kv_a_layernorm": RMSNormalizationBridge(
                                name="kv_a_layernorm", config=self.cfg
                            ),
                            "kv_b_proj": LinearBridge(name="kv_b_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": GeneralizedComponent(name="gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for component testing."""
        rotary_emb = hf_model.model.rotary_emb

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
