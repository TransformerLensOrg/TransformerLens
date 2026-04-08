"""DeepSeek V3 architecture adapter.

Supports DeepSeek V3 and DeepSeek-R1 models (both use DeepseekV3ForCausalLM).
Key features:
- Multi-Head Latent Attention (MLA): Q and KV compressed via LoRA-style projections
- Mixture of Experts (MoE) with shared experts on most layers
- Dense MLP on first `first_k_dense_replace` layers
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MLAAttentionBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class DeepSeekV3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for DeepSeek V3 / R1 models.

    Uses RMSNorm, MLA with compressed Q/KV projections, partial RoPE,
    MoE on most layers (dense MLP on first few), and no biases.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.final_rms = True
        self.cfg.uses_rms_norm = True
        # HF defaults to SDPA which handles MLA correctly.
        # HF's eager attention crashes on MLA's asymmetric Q/K dimensions.

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": MLAAttentionBridge(
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
                    # On dense layers (idx < first_k_dense_replace), gate and
                    # shared_experts are gracefully skipped since DeepseekV3MLP
                    # lacks those attributes.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            # Router is a custom Module, not nn.Linear
                            "gate": GeneralizedComponent(name="gate"),
                            "shared_experts": GatedMLPBridge(
                                name="shared_experts",
                                config=self.cfg,
                                submodules={
                                    "gate": LinearBridge(name="gate_proj"),
                                    "in": LinearBridge(name="up_proj"),
                                    "out": LinearBridge(name="down_proj"),
                                },
                            ),
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

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Also set on template for get_generalized_component() callers
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
