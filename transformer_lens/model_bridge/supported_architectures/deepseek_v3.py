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

    DeepSeek V3 uses:
    - RMSNorm for all normalizations
    - Multi-Head Latent Attention (MLA) with compressed Q and KV projections
    - Rotary position embeddings (RoPE) on partial head dimensions only
    - Mixture of Experts MLP on most layers, dense MLP on first few layers
    - No biases on projections
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.final_rms = True
        self.cfg.uses_rms_norm = True
        # Not used by MLAAttentionBridge (which reimplements forward), but needed
        # when the HF model is used as a reference in setup_component_testing /
        # benchmarks — SDPA doesn't support output_attentions=True.
        self.cfg.attn_implementation = "eager"

        # MLA doesn't use standard Q/K/V/O weight rearrangements
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
                    # MoEBridge wraps both MoE and dense MLP layers. On dense layers
                    # (layer_idx < first_k_dense_replace), MoE-specific submodules
                    # (gate, shared_experts) are gracefully skipped by setup_submodules
                    # since DeepseekV3MLP lacks those attributes. On MoE layers, all
                    # submodules are wired and hook_router_scores fires.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            # DeepseekV3TopkRouter is a custom Module (not nn.Linear),
                            # so we use GeneralizedComponent instead of LinearBridge.
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
        """Set up rotary embedding references for DeepSeek V3 component testing.

        Args:
            hf_model: The HuggingFace DeepSeek V3 model instance
            bridge_model: The TransformerBridge model (if available)
        """
        rotary_emb = hf_model.model.rotary_emb

        # Set on live block instances (used by forward passes)
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        # Set on template (used by get_generalized_component() callers — benchmarks,
        # component tests)
        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
