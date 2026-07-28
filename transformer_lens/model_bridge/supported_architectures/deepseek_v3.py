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
    EmbeddingBridge,
    LinearBridge,
    MLAAttentionBridge,
    MLABlockBridge,
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

    _testing_eager = None

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.final_rms = True
        self.cfg.uses_rms_norm = True
        # HF defaults to SDPA which handles MLA correctly.
        # HF's eager attention crashes on MLA's asymmetric Q/K dimensions.

        # MLA has no per-head q/k/v to fold into; skip LN folding.
        self.supports_fold_ln = False

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": MLABlockBridge(
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
                    # shared_experts are marked optional so setup gracefully
                    # skips them when the layer is DeepseekV3MLP instead of MoE.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            # Router is a custom Module, not nn.Linear
                            "gate": GeneralizedComponent(name="gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
