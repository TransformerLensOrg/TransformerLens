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
                            # Two-stage LoRA Q compression, built only when
                            # q_lora_rank is set; a config that leaves it null
                            # gets a single q_proj instead, and MLAAttentionBridge
                            # already forwards down whichever path exists.
                            "q_a_proj": LinearBridge(name="q_a_proj", optional=True),
                            "q_a_layernorm": RMSNormalizationBridge(
                                name="q_a_layernorm", config=self.cfg, optional=True
                            ),
                            "q_b_proj": LinearBridge(name="q_b_proj", optional=True),
                            "q_proj": LinearBridge(name="q_proj", optional=True),
                            "kv_a_proj_with_mqa": LinearBridge(name="kv_a_proj_with_mqa"),
                            "kv_a_layernorm": RMSNormalizationBridge(
                                name="kv_a_layernorm", config=self.cfg
                            ),
                            "kv_b_proj": LinearBridge(name="kv_b_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    # Dense-prefix layers (idx < first_k_dense_replace) bind as
                    # gated MLPs with neuron-basis hooks; sparse layers keep the
                    # MoE mapping with its optional router/shared experts (#1645).
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        sparse_required=("gate",),
                        submodules={
                            # Router is a custom Module, not nn.Linear
                            "gate": GeneralizedComponent(name="gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                            # Dense-layer projections (present only on the
                            # dense layers of this interleaved stack); their
                            # presence is what makes MoEBridge bind gated-MLP
                            # neuron hooks there (#1645).
                            "dense_gate": LinearBridge(name="gate_proj", optional=True),
                            "dense_in": LinearBridge(name="up_proj", optional=True),
                            "dense_out": LinearBridge(name="down_proj", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
