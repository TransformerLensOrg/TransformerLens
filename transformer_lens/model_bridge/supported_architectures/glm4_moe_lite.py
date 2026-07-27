"""GLM-4 MoE Lite architecture adapter.

Supports the GLM-4.7-Flash family (`Glm4MoeLiteForCausalLM`): DeepSeek-style
Multi-head Latent Attention (LoRA-compressed Q and KV, nope/rope split heads,
interleaved partial RoPE) combined with GLM's sparse MoE — sigmoid router with
e_score_correction_bias, batched routed experts, one shared expert — and a
per-layer dense/sparse MLP mix declared in ``config.mlp_layer_types``.
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
from transformer_lens.model_bridge.supported_architectures.glm4_moe import (
    Glm4MoeRouterBridge,
)


class Glm4MoeLiteArchitectureAdapter(ArchitectureAdapter):
    """GLM-4.7-Flash (Glm4MoeLiteForCausalLM) adapter: DeepSeek-V2 MLA + GLM-4-MoE
    routing (dense/sparse per mlp_layer_types)."""

    _testing_eager = None

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.final_rms = True
        self.cfg.uses_rms_norm = True
        # Verified against zai-org/GLM-4.7-Flash: tokenizer has no BOS token.
        self.cfg.default_prepend_bos = False

        # MLA has no per-head q/k/v to fold into; skip LN folding.
        self.supports_fold_ln = False

        # MLA weights keep their HF layout; no QKVO rearrangements apply.
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
                            # Public GLM-4.7 checkpoints set q_lora_rank — two-stage
                            # LoRA Q compression; direct q_proj kept optional for
                            # hypothetical uncompressed variants.
                            "q_a_proj": LinearBridge(name="q_a_proj", optional=True),
                            "q_a_layernorm": GeneralizedComponent(
                                name="q_a_layernorm", optional=True
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
                    # Layers marked "dense" in mlp_layer_types hold a plain gated MLP:
                    # router and shared expert absent, so both are optional.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": Glm4MoeRouterBridge(name="gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
