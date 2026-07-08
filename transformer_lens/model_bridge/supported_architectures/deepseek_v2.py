"""DeepSeek V2 architecture adapter.

Supports DeepSeek-V2, DeepSeek-V2-Lite, and DeepSeek-Coder-V2 models
(all use DeepseekV2ForCausalLM).

Key features:
- Multi-Head Latent Attention (MLA): Q and KV compressed via LoRA-style projections.
  DeepSeek-V2-Lite sets q_lora_rank=None, skipping Q compression and using a direct
  q_proj instead — MLAAttentionBridge.forward handles both paths automatically.
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


class DeepSeekV2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for DeepSeek V2 / V2-Lite / Coder-V2 models.

    Uses RMSNorm, MLA with compressed Q/KV projections (or direct Q projection
    when q_lora_rank is None), partial RoPE, MoE on most layers (dense MLP on
    first few), and no biases.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.final_rms = True
        self.cfg.uses_rms_norm = True

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
                            # V2-full (q_lora_rank set): two-stage LoRA Q compression.
                            # These are absent in V2-Lite — marked optional so bridge
                            # setup skips them gracefully. The actual forward call is
                            # handled inside MLAAttentionBridge which checks q_lora_rank.
                            "q_a_proj": LinearBridge(name="q_a_proj", optional=True),
                            # q_a_layernorm is a norm inside the attention block; its
                            # forward is called directly by MLAAttentionBridge, so a
                            # plain GeneralizedComponent (with optional support) suffices.
                            "q_a_layernorm": GeneralizedComponent(
                                name="q_a_layernorm", optional=True
                            ),
                            "q_b_proj": LinearBridge(name="q_b_proj", optional=True),
                            # V2-Lite only: direct Q projection, no compression.
                            "q_proj": LinearBridge(name="q_proj", optional=True),
                            # KV path — always present across all V2 variants.
                            "kv_a_proj_with_mqa": LinearBridge(name="kv_a_proj_with_mqa"),
                            "kv_a_layernorm": RMSNormalizationBridge(
                                name="kv_a_layernorm", config=self.cfg
                            ),
                            "kv_b_proj": LinearBridge(name="kv_b_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    # On dense layers (idx < first_k_dense_replace), shared_experts
                    # are absent — marked optional so setup gracefully skips them when
                    # the layer is DeepseekV2MLP instead of MoE.
                    # Note: the gate module is NOT bridged — DeepseekV2Moe.forward()
                    # calls nn.functional.linear(..., self.gate.weight) directly,
                    # bypassing forward(), so no hook can be attached to it.
                    "mlp": self._build_mlp_bridge(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def _build_mlp_bridge(self):
        """Routed MoE with optional shared experts; Youtu (all-dense) overrides."""
        return MoEBridge(
            name="mlp",
            config=self.cfg,
            submodules={
                "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
            },
        )

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire the shared rotary onto attention bridges (attn implementation untouched)."""
        self._wire_rotary_for_testing(hf_model, bridge_model, eager=None)
