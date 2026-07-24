"""AFMoE (Arcee Trinity, ``AfmoeForCausalLM``) adapter: sandwich norms, QK-norm
attention with sigmoid gating and NoPE/sliding RoPE (so attention delegates to HF),
dense + sparse-MoE MLP layers split at ``num_dense_layers``."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)


class AfmoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for AfmoeForCausalLM models."""

    # Sandwich norms scale sublayer outputs before the residual add; folding
    # ln1/ln2 into the projections changes the function (Trinity-Nano compat
    # mode diverged to loss 10.9 vs 2.3 before this was disabled).
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the AFMoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln1_post": RMSNormalizationBridge(
                        name="post_attention_layernorm", config=self.cfg
                    ),
                    "ln2": RMSNormalizationBridge(name="pre_mlp_layernorm", config=self.cfg),
                    "ln2_post": RMSNormalizationBridge(name="post_mlp_layernorm", config=self.cfg),
                    # Per-head QK-norm before RoPE, RoPE only on sliding
                    # layers, and sigmoid output gating live in HF's forward.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "gate": LinearBridge(name="gate_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                    ),
                    # Dense layers (< num_dense_layers) hold a plain gated MLP
                    # under the same name; router and shared experts are
                    # optional. The tuple-returning router stays unwrapped —
                    # only its inner gate Linear is hookable.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "router_gate": LinearBridge(name="router.gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
