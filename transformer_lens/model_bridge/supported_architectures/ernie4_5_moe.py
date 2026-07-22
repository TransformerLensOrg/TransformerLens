"""ERNIE 4.5 MoE architecture adapter.

Baidu's ERNIE 4.5 MoE (``Ernie4_5_MoeForCausalLM``): the dense ERNIE
attention (GLM-style interleaved RoPE, config-gated biases) with a sparse
MoE MLP — sigmoid-corrected top-k router, batched fused gate_up experts,
optional shared experts, and a dense-MLP prefix before
``moe_layer_start_index``.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class Ernie4_5_MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Ernie4_5_MoeForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the ERNIE 4.5 MoE architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Same conventions as dense ERNIE 4.5.
        self.cfg.rotary_adjacent_pairs = True
        self.cfg.default_prepend_bos = False

        # Biases are config-gated (use_bias); reshape them so a use_bias=True
        # GQA checkpoint gets the (n_kv, d_head) K/V bias layout. No-op when
        # the checkpoint carries no attention biases.
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(include_biases=True),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # Layers before moe_layer_start_index hold a plain gated
                    # MLP; router and shared experts are absent there.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            # Raw-Parameter router; tuple-safe hook via base.
                            "gate": GeneralizedComponent(name="gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Force eager attention and wire the shared rotary onto attention bridges."""
        self._wire_rotary_for_testing(hf_model, bridge_model, eager="config")
