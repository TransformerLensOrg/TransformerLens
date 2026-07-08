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
    GatedMLPBridge,
    LinearBridge,
    MoEBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
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

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
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
                            "shared_experts": GatedMLPBridge(
                                name="shared_experts",
                                config=self.cfg,
                                optional=True,
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
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire the model-level rotary embedding through to attention bridges."""
        rotary_emb = hf_model.model.rotary_emb

        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
