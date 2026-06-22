"""GLM-4.5 MoE architecture adapter.

Supports GLM-4.5/4.6/4.7 mixture-of-experts families (`Glm4MoeForCausalLM`).

Key features:
- RMSNorm with partial pre-norm layout.
- RoPE-style rotary embeddings (partial RoPE supported by Hugging Face model logic).
- Q/K normalization blocks (`q_norm`, `k_norm`) and GQA / MQA handling.
- Sparse MoE block in `model.layers[i].mlp`, with optional dense-prefix layers.
- QKVO rearrangements for bridge-side attention hooks.

Optional Parameters (may not exist in state_dict):
-------------------------------------------------
- blocks.{i}.mlp.gate - absent on dense-prefix layers before sparse MoE starts.
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


class Glm4MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for GLM-4.5 / 4.6 / 4.7 MoE decoder models.

    GLM-4x MoE families use RMSNorm, RoPE and sparse routing, with early
    dense-MLP layers in some checkpoints. The dense layers are represented by
    a present-but-slightly-thinner `mlp` sub-module where routing is absent.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the GLM-4 MoE architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        # Force eager attention for output_attentions / compatibility-path parity.
        self.cfg.attn_implementation = "eager"
        # GLM-4 defaults do not prepend BOS in current tiny checkpoints.
        self.cfg.default_prepend_bos = False

        # GQA / MQA support
        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        # QKVO rearrangements; MoE experts and gate are passed through unchanged.
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
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # Dense prefix layers expose `mlp` but no router; mark gate optional
                    # for the dense-MoE boundary.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for GLM-4 MoE component testing."""
        rotary_emb = hf_model.model.rotary_emb

        # Force HF attention implementation to eager so bridge and reference agree
        # on attention-path expectations during eager-only tests.
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
            for layer in hf_model.model.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"

        # Set rotary embeddings on bridge instances if available.
        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)
