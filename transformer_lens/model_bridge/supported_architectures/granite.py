"""Granite architecture adapter.

Base adapter for the IBM Granite model family. Provides shared config setup and
helper methods used by GraniteMoe and GraniteMoeHybrid variants.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    MoEBridge,
    MoERouterBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    ScaledResidualBlockBridge,
    UnembeddingBridge,
)


class GraniteArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for IBM Granite models (dense).

    Granite is a Llama-like architecture with RMSNorm, rotary position embeddings
    (RoPE), GQA, and a gated MLP (SiLU activation). Granite-specific scaling
    multipliers are applied by the HF model's native forward pass;
    ScaledResidualBlockBridge accounts for residual_multiplier so
    hook_attn_out / hook_mlp_out expose the scaled residual contributions.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    Granite models do NOT have biases on attention and MLP projections:

    - blocks.{i}.attn.b_Q/b_K/b_V/b_O - No bias on attention projections
    - blocks.{i}.mlp.b_in/b_gate/b_out - No bias on MLP projections
    - blocks.{i}.ln1.b, blocks.{i}.ln2.b, ln_final.b - RMSNorm has no bias
    """

    _testing_hybrid = True
    _testing_eager = None

    def __init__(self, cfg: Any) -> None:
        """Initialize the Granite architecture adapter."""
        super().__init__(cfg)

        self._setup_common_config(cfg)
        self.weight_processing_conversions = {**self._qkvo_weight_conversions()}
        self.component_mapping = self._build_component_mapping()

    def _setup_common_config(self, cfg: Any) -> None:
        """Set up config variables shared across all Granite variants."""
        self._set_rms_rotary_defaults()
        self.cfg.default_prepend_bos = False

    def _build_attention_bridge(self, optional: bool = False) -> AttentionBridge:
        """Build the standard Granite attention bridge (GraniteMoeHybrid passes optional)."""
        return self._qkvo_attention_bridge(optional=optional)

    def _build_mlp_bridge(self) -> GatedMLPBridge:
        """Build the dense gated MLP bridge."""
        return self._gated_mlp()

    def _build_moe_bridge(self) -> MoEBridge:
        """Sparse MoE block with a hookable router.

        logits_index=-1: GraniteMoe-family routers return
        (top_k_index, top_k_weights, router_logits) — index 0 is an int64
        index tensor, so the default would hook the wrong element.
        """
        return MoEBridge(
            name="block_sparse_moe",
            config=self.cfg,
            submodules={
                "gate": MoERouterBridge(
                    name="router", logits_index=-1, indices_index=0, weights_index=1
                )
            },
        )

    def _build_component_mapping(self) -> dict:
        """Build the full component mapping for dense Granite."""
        return {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            # HF multiplies each sublayer output by residual_multiplier before the
            # residual add, so hook_attn_out / hook_mlp_out must expose the scaled
            # contribution, not the raw module output.
            "blocks": ScaledResidualBlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": self._build_attention_bridge(),
                    "mlp": self._build_mlp_bridge(),
                },
                residual_contribution_scale=getattr(self.cfg, "residual_multiplier", 1.0),
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def apply_output_logits_transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Match Granite's ``lm_head / logits_scaling`` output path."""
        scaling = float(getattr(self.cfg, "logits_scaling", 1.0))
        return super().apply_output_logits_transform(logits / scaling)
