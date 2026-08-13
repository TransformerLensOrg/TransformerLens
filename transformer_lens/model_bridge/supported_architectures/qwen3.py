"""Qwen3 architecture adapter.

Base adapter for the Qwen3 model family. Provides shared config setup,
attention bridge construction, and setup_component_testing used by
Qwen3, Qwen3.5, and Qwen3Next variants.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.gated_delta_net import (
    GatedDeltaNetBridge,
)


class Qwen3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen3 dense models.

    RMSNorm, RoPE, GQA, Q/K head norms, gated MLP. No biases.
    Serves as base class for Qwen3.5 and Qwen3Next hybrid variants.
    """

    _testing_hybrid = True

    def __init__(self, cfg: Any, *, hybrid: bool = False, lm_prefix: str = "model") -> None:
        super().__init__(cfg)
        self._setup_qwen3_config(cfg)
        if hybrid:
            self.supports_fold_ln = False
            self.weight_processing_conversions: dict = {}
        else:
            self.weight_processing_conversions = {**self._qkvo_weight_conversions()}
        self.component_mapping = self._build_component_mapping(hybrid=hybrid, lm_prefix=lm_prefix)

    def _setup_qwen3_config(self, cfg: Any) -> None:
        """Config shared across all Qwen3 variants (dense, hybrid, MoE)."""
        self._set_rms_rotary_defaults()
        self.cfg.default_prepend_bos = False
        self.cfg.attn_implementation = "eager"

    def _build_attention_bridge(self, optional: bool = False) -> AttentionBridge:
        """Standard Qwen3 attention bridge with Q/K norms."""
        return self._qkvo_attention_bridge(
            optional=optional,
            extra_submodules={
                "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
            },
            # None omits the flags, keeping the bridge class's own defaults.
            requires_attention_mask=None,
            requires_position_embeddings=None,
        )

    def _build_mlp_bridge(self):
        """Dense gated MLP (gate_proj + up_proj -> down_proj). Override for MoE."""
        return self._gated_mlp()

    def _build_linear_attn_bridge(self, optional: bool = False) -> GatedDeltaNetBridge:
        """GatedDeltaNet linear-attention bridge for hybrid variants."""
        return GatedDeltaNetBridge(
            name="linear_attn",
            config=self.cfg,
            optional=optional,
        )

    def _build_component_mapping(self, *, hybrid: bool = False, lm_prefix: str = "model") -> dict:
        """Parametric component mapping. hybrid=True adds optional linear_attn; lm_prefix
        nests the text model (``model``, or ``model.language_model`` for multimodal). lm_head
        stays top-level.
        """
        block_submodules: dict = {
            "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
            "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
            "attn": self._build_attention_bridge(optional=hybrid),
            "mlp": self._build_mlp_bridge(),
        }
        if hybrid:
            block_submodules["linear_attn"] = self._build_linear_attn_bridge(optional=True)
        return {
            "embed": EmbeddingBridge(name=f"{lm_prefix}.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name=f"{lm_prefix}.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(name=f"{lm_prefix}.layers", submodules=block_submodules),
            "ln_final": RMSNormalizationBridge(name=f"{lm_prefix}.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }
