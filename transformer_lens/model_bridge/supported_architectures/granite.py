"""Granite architecture adapter.

Base adapter for the IBM Granite model family. Provides shared config setup and
helper methods used by GraniteMoe and GraniteMoeHybrid variants.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class GraniteArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for IBM Granite models (dense).

    Granite is a Llama-like architecture with RMSNorm, rotary position embeddings
    (RoPE), GQA, and a gated MLP (SiLU activation). Granite-specific scaling
    multipliers are handled by the HF model's native forward pass.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    Granite models do NOT have biases on attention and MLP projections:

    - blocks.{i}.attn.b_Q/b_K/b_V/b_O - No bias on attention projections
    - blocks.{i}.mlp.b_in/b_gate/b_out - No bias on MLP projections
    - blocks.{i}.ln1.b, blocks.{i}.ln2.b, ln_final.b - RMSNorm has no bias
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the Granite architecture adapter."""
        super().__init__(cfg)

        self._setup_common_config(cfg)
        self.weight_processing_conversions = {**self._qkvo_weight_conversions()}
        self.component_mapping = self._build_component_mapping()

    def _setup_common_config(self, cfg: Any) -> None:
        """Set up config variables shared across all Granite variants."""
        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.default_prepend_bos = False
        self.cfg.eps_attr = "variance_epsilon"

        self.default_config = {
            "d_model": cfg.d_model,
            "d_head": cfg.d_model // cfg.n_heads,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "d_vocab": cfg.d_vocab,
        }

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.default_config["n_key_value_heads"] = cfg.n_key_value_heads
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

    def _build_attention_bridge(self, optional: bool = False) -> PositionEmbeddingsAttentionBridge:
        """Build the standard Granite attention bridge."""
        return PositionEmbeddingsAttentionBridge(
            name="self_attn",
            config=self.cfg,
            optional=optional,
            submodules={
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
            },
            requires_attention_mask=True,
            requires_position_embeddings=True,
        )

    def _build_mlp_bridge(self) -> GatedMLPBridge:
        """Build the dense gated MLP bridge."""
        return GatedMLPBridge(
            name="mlp",
            config=self.cfg,
            submodules={
                "gate": LinearBridge(name="gate_proj"),
                "in": LinearBridge(name="up_proj"),
                "out": LinearBridge(name="down_proj"),
            },
        )

    def _build_component_mapping(self) -> dict:
        """Build the full component mapping for dense Granite."""
        return {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": self._build_attention_bridge(),
                    "mlp": self._build_mlp_bridge(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Set up rotary embedding references for Granite component testing.

        Args:
            hf_model: The HuggingFace Granite model instance
            bridge_model: The TransformerBridge model (if available)
        """
        if not hasattr(hf_model.model, "rotary_emb"):
            return

        rotary_emb = hf_model.model.rotary_emb

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if "attn" in block._modules:
                    block.attn.set_rotary_emb(rotary_emb)

        try:
            attn_bridge = self.get_generalized_component("blocks.0.attn")
            attn_bridge.set_rotary_emb(rotary_emb)
        except (AttributeError, KeyError, ValueError):
            pass
