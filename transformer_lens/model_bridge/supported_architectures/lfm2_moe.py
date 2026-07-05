"""LiquidAI LFM2 MoE architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)


class Lfm2MoeBlockBridge(BlockBridge):
    """Whole-layer LFM2 bridge exposing only residual stream hooks.

    LFM2 MoE interleaves short-convolution and full-attention operator layers.
    Wrapping the HF layer as a whole preserves correct execution while avoiding
    unresolved standard attention/MLP aliases on layers that do not have them.
    """

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_post": "hook_out",
    }


class Lfm2MoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for LiquidAI LFM2 MoE models.

    LFM2 MoE is a hybrid decoder with both short-convolution and full-attention
    layers. The adapter delegates each decoder layer to HF and exposes residual
    hooks around the whole layer rather than pretending every layer has a
    homogeneous attention/MLP substructure.
    """

    # Phases 1-3 compare standard attention/MLP components, which this hybrid
    # adapter intentionally doesn't expose (whole-layer residual hooks only).
    # Phase 4 (generation + text-quality) needs no component comparison, so it applies.
    applicable_phases: list[int] = [4]

    def __init__(self, cfg: Any) -> None:
        """Initialize the LFM2 MoE architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.default_prepend_bos = False

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        if hasattr(cfg, "num_experts"):
            self.cfg.num_experts = cfg.num_experts
        if hasattr(cfg, "experts_per_token"):
            self.cfg.experts_per_token = cfg.experts_per_token
        if hasattr(cfg, "moe_intermediate_size"):
            setattr(self.cfg, "moe_intermediate_size", cfg.moe_intermediate_size)
        if hasattr(cfg, "layer_types"):
            setattr(self.cfg, "layer_types", cfg.layer_types)

        norm_eps = getattr(cfg, "norm_eps", None)
        if norm_eps is not None:
            self.cfg.eps = norm_eps

        rope_parameters = getattr(cfg, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta") or getattr(cfg, "rope_theta", None)
        if rope_theta is not None:
            self.cfg.rotary_base = rope_theta

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": Lfm2MoeBlockBridge(name="model.layers", config=self.cfg),
            # LFM2 stores the decoder-final norm at embedding_norm, not model.norm.
            "ln_final": RMSNormalizationBridge(name="model.embedding_norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force eager attention when the HF config exposes the implementation knob."""
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

    def prepare_model(self, hf_model: Any) -> None:
        """Force eager attention on the loaded HF model when supported."""
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"
