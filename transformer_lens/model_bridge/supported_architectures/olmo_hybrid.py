"""OLMo Hybrid architecture adapter.

AllenAI's OLMo Hybrid (``OlmoHybridForCausalLM``, Olmo-Hybrid-7B):
alternating layer types — OLMo2-style full-attention layers (post-norms
in the residual, full-width QK-norm, NoPE mode when position embeddings
are withheld) and GatedDeltaNet linear-attention layers (pre-norm, with
separate q/k/v short convolutions). Attention stays HF-native; the
OlmoHybrid GatedDeltaNet variant differs from Qwen3Next's (separate
q/k/v conv states), so it is delegated opaquely rather than through
GatedDeltaNetBridge's reimplementation. Generation uses the model's own
OlmoHybridDynamicCache.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class OlmoHybridArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OlmoHybridForCausalLM models."""

    # Post-norm attention layers and the linear-attention state are not
    # fold-safe; compatibility-mode weight processing does not apply.
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the OLMo Hybrid architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.attn_implementation = "eager"
        self.cfg.is_stateful = True

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        # RMSNormalizationBridge has no optional kwarg; per-layer-type norms
        # are marked optional post-construction.
        ln1 = RMSNormalizationBridge(name="input_layernorm", config=self.cfg)
        ln1.optional = True
        ln2_post = RMSNormalizationBridge(name="post_feedforward_layernorm", config=self.cfg)
        ln2_post.optional = True

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    # Linear-attention layers are pre-norm (input_layernorm);
                    # full-attention layers are OLMo2 post-norm and have
                    # post_feedforward_layernorm instead.
                    "ln1": ln1,
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "ln2_post": ln2_post,
                    "attn": AttentionBridge(
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
                        maintain_native_attention=True,
                        requires_attention_mask=True,
                        optional=True,
                    ),
                    "linear_attn": GeneralizedComponent(name="linear_attn", optional=True),
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def create_stateful_cache(
        self,
        hf_model: Any,
        batch_size: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        """OLMo Hybrid keeps per-layer q/k/v conv states in its own cache class."""
        from transformers.models.olmo_hybrid.modeling_olmo_hybrid import (
            OlmoHybridDynamicCache,
        )

        return OlmoHybridDynamicCache(config=hf_model.config)

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Force eager attention so the NoPE / full-attention mix stays hookable."""
        config = model_kwargs.get("config")
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

    def prepare_model(self, hf_model: Any) -> None:
        """Force eager attention on the loaded HF model."""
        if hasattr(hf_model, "config"):
            hf_model.config._attn_implementation = "eager"
