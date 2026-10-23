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

import torch.nn as nn

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class _OlmoHybridBlockBridge(BlockBridge):
    """BlockBridge whose norm keys and hook aliases are chosen per layer type.

    ``post_attention_layernorm`` is pre-MLP on linear-attention layers, so only those keep ``ln2``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        # Per-layer targets are set at bind time; this fallback covers the
        # template, which has both ln2 and ln2_post declared.
        self.hook_aliases["hook_mlp_out"] = ["ln2_post.hook_out", "mlp.hook_out"]

    def set_original_component(self, original_component: nn.Module) -> None:
        super().set_original_component(original_component)
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        # Runs before setup_submodules binds the norms, so re-keying is safe.
        if getattr(original_component, "post_feedforward_layernorm", None) is not None:
            if "ln2" in self.submodules:
                self.submodules["ln1_post"] = self.submodules.pop("ln2")
            self.hook_aliases["hook_resid_mid"] = "mlp.hook_in"
            self.hook_aliases["hook_attn_out"] = "ln1_post.hook_out"
            self.hook_aliases["hook_mlp_out"] = "ln2_post.hook_out"
            self.mlp_reads_resid_directly = True
        else:
            self.hook_aliases["hook_resid_mid"] = "ln2.hook_in"
            self.hook_aliases["hook_attn_out"] = "linear_attn.hook_out"
            self.hook_aliases["hook_mlp_out"] = "mlp.hook_out"
            self.mlp_reads_resid_directly = False


class OlmoHybridArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for OlmoHybridForCausalLM models."""

    # Post-norm attention layers and the linear-attention state are not
    # fold-safe; compatibility-mode weight processing does not apply.
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the OLMo Hybrid architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        self.cfg.attn_implementation = "eager"
        self.cfg.is_stateful = True

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": _OlmoHybridBlockBridge(
                name="model.layers",
                submodules={
                    # Linear-attention layers are pre-norm (input_layernorm);
                    # full-attention layers are OLMo2 post-norm and have
                    # post_feedforward_layernorm instead.
                    "ln1": RMSNormalizationBridge(
                        name="input_layernorm", config=self.cfg, optional=True
                    ),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "ln2_post": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg, optional=True
                    ),
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
                    "mlp": self._gated_mlp(),
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
