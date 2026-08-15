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
    """BlockBridge with per-layer-type hook aliases and no hook_resid_mid.

    hook_resid_mid: no single target fits both layer types (ln2.hook_in is the
    mid-point on linear-attention layers but the raw attn-branch output on
    full-attention layers); dropped type-visibly, as on ParallelBlockBridge.

    hook_attn_out / hook_mlp_out must expose the tensor added to the residual
    stream, which also differs by layer type: full-attention layers are OLMo2
    post-norm (contribution = norm output), linear-attention layers are pre-norm
    (contribution = raw sublayer output). Both candidate targets exist on both
    layer types, so alias fallback lists cannot discriminate; instead the
    aliases are selected per layer at bind time in set_original_component.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        self.hook_aliases.pop("hook_resid_mid", None)

    def set_original_component(self, original_component: nn.Module) -> None:
        super().set_original_component(original_component)
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        if getattr(original_component, "post_feedforward_layernorm", None) is not None:
            # Full-attention (post-norm) layer: ln2 = post_attention_layernorm
            # applied after attention, ln2_post = post_feedforward_layernorm.
            # The MLP consumes the mid-residual directly, so the hook_mlp_in
            # capture must sit on the MLP, not on ln2 (whose input here is the
            # raw attention output).
            self.hook_aliases["hook_attn_out"] = "ln2.hook_out"
            self.hook_aliases["hook_mlp_out"] = "ln2_post.hook_out"
            self.mlp_reads_resid_directly = True
        else:
            # Linear-attention (pre-norm) layer.
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
