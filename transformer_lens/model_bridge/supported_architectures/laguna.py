"""Laguna architecture adapter.

poolside's Laguna coding models (``LagunaForCausalLM``, remote code):
Llama-shaped causal decoders with two first-of-kind mechanisms —
heterogeneous per-layer attention head counts
(``num_attention_heads_per_layer``) and per-head softplus output gating
(``g_proj``) — over a batched-expert MoE (FlexOlmo-style 3D expert
parameters, top-k router, always-on shared experts) with per-layer
dense/sparse selection via ``mlp_layer_types``.

Attention delegates to HF: the bridge reimplementation assumes one
uniform head count, and the softplus gate has no reconstruction. The
per-layer head heterogeneity also rules out uniform Q/K/V reshape
conversions, so no HookedTransformer-format conversions ship and LN
folding is disabled.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class LagunaArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for LagunaForCausalLM models."""

    # Per-layer head counts: no uniform "(n h) m -> n m h" reshape exists.
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the Laguna architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # Heterogeneous head counts + softplus output gating:
                    # delegate; projections and the gate are hookable.
                    "attn": AttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                            "gate": LinearBridge(name="g_proj"),
                            "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                            "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
                        },
                        maintain_native_attention=True,
                    ),
                    # Dense or sparse per mlp_layer_types; router and shared
                    # experts optional so dense layers skip them.
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": GeneralizedComponent(name="gate", optional=True),
                            "shared_experts": self._gated_mlp(name="shared_experts", optional=True),
                            # Dense-layer projections (absent on MoE layers).
                            "dense_gate": LinearBridge(name="gate_proj", optional=True),
                            "dense_in": LinearBridge(name="up_proj", optional=True),
                            "dense_out": LinearBridge(name="down_proj", optional=True),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Delegated attention computes rotary inside HF; nothing to wire."""

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Admit the per-expert→batched expert conversion under Laguna's remote code.

        Laguna ships its own modeling code, so ``from_pretrained`` loads a custom-code
        class. transformers skips its native ``mapping['laguna']`` weight conversion for
        custom-code modules unless the model type is *user*-registered
        (``get_model_conversion_mapping``'s ``is_custom_code`` guard). Without it the
        checkpoint's per-expert ``experts.{i}.gate_proj/up_proj/down_proj`` never merge
        into the batched ``experts.gate_up_proj``/``down_proj`` the model expects, so the
        batched params stay at random init (30,108 unexpected + 234 missing keys) and the
        experts are noise. Registering the existing native mapping here lets the
        conversion run under remote code so the experts load correctly.
        """
        try:
            from transformers.conversion_mapping import (
                USER_REGISTERED_MAPPINGS,
                get_checkpoint_conversion_mapping,
                register_checkpoint_conversion_mapping,
            )
        except ImportError:
            pass  # transformers predates the conversion-mapping API; nothing to register
        else:
            if "laguna" not in USER_REGISTERED_MAPPINGS:
                mapping = get_checkpoint_conversion_mapping("laguna")
                if mapping:
                    register_checkpoint_conversion_mapping("laguna", mapping, overwrite=True)
        super().prepare_loading(model_name, model_kwargs)
