"""JetMoE architecture adapter.

MIT-IBM's JetMoE (``JetMoeForCausalLM``, native in transformers): the only
open at-scale Mixture-of-Attention-heads model — attention Q and output
projections are per-expert parallel 3D tensors behind a top-k router
(``experts``: JetMoeMoA), with a shared fused KV projection, alongside a
conventional parallel-experts MoE MLP. Both routers are hookable; the
mixers delegate to HF (per-expert 3D projections have no uniform
reconstruction, so no fold target exists either).
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MoEBridge,
    MoERouterBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class _JetMoeAttentionBridge(AttentionBridge):
    """Mixture-of-Attention: no separate q/k/v/o Linears to alias — Q and O
    live inside the per-expert MoA; only the shared fused KV is a Linear."""

    hook_aliases = {
        "hook_kv": "kv.hook_out",
    }


class JetMoeArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for JetMoeForCausalLM models."""

    # Per-expert 3D Q/O projections: nothing to fold a norm into.
    supports_fold_ln = False
    # TopKGating's forward sorts/scatters expert assignments and crashes on
    # the harness's isolated probes; routers stay hookable at runtime.
    component_test_skip_suffixes = ("mlp.gate", "attn.experts.router")

    def __init__(self, cfg: Any) -> None:
        """Initialize the JetMoE architecture adapter."""
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
                    # MoA: delegate; the attention router and shared KV are
                    # hookable, per-expert Q/O stay inside the delegated MoA.
                    "attn": _JetMoeAttentionBridge(
                        name="self_attention",
                        config=self.cfg,
                        submodules={
                            "kv": LinearBridge(name="kv_proj"),
                            "experts": GeneralizedComponent(
                                name="experts",
                                submodules={
                                    # JetMoeTopKGating puts logits last in its 5-tuple.
                                    "router": MoERouterBridge(name="router", logits_index=-1),
                                },
                            ),
                        },
                        maintain_native_attention=True,
                    ),
                    "mlp": MoEBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": MoERouterBridge(name="router", logits_index=-1),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Delegated attention computes rotary inside HF; nothing to wire."""
