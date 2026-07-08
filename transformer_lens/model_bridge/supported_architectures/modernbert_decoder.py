"""ModernBERT Decoder architecture adapter.

JHU CLSP's Ettin decoder suite (``ModernBertDecoderForCausalLM``, native
in transformers): the ModernBERT recipe run causally — alternating
sliding-window/global attention (``layer_types``), fused-GLU MLPs
(``Wi`` produces input and gate halves, ``Wo`` projects back), an
embedding LayerNorm, an Identity attention norm on layer 0, and a
BERT-style prediction head (dense + act + norm) before the untied
``decoder`` vocab projection.

Attention delegates to HF (per-layer sliding windows have no uniform
reconstruction); the layer-0 Identity norm also makes LN folding
unsound, so it is disabled.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class ModernBertDecoderArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for ModernBertDecoderForCausalLM models."""

    # Layer 0's attention norm is Identity: folding assumes a real norm on
    # every sublayer, and centering writing weights assumes every residual
    # reader is mean-invariant (layer 0's attention reads the residual raw).
    supports_fold_ln = False
    supports_center_writing_weights = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the ModernBERT Decoder architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "LN"
        self.cfg.uses_rms_norm = False
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = False  # fused-GLU: Wi carries both halves
        self.cfg.attn_only = False
        self.cfg.final_rms = False

        self.weight_processing_conversions = {}

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embeddings.tok_embeddings"),
            "embed_ln": NormalizationBridge(
                name="model.embeddings.norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    # Identity on layer 0 (pre-normed by the embedding norm).
                    "ln1": NormalizationBridge(
                        name="attn_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    # Sliding-window/global mix per layer_types: delegate.
                    "attn": AttentionBridge(
                        name="attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="Wo"),
                        },
                        maintain_native_attention=True,
                    ),
                    "ln2": NormalizationBridge(
                        name="mlp_norm",
                        config=self.cfg,
                        use_native_layernorm_autograd=True,
                    ),
                    "mlp": MLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="Wi"),
                            "out": LinearBridge(name="Wo"),
                        },
                    ),
                },
            ),
            "ln_final": NormalizationBridge(
                name="model.final_norm",
                config=self.cfg,
                use_native_layernorm_autograd=True,
            ),
            # BERT-style head (dense + act + norm) ahead of the vocab projection.
            "prediction_head": GeneralizedComponent(name="lm_head"),
            "unembed": UnembeddingBridge(name="decoder"),
        }

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Delegated attention computes rotary inside HF; nothing to wire."""
