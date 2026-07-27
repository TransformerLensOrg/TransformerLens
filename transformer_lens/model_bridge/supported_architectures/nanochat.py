"""NanoChat (``NanoChatForCausalLM``) adapter: Llama-style decoder with weightless
RMSNorm (so nothing for fold_ln to fold), ungated relu^2 MLP, and soft-capped logits;
attention delegated (full-width q/k norm after RoPE)."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class NanoChatArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for NanoChatForCausalLM models."""

    # Weightless RMSNorms: no scale to fold into downstream projections.
    supports_fold_ln = False

    def __init__(self, cfg: Any) -> None:
        """Initialize the NanoChat architecture adapter."""
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = False  # ungated relu^2 MLP (fc1 -> act -> fc2)
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        soft_cap = getattr(cfg, "final_logit_softcapping", None) or getattr(
            cfg, "logits_soft_cap", None
        )
        if soft_cap:
            self.cfg.output_logits_soft_cap = float(soft_cap)

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    # q/k norms run AFTER rope here; the bridge reimplementation
                    # applies them before, so delegate attention to HF.
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
                    ),
                    "mlp": self._ungated_mlp(up="fc1", down="fc2"),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
