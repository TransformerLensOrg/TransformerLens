"""EXAONE 4.0 architecture adapter.

LG AI Research's EXAONE-4.0 (``Exaone4ForCausalLM``, native transformers —
distinct from the remote-code EXAONE-3.x family in exaone.py): llama-shaped
GQA decoder with per-head Q/K RMSNorms, hybrid sliding/global attention
(``layer_types``), and POST-norms applied inside the residual branch
(``post_attention_layernorm`` after attention, ``post_feedforward_layernorm``
after the MLP) with no pre-norms.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class _Exaone4AttentionBridge(PositionEmbeddingsAttentionBridge):
    """Suppress RoPE on hybrid full-attention (global NoPE) layers.

    HF gates rotation on ``sliding_window is None or is_sliding``; the base
    bridge rotates whenever position_embeddings is present, so NoPE layers
    null the argument first. Non-hybrid checkpoints rotate everywhere.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Drop position_embeddings on hybrid full-attention NoPE layers."""
        if self._is_nope_layer():
            kwargs["position_embeddings"] = None
            if len(args) >= 2 and not isinstance(args[1], torch.Tensor):
                args = (args[0], None) + args[2:]
        return super().forward(*args, **kwargs)

    def _is_nope_layer(self) -> bool:
        """Return True when the wrapped attention is a hybrid model's full-attention layer."""
        hf_attn = self.original_component
        if hf_attn is None:
            return False
        if getattr(hf_attn, "sliding_window", None) is None:
            return False
        return not getattr(hf_attn, "is_sliding", True)


class Exaone4ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Exaone4ForCausalLM models."""

    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the EXAONE 4.0 architecture adapter."""
        super().__init__(cfg)

        self._set_rms_rotary_defaults()
        # Same tokenizer family as EXAONE-3.x: no BOS prepending.
        self.cfg.default_prepend_bos = False

        layer_types = getattr(cfg, "layer_types", None)
        if layer_types:
            setattr(self.cfg, "layer_types", list(layer_types))

        # Norms are applied after each sublayer inside the residual branch;
        # pre-LN folding does not apply.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.rotary_emb", config=self.cfg),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "attn": _Exaone4AttentionBridge(
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
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    # Post-norms: ln1 follows attention, ln2 follows the MLP.
                    "ln1": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "mlp": self._gated_mlp(),
                    "ln2": RMSNormalizationBridge(
                        name="post_feedforward_layernorm", config=self.cfg
                    ),
                },
                # Post-norm override: ln2 is post_feedforward_layernorm applied AFTER
                # MLP, so "ln2.hook_in" captures the MLP output (wrong mid-point).
                # The true residual mid-point (between attention and MLP) is mlp.hook_in.
                hook_alias_overrides={
                    "hook_resid_mid": "mlp.hook_in",
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
