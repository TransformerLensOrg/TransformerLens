"""EXAONE 4.0 architecture adapter.

LG AI Research's EXAONE-4.0 (``Exaone4ForCausalLM``, native transformers —
distinct from the remote-code EXAONE-3.x family in exaone.py): the OLMo-2
post-norm block shape (per-head Q/K RMSNorms, post_attention_layernorm /
post_feedforward_layernorm inside the residual branch) plus hybrid
sliding/global attention (``layer_types``) with global-NoPE gating.
"""

from typing import Any

import torch

from transformer_lens.model_bridge.generalized_components import (
    PositionEmbeddingsAttentionBridge,
)
from transformer_lens.model_bridge.supported_architectures.olmo2 import (
    Olmo2ArchitectureAdapter,
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


class Exaone4ArchitectureAdapter(Olmo2ArchitectureAdapter):
    """Architecture adapter for Exaone4ForCausalLM models."""

    _attention_bridge_cls = _Exaone4AttentionBridge
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the EXAONE 4.0 architecture adapter."""
        super().__init__(cfg)

        # Same tokenizer family as EXAONE-3.x: no BOS prepending.
        self.cfg.default_prepend_bos = False
        self.supports_center_writing_weights = False

        layer_types = getattr(cfg, "layer_types", None)
        if layer_types:
            setattr(self.cfg, "layer_types", list(layer_types))
