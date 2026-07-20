"""Bridge for T5Gemma2's merged self+cross decoder attention.

T5Gemma2MergedAttention runs decoder self-attention and encoder-decoder
cross-attention through one module with shared q/k/v/o projections: it projects
``encoder_hidden_states`` through the same ``k_proj``/``v_proj``, concatenates the
encoder K/V onto the decoder K/V, and does a single softmax. In eager mode it
returns ``(attn_output, self_attn_weights, cross_attn_weights)``, where the self
and cross weights are the leading and trailing key-position slices of the merged
pattern.

This bridge delegates the math to the native module (the merged/cross logic
cannot be reimplemented by the manual attention path) while exposing both pattern
slices: ``hook_pattern`` for the self-attention slice (handled by the base class)
and ``hook_cross_pattern`` for the cross-attention slice.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)


class T5Gemma2MergedAttentionBridge(AttentionBridge):
    """Native-delegating attention bridge that also hooks the cross-attention pattern."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # The native module unpacks position_embeddings unconditionally, so
        # component testing must always supply a (cos, sin) tuple.
        kwargs.setdefault("requires_position_embeddings", True)
        super().__init__(*args, **kwargs)
        self.hook_cross_pattern = HookPoint()

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Add the merged-attention inputs the native module requires positionally."""
        inputs = super().get_random_inputs(
            batch_size=batch_size, seq_len=seq_len, device=device, dtype=dtype
        )
        hidden_states = inputs["hidden_states"]
        inputs["merged_attention_mask"] = None
        inputs["encoder_hidden_states"] = torch.randn_like(hidden_states)
        return inputs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Base forward delegates to native and hooks output[0] (hook_out) and the
        # self-attention weights output[1] (hook_pattern). Native returns a third
        # element — the cross-attention weights — which we hook here.
        output = super().forward(*args, **kwargs)
        if isinstance(output, tuple) and len(output) >= 3:
            cross_weights = output[2]
            if isinstance(cross_weights, torch.Tensor) and cross_weights.dim() == 4:
                cross_weights = self.hook_cross_pattern(cross_weights)
                output = output[:2] + (cross_weights,) + output[3:]
        return output
