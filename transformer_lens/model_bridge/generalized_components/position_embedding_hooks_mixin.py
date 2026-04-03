"""Mixin for position embedding hooks (cos/sin) shared across attention bridges."""
from __future__ import annotations

from typing import Any

from transformer_lens.hook_points import HookPoint


class PositionEmbeddingHooksMixin:
    """Mixin providing hook_cos/hook_sin and _apply_position_embedding_hooks().

    Used by both PositionEmbeddingsAttentionBridge and
    JointQKVPositionEmbeddingsAttentionBridge to avoid duplicating this logic.
    """

    def _init_position_embedding_hooks(self):
        """Initialize rotary embedding state and hooks. Call from __init__."""
        self._rotary_emb = None
        self.hook_cos = HookPoint()
        self.hook_sin = HookPoint()

    def set_rotary_emb(self, rotary_emb: Any) -> None:
        """Set reference to the model's rotary embedding component."""
        self._rotary_emb = rotary_emb

    def _apply_position_embedding_hooks(self, position_embeddings):
        """Apply hook_cos/hook_sin to a (cos, sin) position embeddings tuple."""
        if isinstance(position_embeddings, tuple) and len(position_embeddings) == 2:
            cos, sin = position_embeddings
            hooked_cos = self.hook_cos(cos)
            hooked_sin = self.hook_sin(sin)
            return (hooked_cos, hooked_sin)
        return position_embeddings
