"""Shared helpers for vision encoder bridges."""

from types import SimpleNamespace
from typing import Any


def vision_attention_config(config: Any) -> Any:
    """Return a config view carrying the vision tower's attention dimensions.

    ``AttentionBridge`` reshapes q/k/v/z hooks from ``n_heads`` at fire time. A
    multimodal model's top-level config describes the language model, so using
    it directly can reshape vision activations with the text tower's head count.
    """
    n_heads = getattr(config, "vision_num_heads", None)
    d_model = getattr(config, "vision_hidden_size", None)
    if not n_heads or not d_model:
        return config
    return SimpleNamespace(n_heads=n_heads, d_model=d_model, d_head=d_model // n_heads)
