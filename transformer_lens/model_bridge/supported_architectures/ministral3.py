"""Ministral 3 architecture adapter.

Mistral AI's Ministral 3 (``Ministral3ForCausalLM``, native in
transformers): same module tree as Mistral, so the mapping subclasses the
Mistral adapter — but not identical math: Ministral3Attention adds a
llama-4 positional query scale after RoPE, run natively on the delegating
text route and by PositionEmbeddingsAttentionBridge on the VLM route.
"""

from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)


class Ministral3ArchitectureAdapter(MistralArchitectureAdapter):
    """Architecture adapter for Ministral3ForCausalLM models."""
