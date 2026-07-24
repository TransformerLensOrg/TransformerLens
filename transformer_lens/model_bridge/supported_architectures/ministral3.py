"""Ministral 3 (``Ministral3ForCausalLM``) adapter: Mistral module tree (subclasses
the Mistral adapter), plus a llama-4 positional query scale applied after RoPE."""

from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)


class Ministral3ArchitectureAdapter(MistralArchitectureAdapter):
    """Architecture adapter for Ministral3ForCausalLM models."""
