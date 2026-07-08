"""Ministral 3 architecture adapter.

Mistral AI's Ministral 3 (``Ministral3ForCausalLM``, native in
transformers): the edge-scale Mistral — identical GQA q/k/v/o attention,
gated SiLU MLP, RMS pre-norms, and module names, so this is a pure
subclass of the Mistral adapter.
"""

from transformer_lens.model_bridge.supported_architectures.mistral import (
    MistralArchitectureAdapter,
)


class Ministral3ArchitectureAdapter(MistralArchitectureAdapter):
    """Architecture adapter for Ministral3ForCausalLM models."""
