"""Jais 2 architecture adapter.

G42/Inception's Jais 2 Arabic-English family (``Jais2ForCausalLM``,
native in transformers): a pre-LayerNorm rotary decoder with an ungated
biased up/down MLP — the exact Nemotron block shape under the same module
names, so this is a pure subclass.
"""

from transformer_lens.model_bridge.supported_architectures.nemotron import (
    NemotronArchitectureAdapter,
)


class Jais2ArchitectureAdapter(NemotronArchitectureAdapter):
    """Architecture adapter for Jais2ForCausalLM models."""
