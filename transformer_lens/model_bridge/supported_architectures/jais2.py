"""Jais 2 architecture adapter.

G42/Inception's Jais 2 Arabic-English family (``Jais2ForCausalLM``,
native in transformers): a pre-LayerNorm rotary decoder with an ungated
biased up/down MLP — the exact Nemotron block shape under the same module
names, so this subclasses Nemotron and only re-adds the attention biases.
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.nemotron import (
    NemotronArchitectureAdapter,
)


class Jais2ArchitectureAdapter(NemotronArchitectureAdapter):
    """Architecture adapter for Jais2ForCausalLM models."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        # Nemotron's fold/center disable is for LayerNorm1P (weight+1 gamma);
        # Jais 2 uses plain nn.LayerNorm, the standard foldable case.
        self.supports_fold_ln = True
        self.supports_center_writing_weights = True
        # Jais 2 sets attention_bias=True; the Nemotron parent is bias-free by
        # default and omits bias reshapes, so Q/K/V biases would keep the flat
        # (n*d_head,) layout instead of (n, d_head).
        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(include_biases=True),
        }
