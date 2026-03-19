"""Convolutional positional embedding bridge component.

This module contains the bridge component for convolutional positional
embeddings used in audio models like HuBERT and wav2vec2.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class ConvPosEmbedBridge(GeneralizedComponent):
    """Bridge for convolutional positional embeddings (HuBERT, wav2vec2).

    Unlike learned absolute position embeddings (PosEmbedBridge) or rotary
    embeddings (RotaryEmbeddingBridge), convolutional positional embeddings
    operate on hidden states via a grouped 1D convolution to produce
    relative positional information.
    """

    hook_aliases = {
        "hook_pos_embed": "hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the convolutional positional embedding bridge.

        Args:
            name: The name of this component (e.g., "hubert.encoder.pos_conv_embed")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules or {})

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the convolutional positional embedding.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            **kwargs: Additional arguments

        Returns:
            Positional embeddings [batch, seq_len, hidden_size]
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(hidden_states, **kwargs)

        if isinstance(output, tuple):
            output = (self.hook_out(output[0]),) + output[1:]
        else:
            output = self.hook_out(output)

        return output
