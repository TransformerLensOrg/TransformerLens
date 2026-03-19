"""Bridge component for convolutional positional embeddings (HuBERT, wav2vec2)."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class ConvPosEmbedBridge(GeneralizedComponent):
    """Wraps a grouped 1D conv that produces relative positional information.

    Unlike PosEmbedBridge (lookup table) or RotaryEmbeddingBridge (rotation matrices),
    this operates on hidden states via convolution.
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
        super().__init__(name, config, submodules=submodules or {})

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """hidden_states: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size]"""
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
