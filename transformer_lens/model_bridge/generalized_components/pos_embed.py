"""Positional embedding bridge component.

This module contains the bridge component for positional embedding layers.
"""
from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class PosEmbedBridge(GeneralizedComponent):
    """Positional embedding bridge that wraps transformer positional embedding layers.

    This component provides standardized input/output hooks for positional embeddings.
    """

    property_aliases = {"W_pos": "weight"}

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the positional embedding bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for PosEmbedBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)

    @property
    def W_pos(self) -> torch.Tensor:
        """Return the positional embedding weight matrix."""
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            if hasattr(self, "_processed_weight"):
                return self._processed_weight
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}")
        assert hasattr(
            self.original_component, "weight"
        ), f"Component {self.name} has no weight attribute"
        weight = self.original_component.weight
        assert isinstance(weight, torch.Tensor), f"Weight is not a tensor for {self.name}"
        return weight

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the positional embedding bridge.

        This method accepts variable arguments to support different architectures:
        - Standard models (GPT-2, GPT-Neo): (input_ids, position_ids=None)
        - OPT models: (attention_mask, past_key_values_length=0, position_ids=None)
        - Others may have different signatures

        Args:
            *args: Positional arguments forwarded to the original component
            **kwargs: Keyword arguments forwarded to the original component

        Returns:
            Positional embeddings
        """
        if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
            input_ids = args[0] if args else kwargs.get("input_ids")
            position_ids = args[1] if len(args) > 1 else kwargs.get("position_ids")
            input_ids = self.hook_in(input_ids)
            if position_ids is None:
                batch_size, seq_len = input_ids.shape[:2]
                position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            if hasattr(self, "_processed_weight"):
                output = torch.nn.functional.embedding(position_ids, self._processed_weight)
            else:
                output = torch.nn.functional.embedding(position_ids, self.W_pos)
            output = self.hook_out(output)
            return output
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        if args:
            first_arg = self.hook_in(args[0])
            args = (first_arg,) + args[1:]
        output = self.original_component(*args, **kwargs)
        output = self.hook_out(output)
        return output

    def set_processed_weight(self, weight: torch.Tensor) -> None:
        """Set the processed weight to use when layer norm is folded.

        Args:
            weight: The processed positional embedding weight tensor
        """
        self._processed_weight = weight
        self._use_processed_weights = True
