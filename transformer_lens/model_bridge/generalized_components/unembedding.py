"""Unembedding bridge component.

This module contains the bridge component for unembedding layers.
"""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class UnembeddingBridge(GeneralizedComponent):
    """Bridge component for unembedding layers.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: ArchitectureAdapter,
    ):
        """Initialize the unembedding bridge.

        Args:
            original_component: The original unembedding component to wrap
            name: The name of the component in the model
            architecture_adapter: The architecture adapter instance
        """
        super().__init__(original_component, name, architecture_adapter)
        # No extra hooks; use only hook_in and hook_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the unembedding bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Logits output
        """
        hidden_states = self.hook_in(hidden_states)
        output = self.original_component(hidden_states, **kwargs)
        output = self.hook_out(output)
        self.hook_outputs.update({"output": output})
        return output

    @classmethod
    def wrap_component(
        cls, component: nn.Module, name: str, architecture_adapter: ArchitectureAdapter
    ) -> nn.Module:
        """Wrap a component with this bridge if it's an unembedding layer.

        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance

        Returns:
            The wrapped component if it's an unembedding layer, otherwise the original component
        """
        if name.endswith(".unembed") or name.endswith(".lm_head"):
            return cls(component, name, architecture_adapter)
        return component
