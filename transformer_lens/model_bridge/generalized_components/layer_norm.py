"""Layer norm bridge component implementation."""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class LayerNormBridge(GeneralizedComponent):
    """Layer norm bridge that wraps transformer layer normalization layers.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: ArchitectureAdapter,
    ):
        """Initialize the layer norm bridge.

        Args:
            original_component: The original layer norm component to wrap
            name: The name of this component
            architecture_adapter: Optional architecture adapter for component-specific operations
        """
        super().__init__(original_component, name, architecture_adapter)
        # No extra hooks; use only hook_in and hook_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the layer norm bridge.

        Args:
            hidden_states: Input hidden states
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Normalized output
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
        """Wrap a component with this bridge if it's a LayerNorm layer.

        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance

        Returns:
            The wrapped component if it's a LayerNorm layer, otherwise the original component
        """
        if name.endswith(".ln") or name.endswith(".ln1") or name.endswith(".ln2"):
            return cls(component, name, architecture_adapter)
        return component
