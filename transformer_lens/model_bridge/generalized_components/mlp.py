"""MLP bridge component.

This module contains the bridge component for MLP layers.
"""


import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MLPBridge(GeneralizedComponent):
    """Bridge component for MLP layers.

    This component wraps an MLP layer from a remote model and provides a consistent interface
    for accessing its weights and performing MLP operations.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: ArchitectureAdapter,
    ):
        """Initialize the MLP bridge.

        Args:
            original_component: The original MLP component to wrap
            name: The name of the component in the model
            architecture_adapter: The architecture adapter instance
        """
        super().__init__(original_component, name, architecture_adapter)

        # Initialize hook points
        self.hook_pre = HookPoint()  # Input to MLP
        self.hook_post = HookPoint()  # Final output

        # Set hook names
        self.hook_pre.name = f"{name}.pre"
        self.hook_post.name = f"{name}.post"

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the MLP bridge.

        Args:
            *args: Positional arguments for the original component
            **kwargs: Keyword arguments for the original component

        Returns:
            Output hidden states
        """
        # The first argument is always the hidden state
        hidden_states = args[0]

        # Apply pre hook
        hidden_states = self.hook_pre(hidden_states)

        # Reconstruct args
        new_args = (hidden_states,) + args[1:]

        # Forward through original component
        output = self.original_component(*new_args, **kwargs)

        # Apply post hook
        output = self.hook_post(output)

        # Store hook outputs
        self.hook_outputs.update({"output": output})

        return output

    @classmethod
    def wrap_component(
        cls, component: nn.Module, name: str, architecture_adapter: ArchitectureAdapter
    ) -> nn.Module:
        """Wrap a component with this bridge if it's an MLP layer.

        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance

        Returns:
            The wrapped component if it's an MLP layer, otherwise the original component
        """
        if name.endswith(".mlp"):
            return cls(component, name, architecture_adapter)
        return component
