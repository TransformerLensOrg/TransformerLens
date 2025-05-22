"""Mixture of Experts bridge component.

This module contains the bridge component for Mixture of Experts layers.
"""

from typing import Any

import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class MoEBridge(GeneralizedComponent):
    """Bridge component for Mixture of Experts layers.
    
    This component wraps a Mixture of Experts layer from a remote model and provides a consistent interface
    for accessing its weights and performing MoE operations.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: Any | None = None,
    ):
        """Initialize the MoE bridge.
        
        Args:
            original_component: The original MoE component to wrap
            name: The name of the component in the model
            architecture_adapter: The architecture adapter instance
        """
        super().__init__(original_component, name, architecture_adapter)

    @classmethod
    def wrap_component(cls, component: nn.Module, name: str, architecture_adapter: Any | None = None) -> nn.Module:
        """Wrap a component with this bridge if it's a MoE layer.
        
        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance
            
        Returns:
            The wrapped component if it's a MoE layer, otherwise the original component
        """
        if name.endswith(".moe"):
            return cls(component, name, architecture_adapter)
        return component

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the MoE bridge.
        
        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments
            
        Returns:
            The output from the original component
        """
        return self.original_component(*args, **kwargs) 