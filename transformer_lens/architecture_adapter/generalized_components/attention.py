"""Attention bridge component.

This module contains the bridge component for attention layers.
"""

from typing import Any

import torch.nn as nn

from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class AttentionBridge(GeneralizedComponent):
    """Bridge component for attention layers.
    
    This component wraps an attention layer from a remote model and provides a consistent interface
    for accessing its weights and performing attention operations.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: Any | None = None,
    ):
        """Initialize the attention bridge.
        
        Args:
            original_component: The original attention component to wrap
            name: The name of the component in the model
            architecture_adapter: The architecture adapter instance
        """
        super().__init__(original_component, name, architecture_adapter)

    @classmethod
    def wrap_component(cls, component: nn.Module, name: str, architecture_adapter: Any | None = None) -> nn.Module:
        """Wrap a component with this bridge if it's an attention layer.
        
        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance
            
        Returns:
            The wrapped component if it's an attention layer, otherwise the original component
        """
        if name.endswith(".attn"):
            return cls(component, name, architecture_adapter)
        return component 