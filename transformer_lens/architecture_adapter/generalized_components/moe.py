"""Mixture of Experts bridge component."""

from typing import Any

import torch.nn as nn

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.generalized_components.base import (
    GeneralizedComponent,
)


class MoEBridge(GeneralizedComponent):
    """Bridge for Mixture of Experts components.
    
    This bridge handles MoE layers that consist of multiple expert MLPs and a router.
    """

    def __init__(self, original_component: nn.Module, name: str, architecture_adapter: ArchitectureAdapter):
        """Initialize the MoE bridge.
        
        Args:
            original_component: The original MoE component
            name: The name of this component
            architecture_adapter: The architecture adapter
        """
        super().__init__(original_component, name, architecture_adapter)
        self.experts = nn.ModuleList()
        self.router = None
        
        # Extract experts and router from the original component
        if hasattr(original_component, 'experts'):
            if isinstance(original_component.experts, nn.ModuleList):
                self.experts = original_component.experts
            elif isinstance(original_component.experts, list):
                self.experts = nn.ModuleList(original_component.experts)
        if hasattr(original_component, 'router'):
            self.router = original_component.router

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the MoE layer.
        
        This will execute any registered hooks and then pass through to the original component.
        """
        # Execute pre-hooks if any
        if 'pre' in self.hooks:
            args = self.execute_hooks('pre', args)
            
        # Forward through original component
        output = self.original_component(*args, **kwargs)
        
        # Execute post-hooks if any
        if 'post' in self.hooks:
            output = self.execute_hooks('post', output)
            
        return output 