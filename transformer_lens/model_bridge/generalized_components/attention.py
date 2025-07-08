"""Attention bridge component.

This module contains the bridge component for attention layers.
"""

from typing import Any

import torch.nn as nn

from transformer_lens.HookedTransformer import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AttentionBridge(GeneralizedComponent):
    """Bridge component for attention layers.

    This component wraps attention layers from different architectures and provides
    a standardized interface for hook registration and execution.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: Any,
    ):
        """Initialize the attention bridge.

        Args:
            original_component: The original attention component to wrap
            name: The name of this component
            architecture_adapter: Architecture adapter for component-specific operations
        """
        super().__init__(original_component, name, architecture_adapter)
        # No extra hooks; use only hook_in and hook_out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the attention layer.

        This method forwards all arguments to the original component and applies hooks
        to the output. The arguments should match the original component's forward method.

        Args:
            *args: Input arguments to pass to the original component
            **kwargs: Input keyword arguments to pass to the original component

        Returns:
            The output from the original component, with hooks applied
        """
        # Use hook_in on the main input (query_input if present, else first arg)
        if "query_input" in kwargs:
            kwargs["query_input"] = self.hook_in(kwargs["query_input"])
        elif len(args) > 0:
            args = (self.hook_in(args[0]),) + args[1:]
        output = self.original_component(*args, **kwargs)
        output = self.hook_out(output)
        self.hook_outputs.update({"output": output})
        return output
