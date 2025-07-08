"""Embedding bridge component.

This module contains the bridge component for embedding layers.
"""

from typing import Any

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class EmbeddingBridge(GeneralizedComponent):
    """Embedding bridge that wraps transformer embedding layers.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        original_component: nn.Module,
        name: str,
        architecture_adapter: ArchitectureAdapter,
    ):
        """Initialize the embedding bridge.

        Args:
            original_component: The original embedding component to wrap
            name: The name of this component
            architecture_adapter: Optional architecture adapter for component-specific operations
        """
        super().__init__(original_component, name, architecture_adapter)
        # No extra hooks; use only hook_in and hook_out

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the embedding bridge.

        Args:
            input_ids: Input token IDs
            position_ids: Optional position IDs (ignored if not supported)
            **kwargs: Additional arguments to pass to the original component

        Returns:
            Embedded output
        """
        input_ids = self.hook_in(input_ids)
        # Remove position_ids if not supported
        if (
            not hasattr(self.original_component, "forward")
            or "position_ids" not in self.original_component.forward.__code__.co_varnames
        ):
            kwargs.pop("position_ids", None)
            output = self.original_component(input_ids, **kwargs)
        else:
            output = self.original_component(input_ids, position_ids=position_ids, **kwargs)
        output = self.hook_out(output)
        self.hook_outputs.update({"output": output})
        return output

    @classmethod
    def wrap_component(
        cls, component: nn.Module, name: str, architecture_adapter: ArchitectureAdapter
    ) -> nn.Module:
        """Wrap a component with this bridge if it's an embedding layer.

        Args:
            component: The component to wrap
            name: The name of the component
            architecture_adapter: The architecture adapter instance

        Returns:
            The wrapped component if it's an embedding layer, otherwise the original component
        """
        if name.endswith(".embed") or name.endswith(".embed_tokens"):
            return cls(component, name, architecture_adapter)
        return component
