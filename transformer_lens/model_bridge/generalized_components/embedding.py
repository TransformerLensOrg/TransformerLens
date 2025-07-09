"""Embedding bridge component.

This module contains the bridge component for embedding layers.
"""

from typing import Any, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class EmbeddingBridge(GeneralizedComponent):
    """Embedding bridge that wraps transformer embedding layers.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
    ):
        """Initialize the embedding bridge.

        Args:
            name: The name of this component
            config: Optional configuration (unused for EmbeddingBridge)
        """
        super().__init__(name, config)
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
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Try to access hook_in directly from _modules
        hook_in = self._modules.get("hook_in", None)
        if hook_in is not None:
            input_ids = hook_in(input_ids)

        # Remove position_ids if not supported
        if (
            not hasattr(self.original_component, "forward")
            or "position_ids" not in self.original_component.forward.__code__.co_varnames
        ):
            kwargs.pop("position_ids", None)
            output = self.original_component(input_ids, **kwargs)
        else:
            output = self.original_component(input_ids, position_ids=position_ids, **kwargs)

        # Try to access hook_out directly from _modules
        hook_out = self._modules.get("hook_out", None)
        if hook_out is not None:
            output = hook_out(output)

        self.hook_outputs.update({"output": output})
        return output
