"""Mixture of Experts bridge component.

This module contains the bridge component for Mixture of Experts layers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the MoE bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for MoEBridge)
            submodules: Dictionary of GeneralizedComponent submodules to register
        """
        super().__init__(name, config, submodules=submodules)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the MoE bridge.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            The output from the original component
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        if len(args) > 0:
            args = (self.hook_in(args[0]),) + args[1:]
        output = self.original_component(*args, **kwargs)
        output = self.hook_out(output)

        return output
