"""Block bridge component.

This module contains the bridge component for transformer blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

if TYPE_CHECKING:
    pass


class BlockBridge(GeneralizedComponent):
    """Bridge component for transformer blocks.

    This component provides standardized input/output hooks.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the block bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for BlockBridge)
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config)

        # Register submodules from dictionary
        if submodules is not None:
            for module_name, module in submodules.items():
                self.add_module(module_name, module)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the block bridge.

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
        self.hook_outputs.update({"output": output})
        return output
