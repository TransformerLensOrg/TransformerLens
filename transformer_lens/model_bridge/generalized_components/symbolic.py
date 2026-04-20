"""Symbolic bridge component.

This module contains a bridge component that acts as a structural placeholder
when a model doesn't have a corresponding container for grouped subcomponents.
"""
from typing import Any, Dict, Optional

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class SymbolicBridge(GeneralizedComponent):
    """A placeholder bridge component for maintaining TransformerLens structure.

    This bridge is used when a model doesn't have a container component that exists
    in the TransformerLens standard structure. For example, OPT has fc1/fc2 layers
    directly on the block rather than inside an MLP container.

    When the model is set up, the subcomponents defined in this SymbolicBridge
    are promoted to the parent component, allowing the TransformerLens structure
    to be maintained while correctly mapping to the underlying model's architecture.

    Example usage:
        # OPT doesn't have an "mlp" container - fc1/fc2 are on the block directly
        "mlp": SymbolicBridge(
            submodules={
                "in": LinearBridge(name="fc1"),
                "out": LinearBridge(name="fc2"),
            },
        )

        # During setup, "in" and "out" will be accessible as:
        # - blocks[i].mlp.in  (pointing to blocks[i].fc1)
        # - blocks[i].mlp.out (pointing to blocks[i].fc2)

    Attributes:
        is_symbolic: Always True, indicates this is a structural placeholder.
    """

    is_symbolic: bool = True

    def __init__(
        self,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        config: Optional[Any] = None,
    ):
        """Initialize the SymbolicBridge.

        Args:
            submodules: Dictionary of submodules to register. These will be set up
                       using the parent's original_component as their context.
            config: Optional configuration object
        """
        # SymbolicBridge always has name=None since it doesn't map to a real component
        super().__init__(name=None, config=config, submodules=submodules or {})

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass is not supported for SymbolicBridge.

        SymbolicBridge is a structural placeholder and should not be called directly.
        The actual computation should go through the subcomponents which are set up
        on the parent.

        Raises:
            RuntimeError: Always, since SymbolicBridge should not be called directly.
        """
        raise RuntimeError(
            "SymbolicBridge is a structural placeholder and should not be called directly. "
            "Use the subcomponents (e.g., 'in', 'out') instead."
        )
