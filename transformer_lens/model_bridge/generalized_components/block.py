"""Block bridge component.

This module contains the bridge component for transformer blocks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class BlockBridge(GeneralizedComponent):
    """Bridge component for transformer blocks.

    This component provides standardized input/output hooks.
    """

    # Override the class attribute to indicate this is a list item
    is_list_item: bool = True

    hook_aliases = {
        "hook_resid_pre": "hook_in",
        "hook_resid_mid": "ln2.hook_in",
        "hook_resid_post": "hook_out",
        "hook_attn_in": "attn.hook_in",
        "hook_attn_out": "attn.hook_out",
        "hook_q_input": "attn.q.hook_in",
        "hook_k_input": "attn.k.hook_in",
        "hook_v_input": "attn.v.hook_in",
        "hook_mlp_in": "mlp.hook_in",
        "hook_mlp_out": "mlp.hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the block bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for BlockBridge)
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules)

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

        # Handle tuple outputs from transformer blocks
        if isinstance(output, tuple):
            # Apply hook to first element (hidden states) and preserve the rest
            hooked_first = self.hook_out(output[0])
            output = (hooked_first,) + output[1:]
        else:
            output = self.hook_out(output)

        return output

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        """Get the expected TransformerLens parameter names for this block component.

        Block components delegate to their subcomponents to get parameter names.

        Args:
            prefix: Prefix to add to parameter names (e.g., "blocks.0")

        Returns:
            List of expected parameter names in TransformerLens format
        """
        param_names = []

        # Delegate to all subcomponents
        for sub_name, sub_component in self.submodules.items():
            sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
            param_names.extend(sub_component.get_expected_parameter_names(sub_prefix))

        return param_names

    def get_list_size(self) -> int:
        """Get the number of transformer blocks.

        For BlockBridge, this returns n_layers from the config.

        Returns:
            Number of layers in the model
        """
        if self.config is None:
            return 0
        return getattr(self.config, 'n_layers', 0)
