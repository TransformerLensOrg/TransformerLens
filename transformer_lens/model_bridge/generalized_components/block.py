"""Block bridge component.

This module contains the bridge component for transformer blocks.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class BlockBridge(GeneralizedComponent):
    """Bridge component for transformer blocks.

    This component provides standardized input/output hooks and monkey-patches
    HuggingFace blocks to insert hooks at positions matching HookedTransformer.
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
        "hook_mlp_out": "mlp.hook_out",  # Alias hook_mlp_out to mlp.hook_out
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

        self._original_block_forward: Optional[Callable[..., Any]] = None

        # Conditionally set hook_mlp_out alias based on whether ln2_post exists
        # For models with post-normalization (Gemma-2, Gemma-3), hook_mlp_out should
        # capture the output AFTER ln2_post to match HookedTransformer behavior
        if "ln2_post" in submodules:
            # Create instance-specific hook_aliases dict
            self.hook_aliases = self.__class__.hook_aliases.copy()
            self.hook_aliases["hook_mlp_out"] = "ln2_post.hook_out"

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

        # Apply hook_in to the primary tensor input before delegating
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        output = self.original_component(*args, **kwargs)

        # Apply hook_out on the resulting tensor(s)
        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                first = self.hook_out(first)
                if len(output) == 1:
                    return first
                output = (first,) + output[1:]
            return output

        if isinstance(output, torch.Tensor):
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
        return getattr(self.config, "n_layers", 0)
