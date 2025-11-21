"""Block bridge component.

This module contains the bridge component for transformer blocks.
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Optional

import torch

from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class BlockBridge(GeneralizedComponent):
    """Bridge component for transformer blocks.

    This component provides standardized input/output hooks and monkey-patches
    HuggingFace blocks to insert hooks at positions matching HookedTransformer.
    """

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
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the block bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for BlockBridge)
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules if submodules is not None else {})
        self._original_block_forward: Optional[Callable[..., Any]] = None
        if submodules is not None and "ln2_post" in submodules:
            self.hook_aliases = self.__class__.hook_aliases.copy()
            self.hook_aliases["hook_mlp_out"] = "ln2_post.hook_out"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the block bridge.

        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments

        Returns:
            The output from the original component

        Raises:
            StopAtLayerException: If stop_at_layer is set and this block should stop execution
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Check if we should stop before executing this block
        # The _stop_at_layer_idx attribute is set by the bridge's forward method
        if hasattr(self, "_stop_at_layer_idx") and self._stop_at_layer_idx is not None:
            # Extract layer index from name
            # Supports multiple naming patterns:
            # - "blocks.0" (TransformerLens style)
            # - "transformer.h.0" (HuggingFace GPT-2 style)
            # - "model.layers.0" (HuggingFace LLaMA style)
            if self.name is not None:
                # Try multiple patterns to extract layer index
                match = (
                    re.search(r"blocks\.(\d+)", self.name)
                    or re.search(r"\.h\.(\d+)", self.name)
                    or re.search(r"\.layers\.(\d+)", self.name)
                )
            else:
                match = None
            if match:
                layer_idx = int(match.group(1))
                if layer_idx == self._stop_at_layer_idx:
                    # Get the input tensor to return
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        input_tensor = args[0]
                    elif "hidden_states" in kwargs and isinstance(
                        kwargs["hidden_states"], torch.Tensor
                    ):
                        input_tensor = kwargs["hidden_states"]
                    else:
                        raise ValueError(f"Cannot find input tensor to stop at layer {layer_idx}")
                    # Run hook_in on the input before stopping
                    input_tensor = self.hook_in(input_tensor)
                    raise StopAtLayerException(input_tensor)

        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        # Filter kwargs to only include parameters accepted by the original component
        # This prevents errors when passing encoder-specific params to decoder-only models
        filtered_kwargs = self._filter_kwargs_for_forward(kwargs, len(args))

        output = self.original_component(*args, **filtered_kwargs)
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

    def _filter_kwargs_for_forward(
        self, kwargs: Dict[str, Any], num_positional_args: int = 0
    ) -> Dict[str, Any]:
        """Filter kwargs to only include parameters accepted by original_component.forward().

        This prevents TypeErrors when the bridge passes parameters (like encoder_attention_mask)
        that aren't accepted by decoder-only models. It also removes any kwargs that would
        conflict with positional arguments already being passed.

        Args:
            kwargs: The full set of keyword arguments
            num_positional_args: Number of positional arguments being passed (to avoid conflicts)

        Returns:
            Filtered kwargs containing only accepted parameters
        """
        if self.original_component is None:
            return kwargs

        try:
            # Get the signature of the original component's forward method
            sig = inspect.signature(self.original_component.forward)
            param_list = list(sig.parameters.keys())
            valid_params = set(param_list)

            # Check if the signature accepts **kwargs (VAR_KEYWORD)
            accepts_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            # If it accepts **kwargs, pass everything through
            if accepts_var_keyword:
                return kwargs

            # Determine which parameters are already satisfied by positional args
            # (to avoid "multiple values for argument" errors)
            positional_param_names = set(param_list[:num_positional_args])

            # Filter kwargs: include only if in signature AND not already provided positionally
            filtered = {
                k: v
                for k, v in kwargs.items()
                if k in valid_params and k not in positional_param_names
            }
            return filtered

        except (ValueError, TypeError):
            # If we can't inspect the signature, pass through all kwargs
            # (better to potentially fail than to silently drop important params)
            return kwargs
