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
        hook_alias_overrides: Optional[Dict[str, str]] = None,
    ):
        """Initialize the block bridge.

        Args:
            name: The name of the component in the model
            config: Optional configuration (unused for BlockBridge)
            submodules: Dictionary of submodules to register
            hook_alias_overrides: Optional dictionary to override default hook aliases.
                For example, {"hook_attn_out": "ln1_post.hook_out"} will make hook_attn_out
                point to ln1_post.hook_out instead of the default attn.hook_out.
        """
        # Apply automatic aliases based on submodules before calling parent
        # This allows submodule-based aliases to be combined with explicit overrides
        auto_overrides = {}
        if submodules is not None:
            # If ln1_post exists, hook_attn_out should point to it instead of attn.hook_out
            # This matches HookedTransformer behavior where ln1_post is applied before hook_attn_out
            if "ln1_post" in submodules:
                auto_overrides["hook_attn_out"] = "ln1_post.hook_out"
            # If ln2_post exists, hook_mlp_out should point to it instead of mlp.hook_out
            if "ln2_post" in submodules:
                auto_overrides["hook_mlp_out"] = "ln2_post.hook_out"

        # Merge automatic and explicit overrides (explicit takes precedence)
        merged_overrides = {**auto_overrides, **(hook_alias_overrides or {})}

        # Call parent with merged overrides
        super().__init__(
            name,
            config,
            submodules=submodules if submodules is not None else {},
            hook_alias_overrides=merged_overrides if merged_overrides else None,
        )
        self._original_block_forward: Optional[Callable[..., Any]] = None

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

        self._check_stop_at_layer(*args, **kwargs)
        args, kwargs = self._hook_input_hidden_states(args, kwargs)

        # Filter kwargs to only include parameters accepted by the original component
        # This prevents errors when passing encoder-specific params to decoder-only models
        filtered_kwargs = self._filter_kwargs_for_forward(kwargs, len(args))

        output = self.original_component(*args, **filtered_kwargs)
        return self._apply_output_hook(output)

    def _apply_output_hook(self, output: Any, wrap_single_element: bool = True) -> Any:
        """Hook the primary tensor in the output and return the result.

        Args:
            output: Raw output from the original component (tensor or tuple).
            wrap_single_element: If True, single-element tuples stay as tuples after
                hooking (default, required by most HF models). If False, single-element
                tuples are unwrapped to a bare tensor (Bloom convention).
        """
        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                first = self.hook_out(first)
                if len(output) == 1:
                    return (first,) if wrap_single_element else first
                output = (first,) + output[1:]
            return output
        if isinstance(output, torch.Tensor):
            output = self.hook_out(output)
        return output

    def _check_stop_at_layer(self, *args: Any, **kwargs: Any) -> None:
        """Check if execution should stop before this block. Raises StopAtLayerException.

        The _stop_at_layer_idx attribute is set by the bridge's forward method.
        Supports TL/GPT-2/LLaMA naming patterns for layer index extraction.
        """
        if not (hasattr(self, "_stop_at_layer_idx") and self._stop_at_layer_idx is not None):
            return
        if self.name is not None:
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
                if len(args) > 0 and isinstance(args[0], torch.Tensor):
                    input_tensor = args[0]
                elif "hidden_states" in kwargs and isinstance(
                    kwargs["hidden_states"], torch.Tensor
                ):
                    input_tensor = kwargs["hidden_states"]
                else:
                    raise ValueError(f"Cannot find input tensor to stop at layer {layer_idx}")
                input_tensor = self.hook_in(input_tensor)
                raise StopAtLayerException(input_tensor)

    def _hook_input_hidden_states(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Apply hook_in to the hidden_states input, whether in args or kwargs."""
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])
        return args, kwargs

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

            # Skip params already provided positionally
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
