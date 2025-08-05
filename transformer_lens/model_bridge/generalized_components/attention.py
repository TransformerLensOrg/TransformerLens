"""Attention bridge component.

This module contains the bridge component for attention layers.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AttentionBridge(GeneralizedComponent):
    """Bridge component for attention layers.

    This component wraps attention layers from different architectures and provides
    a standardized interface for hook registration and execution.
    """

    hook_aliases = {
        "hook_pattern": "hook_in",
        "hook_result": "hook_hidden_states",
        "hook_attn_scores": "o.hook_in",
        "hook_q": "q.hook_out",
        "hook_k": "k.hook_out",
        "hook_v": "v.hook_out",
        "hook_z": "o.hook_out",
    }

    property_aliases = {
        "W_Q": "q.weight",
        "W_K": "k.weight",
        "W_V": "v.weight",
        "W_O": "o.weight",
        "b_Q": "q.bias",
        "b_K": "k.bias",
        "b_V": "v.bias",
        "b_O": "o.bias",
    }

    def __init__(
        self,
        name: str,
        submodules: Optional[Dict[str, GeneralizedComponent]] = {},
    ):
        """Initialize the attention bridge.

        Args:
            name: The name of this component
            submodules: Dictionary of submodules to register (e.g., q_proj, k_proj, etc.)
        """
        super().__init__(name, submodules=submodules)
        self.hook_hidden_states = HookPoint()
        self.hook_attention_weights = HookPoint()

    def _process_output(self, output: Any) -> Any:
        """Process the output from the original component.

        Args:
            output: Raw output from the original component

        Returns:
            Processed output with hooks applied
        """
        if isinstance(output, tuple):
            return self._process_tuple_output(output)
        elif isinstance(output, dict):
            return self._process_dict_output(output)
        else:
            return self._process_single_output(output)

    def _process_tuple_output(self, output: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Process tuple output from attention layer.

        Args:
            output: Tuple output from attention

        Returns:
            Processed tuple with hooks applied
        """
        processed_output = []

        for i, element in enumerate(output):
            if i == 0:  # First element is typically hidden states
                if element is not None:
                    element = self._apply_hook_preserving_structure(
                        element, self.hook_hidden_states
                    )
            elif i == 1:  # Second element is typically attention weights
                if element is not None:
                    element = self._apply_hook_preserving_structure(
                        element, self.hook_attention_weights
                    )
            processed_output.append(element)

        # Apply the main hook_out to the first element (hidden states) if it exists
        if len(processed_output) > 0 and processed_output[0] is not None:
            processed_output[0] = self._apply_hook_preserving_structure(
                processed_output[0], self.hook_out
            )

        return tuple(processed_output)

    def _process_dict_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Process dictionary output from attention layer.

        Args:
            output: Dictionary output from attention

        Returns:
            Processed dictionary with hooks applied
        """
        processed_output = {}

        for key, value in output.items():
            if key in ["last_hidden_state", "hidden_states"] and value is not None:
                value = self._apply_hook_preserving_structure(value, self.hook_hidden_states)
            elif key in ["attentions", "attention_weights"] and value is not None:
                value = self._apply_hook_preserving_structure(value, self.hook_attention_weights)
            processed_output[key] = value

        # Apply hook_hidden_states and hook_out to the main output (usually hidden_states)
        main_key = next((k for k in output.keys() if "hidden" in k.lower()), None)
        if main_key and main_key in processed_output:
            processed_output[main_key] = self._apply_hook_preserving_structure(
                processed_output[main_key], self.hook_out
            )

        return processed_output

    def _process_single_output(self, output: torch.Tensor) -> torch.Tensor:
        """Process single tensor output from attention layer.

        Args:
            output: Single tensor output from attention

        Returns:
            Processed tensor with hooks applied
        """
        # Apply hooks for single tensor output
        output = self._apply_hook_preserving_structure(output, self.hook_hidden_states)
        output = self._apply_hook_preserving_structure(output, self.hook_out)
        return output

    def _apply_hook_preserving_structure(self, element: Any, hook_fn) -> Any:
        """Apply a hook while preserving the original structure.

        Args:
            element: The element to process (tensor, tuple, etc.)
            hook_fn: The hook function to apply to tensors

        Returns:
            The processed element with the same structure as input
        """
        if isinstance(element, torch.Tensor):
            return hook_fn(element)
        elif isinstance(element, tuple) and len(element) > 0:
            # For tuple outputs, process the first element if it's a tensor
            processed_elements = list(element)
            if isinstance(element[0], torch.Tensor):
                processed_elements[0] = hook_fn(element[0])
            return tuple(processed_elements)
        # For other types, return as-is
        return element

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the attention layer.

        This method forwards all arguments to the original component and applies hooks
        to the output.

        Args:
            *args: Input arguments to pass to the original component
            **kwargs: Input keyword arguments to pass to the original component

        Returns:
            The output from the original component, with hooks applied
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Apply input hook
        if "query_input" in kwargs:
            kwargs["query_input"] = self.hook_in(kwargs["query_input"])
        elif "hidden_states" in kwargs:
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            args = (self.hook_in(args[0]),) + args[1:]

        # Forward through original component
        output = self.original_component(*args, **kwargs)

        # Process output
        output = self._process_output(output)

        # Update hook outputs for debugging/inspection

        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get cached attention weights if available.

        Returns:
            Attention weights tensor or None if not cached
        """
        return getattr(self, "_cached_attention_weights", None)

    def get_attention_patterns(self) -> Optional[torch.Tensor]:
        """Get cached attention patterns if available.

        Returns:
            Attention patterns tensor or None if not cached
        """
        return getattr(self, "_cached_attention_patterns", None)

    def __repr__(self) -> str:
        """String representation of the AttentionBridge."""
        return f"AttentionBridge(name={self.name})"
