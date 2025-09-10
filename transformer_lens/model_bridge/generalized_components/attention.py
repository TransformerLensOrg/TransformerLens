"""Attention bridge component.

This module contains the bridge component for attention layers.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from transformer_lens.conversion_utils.conversion_steps.attention_auto_conversion import (
    AttentionAutoConversion,
)
from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AttentionPatternConversion(BaseHookConversion):
    """Custom conversion rule for attention patterns that always removes batch dimension."""

    def handle_conversion(self, tensor: torch.Tensor, *args) -> torch.Tensor:
        """Convert attention pattern tensor to standard shape [n_heads, pos, pos].

        Args:
            tensor: Input tensor with shape [batch, n_heads, pos, pos] or [n_heads, pos, pos]
            *args: Additional context arguments (ignored)

        Returns:
            Tensor with shape [n_heads, pos, pos]
        """
        if tensor.dim() == 4:
            # Remove batch dimension if present
            return tensor.squeeze(0)
        elif tensor.dim() == 3:
            # Already in correct shape
            return tensor
        else:
            raise ValueError(f"Unexpected tensor shape for attention pattern: {tensor.shape}")


class AttentionBridge(GeneralizedComponent):
    """Bridge component for attention layers.

    This component handles the conversion between Hugging Face attention layers
    and TransformerLens attention components.
    """

    hook_aliases = {
        "hook_result": "hook_out",
        "hook_q": "q.hook_out",
        "hook_k": "k.hook_out",
        "hook_v": "v.hook_out",
        "hook_z": "hook_hidden_states",
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
        config: Any,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        conversion_rule: Optional[BaseHookConversion] = None,
        pattern_conversion_rule: Optional[BaseHookConversion] = None,
    ):
        """Initialize the attention bridge.

        Args:
            name: The name of this component
            config: Model configuration (required for auto-conversion detection)
            submodules: Dictionary of submodules to register (e.g., q_proj, k_proj, etc.)
            conversion_rule: Optional conversion rule. If None, AttentionAutoConversion will be used
            pattern_conversion_rule: Optional conversion rule for attention patterns. If None,
                                   uses AttentionPatternConversion to ensure [n_heads, pos, pos] shape
        """
        # Set up conversion rule - use AttentionAutoConversion if None
        if conversion_rule is None:
            conversion_rule = AttentionAutoConversion(config)

        super().__init__(
            name, config=config, submodules=submodules or {}, conversion_rule=conversion_rule
        )

        # Create only the hook points that are actually used for attention processing
        self.hook_attn_scores = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_hidden_states = HookPoint()

        # Apply conversion rule to attention-specific hooks
        self.hook_hidden_states.hook_conversion = conversion_rule

        # Set up pattern conversion rule - use provided rule or create default
        if pattern_conversion_rule is not None:
            pattern_conversion = pattern_conversion_rule
        else:
            # Use custom conversion rule that always removes batch dimension
            pattern_conversion = AttentionPatternConversion()

        self.hook_pattern.hook_conversion = pattern_conversion

        # Store intermediate values for pattern creation
        self._attn_scores = None
        self._pattern = None

    def _process_output(self, output: Any) -> Any:
        """Process the output from the original component.

        This method intercepts the output to create attention patterns
        the same way as the old implementation and applies hook_out.

        Args:
            output: Raw output from the original component

        Returns:
            Processed output with hooks applied
        """
        # Extract attention scores from the output
        attn_scores = self._extract_attention_scores(output)

        if attn_scores is not None:
            # Create attention pattern the same way as old implementation
            attn_scores = self.hook_attn_scores(attn_scores)
            pattern = F.softmax(attn_scores, dim=-1)
            if not isinstance(pattern, torch.Tensor):
                raise TypeError(f"Expected 'pattern' to be a Tensor, got {type(pattern)}")
            pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
            pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]

            # Store the pattern for potential use in result calculation
            self._pattern = pattern

            # Apply the pattern to the output if needed
            output = self._apply_pattern_to_output(output, pattern)
        else:
            # If no attention scores found, still apply hooks to the output
            if isinstance(output, tuple):
                output = self._process_tuple_output(output)
            elif isinstance(output, dict):
                output = self._process_dict_output(output)
            else:
                output = self._process_single_output(output)

        # Always apply hook_out to the main output
        output = self._apply_hook_out_to_output(output)

        return output

    def _extract_attention_scores(self, output: Any) -> Optional[torch.Tensor]:
        """Extract attention scores from the output.

        Args:
            output: Output from the original component

        Returns:
            Attention scores tensor or None if not found
        """
        if isinstance(output, tuple):
            # Look for attention scores in tuple output
            for element in output:
                if isinstance(element, torch.Tensor) and element.dim() == 4:
                    # Assume 4D tensor is attention scores [batch, heads, query_pos, key_pos]
                    return element
        elif isinstance(output, dict):
            # Look for attention scores in dict output
            for key in ["attentions", "attention_weights", "attention_scores"]:
                if key in output and isinstance(output[key], torch.Tensor):
                    return output[key]

        return None

    def _apply_pattern_to_output(self, output: Any, pattern: torch.Tensor) -> Any:
        """Apply the attention pattern to the output.

        This method simulates how the old implementation uses the pattern
        to calculate the final output.

        Args:
            output: Original output from the component
            pattern: Attention pattern tensor

        Returns:
            Modified output with pattern applied
        """
        if isinstance(output, tuple):
            return self._apply_pattern_to_tuple_output(output, pattern)
        elif isinstance(output, dict):
            return self._apply_pattern_to_dict_output(output, pattern)
        else:
            return self._apply_pattern_to_single_output(output, pattern)

    def _apply_pattern_to_tuple_output(
        self, output: Tuple[Any, ...], pattern: torch.Tensor
    ) -> Tuple[Any, ...]:
        """Apply pattern to tuple output.

        Args:
            output: Tuple output from attention
            pattern: Attention pattern tensor

        Returns:
            Processed tuple with pattern applied
        """
        processed_output = []

        for i, element in enumerate(output):
            if i == 0:  # First element is typically hidden states
                if element is not None:
                    element = self._apply_hook_preserving_structure(
                        element, self.hook_hidden_states
                    )
                    # Apply the pattern to the hidden states
                    element = self._apply_pattern_to_hidden_states(element, pattern)
            elif i == 1 or i == 2:  # Attention weights indices
                if isinstance(element, torch.Tensor):
                    # Replace with our computed pattern
                    element = pattern
            processed_output.append(element)

        return tuple(processed_output)

    def _apply_pattern_to_dict_output(
        self, output: Dict[str, Any], pattern: torch.Tensor
    ) -> Dict[str, Any]:
        """Apply pattern to dictionary output.

        Args:
            output: Dictionary output from attention
            pattern: Attention pattern tensor

        Returns:
            Processed dictionary with pattern applied
        """
        processed_output = {}

        for key, value in output.items():
            if key in ["last_hidden_state", "hidden_states"] and value is not None:
                value = self._apply_hook_preserving_structure(value, self.hook_hidden_states)
                # Apply the pattern to the hidden states
                value = self._apply_pattern_to_hidden_states(value, pattern)
            elif key in ["attentions", "attention_weights"] and value is not None:
                # Replace with our computed pattern
                value = pattern
            processed_output[key] = value

        return processed_output

    def _apply_pattern_to_single_output(
        self, output: torch.Tensor, pattern: torch.Tensor
    ) -> torch.Tensor:
        """Apply pattern to single tensor output.

        Args:
            output: Single tensor output from attention
            pattern: Attention pattern tensor

        Returns:
            Processed tensor with pattern applied
        """
        # Apply hooks for single tensor output
        output = self._apply_hook_preserving_structure(output, self.hook_hidden_states)
        # Apply the pattern to the output
        output = self._apply_pattern_to_hidden_states(output, pattern)
        return output

    def _apply_pattern_to_hidden_states(
        self, hidden_states: torch.Tensor, pattern: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention pattern to hidden states.

        This simulates the old implementation's calculate_z_scores method.

        Args:
            hidden_states: Hidden states tensor
            pattern: Attention pattern tensor

        Returns:
            Modified hidden states with pattern applied
        """
        # This is a simplified version - in the real implementation,
        # we would need to extract V from the original component and apply
        # the pattern properly. For now, we just apply the pattern as a hook.
        return self.hook_hidden_states(hidden_states)

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
            elif i == 1:
                # When use_cache=False, attention weights may be at index 1
                if isinstance(element, torch.Tensor):
                    element = self._apply_hook_preserving_structure(element, self.hook_pattern)
                # else: assume KV cache and skip
            elif i == 2:  # With cache enabled, attention weights are typically at index 2
                if isinstance(element, torch.Tensor):
                    element = self._apply_hook_preserving_structure(element, self.hook_pattern)
            processed_output.append(element)

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
                value = self._apply_hook_preserving_structure(value, self.hook_pattern)
            processed_output[key] = value

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
        else:
            return element

    def _apply_hook_out_to_output(self, output: Any) -> Any:
        """Apply hook_out to the main output tensor.

        Args:
            output: The output to process (can be tensor, tuple, or dict)

        Returns:
            The output with hook_out applied to the main tensor
        """
        if isinstance(output, torch.Tensor):
            return self.hook_out(output)
        elif isinstance(output, tuple) and len(output) > 0:
            # Apply hook_out to the first element (typically hidden states)
            processed_tuple = list(output)
            if isinstance(output[0], torch.Tensor):
                processed_tuple[0] = self.hook_out(output[0])
            return tuple(processed_tuple)
        elif isinstance(output, dict):
            # Apply hook_out to the main hidden states in dictionary
            processed_dict = output.copy()
            for key in ["last_hidden_state", "hidden_states"]:
                if key in processed_dict and isinstance(processed_dict[key], torch.Tensor):
                    processed_dict[key] = self.hook_out(processed_dict[key])
                    break  # Only apply to the first found key
            return processed_dict
        else:
            return output

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
