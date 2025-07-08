"""Attention bridge component.

This module contains the bridge component for attention layers.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


@dataclass
class AttentionConfig:
    """Configuration for attention bridge behavior.
    
    This class defines how the attention bridge should handle different output types
    and provides flexibility for different attention architectures.
    """
    
    # Output handling configuration
    output_type: str = "auto"  # "auto", "single", "tuple", "dict"
    tuple_output_mapping: Optional[Dict[int, str]] = None  # Maps tuple indices to hook names
    dict_output_mapping: Optional[Dict[str, str]] = None   # Maps dict keys to hook names
    
    # Hook configuration
    additional_hooks: Optional[Dict[str, str]] = None  # Additional hooks to create
    
    # Attention-specific configuration
    cache_attention_weights: bool = True  # Whether to cache attention weights
    cache_attention_patterns: bool = True  # Whether to cache attention patterns
    
    def __post_init__(self):
        """Set up default configurations."""
        if self.tuple_output_mapping is None:
            # Default: first element is hidden states, second is attention weights
            self.tuple_output_mapping = {
                0: "hidden_states",
                1: "attention_weights"
            }
        
        if self.dict_output_mapping is None:
            # Default mappings for common dict output keys
            self.dict_output_mapping = {
                "last_hidden_state": "hidden_states",
                "hidden_states": "hidden_states", 
                "attentions": "attention_weights",
                "attention_weights": "attention_weights"
            }
        
        if self.additional_hooks is None:
            self.additional_hooks = {}


class AttentionBridge(GeneralizedComponent):
    """Bridge component for attention layers.

    This component wraps attention layers from different architectures and provides
    a standardized interface for hook registration and execution with configurable
    output handling.
    """

    def __init__(
        self,
        name: str,
        config: Optional[AttentionConfig] = None,
        submodules: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the attention bridge.

        Args:
            name: The name of this component
            config: Configuration for attention behavior
            submodules: Dictionary of submodules to register (e.g., q_proj, k_proj, etc.)
        """
        super().__init__(name, config)
        
        # Set up configuration with proper type
        if config is None:
            self.config = AttentionConfig()
        elif isinstance(config, AttentionConfig):
            self.config = config
        else:
            # If config is a dict or other type, convert to AttentionConfig
            self.config = AttentionConfig(**config) if isinstance(config, dict) else AttentionConfig()
        
        # Register submodules from dictionary
        if submodules is not None:
            for module_name, module in submodules.items():
                self.add_module(module_name, module)
        
        # Create additional hooks based on configuration
        self._setup_additional_hooks()

    def _setup_additional_hooks(self) -> None:
        """Set up additional hooks based on configuration."""
        # Always create hooks for common attention outputs
        self.hook_hidden_states = HookPoint()
        self.hook_attention_weights = HookPoint()
        
        # Create any additional hooks specified in config
        for hook_name, hook_description in self.config.additional_hooks.items():
            if not hasattr(self, f"hook_{hook_name}"):
                setattr(self, f"hook_{hook_name}", HookPoint())

    def _process_output(self, output: Any) -> Any:
        """Process the output from the original component based on configuration.
        
        Args:
            output: Raw output from the original component
            
        Returns:
            Processed output with hooks applied
        """
        if self.config.output_type == "auto":
            # Auto-detect output type
            if isinstance(output, tuple):
                return self._process_tuple_output(output)
            elif isinstance(output, dict):
                return self._process_dict_output(output)
            else:
                return self._process_single_output(output)
        elif self.config.output_type == "tuple":
            return self._process_tuple_output(output)
        elif self.config.output_type == "dict":
            return self._process_dict_output(output)
        else:  # single
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
            if i in self.config.tuple_output_mapping:
                hook_name = self.config.tuple_output_mapping[i]
                hook = getattr(self, f"hook_{hook_name}", None)
                if hook is not None and element is not None:
                    # Only apply hook if both hook and element exist
                    element = hook(element)
                elif hook is not None and element is None:
                    # Apply hook to None to maintain consistency
                    element = hook(element)
            processed_output.append(element)
        
        # Apply the main hook_out to the first element (hidden states) if it exists
        if len(processed_output) > 0 and processed_output[0] is not None:
            processed_output[0] = self.hook_out(processed_output[0])
        
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
            if key in self.config.dict_output_mapping:
                hook_name = self.config.dict_output_mapping[key]
                hook = getattr(self, f"hook_{hook_name}", None)
                if hook is not None:
                    value = hook(value)
            processed_output[key] = value
        
        # Apply hook_out to the main output (usually hidden_states)
        main_key = next((k for k in output.keys() if "hidden" in k.lower()), None)
        if main_key and main_key in processed_output:
            processed_output[main_key] = self.hook_out(processed_output[main_key])
        
        return processed_output

    def _process_single_output(self, output: torch.Tensor) -> torch.Tensor:
        """Process single tensor output from attention layer.
        
        Args:
            output: Single tensor output from attention
            
        Returns:
            Processed tensor with hooks applied
        """
        # Apply hooks for single tensor output
        output = self.hook_hidden_states(output)
        output = self.hook_out(output)
        return output

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the attention layer.

        This method forwards all arguments to the original component and applies hooks
        to the output based on the configuration.

        Args:
            *args: Input arguments to pass to the original component
            **kwargs: Input keyword arguments to pass to the original component

        Returns:
            The output from the original component, with hooks applied based on config
        """
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}. Call set_original_component() first.")
        
        # Apply input hook
        if "query_input" in kwargs:
            kwargs["query_input"] = self.hook_in(kwargs["query_input"])
        elif "hidden_states" in kwargs:
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            args = (self.hook_in(args[0]),) + args[1:]
        
        # Forward through original component
        output = self.original_component(*args, **kwargs)
        
        # Process output based on configuration
        output = self._process_output(output)
        
        # Update hook outputs for debugging/inspection
        self.hook_outputs.update({"output": output})
        
        return output

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get cached attention weights if available.
        
        Returns:
            Attention weights tensor or None if not cached
        """
        return getattr(self, '_cached_attention_weights', None)

    def get_attention_patterns(self) -> Optional[torch.Tensor]:
        """Get cached attention patterns if available.
        
        Returns:
            Attention patterns tensor or None if not cached
        """
        return getattr(self, '_cached_attention_patterns', None)

    def __repr__(self) -> str:
        """String representation of the AttentionBridge."""
        return f"AttentionBridge(name={self.name}, config={self.config})"
