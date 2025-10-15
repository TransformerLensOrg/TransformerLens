"""Base class for generalized transformer components."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.utilities.aliases import resolve_alias


class GeneralizedComponent(nn.Module):
    """Base class for generalized transformer components.

    This class provides a standardized interface for transformer components
    and handles hook registration and execution.
    """

    # Class attribute indicating whether this component represents a list item (like blocks)
    is_list_item: bool = False

    # Compatibility mode that can be activated/deactivated for legacy components/hooks
    compatibility_mode: bool = False
    # Whether to disable warnings about deprecated hooks
    disable_warnings: bool = False

    # Dictionary mapping deprecated hook names to their new equivalents
    # Subclasses can override this to define their own aliases
    hook_aliases: Dict[str, str] = {}
    property_aliases: Dict[str, str] = {}

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, "GeneralizedComponent"]] = None,
        conversion_rule: Optional[BaseHookConversion] = None,
    ):
        """Initialize the generalized component.

        Args:
            name: The name of this component
            config: Optional configuration object for the component
            submodules: Dictionary of GeneralizedComponent submodules to register
            conversion_rule: Optional conversion rule for this component's hooks
        """
        super().__init__()
        self.name = name
        self.config = config
        self.submodules = submodules or {}
        self.conversion_rule = conversion_rule
        self._hook_registry: Dict[
            str, HookPoint
        ] = {}  # Dynamic registry of hook names to HookPoints

        # Standardized hooks for all bridge components
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

        # Apply conversion rule to hooks if available
        if self.conversion_rule is not None:
            self.hook_in.hook_conversion = self.conversion_rule
            self.hook_out.hook_conversion = self.conversion_rule

    def _register_hook(self, name: str, hook: HookPoint) -> None:
        """Register a hook in the component's hook registry."""
        # Set the name on the HookPoint
        hook.name = name
        # Add to registry
        self._hook_registry[name] = hook

    def get_hooks(self) -> Dict[str, HookPoint]:
        """Get all hooks registered in this component."""
        hooks = self._hook_registry.copy()

        # Add aliases if compatibility mode is enabled
        if self.compatibility_mode and self.hook_aliases:
            # Temporarily suppress warnings during internal hook collection
            original_disable_warnings = getattr(self, "disable_warnings", False)
            self.disable_warnings = True

            try:
                for alias_name, target_name in self.hook_aliases.items():
                    # Use the existing alias system to resolve the target hook
                    target_hook = resolve_alias(self, alias_name, self.hook_aliases)
                    if target_hook is not None:
                        hooks[alias_name] = target_hook
            finally:
                # Restore original warning state
                self.disable_warnings = original_disable_warnings

        return hooks

    def _is_getattr_called_internally(self) -> bool:
        """This function checks if the __getattr__ method was being called internally
        (e.g by the setup process or run_with_cache).
        """
        # Look through the call stack
        for frame_info in inspect.stack():
            if "setup_components" in frame_info.function or "run_with_cache" in frame_info.function:
                return True
        return False

    def set_original_component(self, original_component: nn.Module) -> None:
        """Set the original component that this bridge wraps.

        Args:
            original_component: The original transformer component to wrap
        """
        self.add_module("_original_component", original_component)

    @property
    def original_component(self) -> Optional[nn.Module]:
        """Get the original component."""
        return self._modules.get("_original_component", None)

    def add_hook(self, hook_fn: Callable[..., torch.Tensor], hook_name: str = "output") -> None:
        """Add a hook function (HookedTransformer-compatible interface).

        Args:
            hook_fn: Function to call at this hook point
            hook_name: Name of the hook point (defaults to "output")
        """
        if hook_name == "output":
            self.hook_out.add_hook(hook_fn)
        elif hook_name == "input":
            self.hook_in.add_hook(hook_fn)
        else:
            raise ValueError(
                f"Hook name '{hook_name}' not supported. Supported names are 'output' and 'input'."
            )

    def remove_hooks(self, hook_name: str | None = None) -> None:
        """Remove hooks (HookedTransformer-compatible interface).

        Args:
            hook_name: Name of the hook point to remove. If None, removes all hooks.
        """
        if hook_name is None:
            self.hook_in.remove_hooks()
            self.hook_out.remove_hooks()
        elif hook_name == "output":
            self.hook_out.remove_hooks()
        elif hook_name == "input":
            self.hook_in.remove_hooks()
        else:
            raise ValueError(
                f"Hook name '{hook_name}' not supported. Supported names are 'output' and 'input'."
            )

    def process_weights(
        self,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process weights according to weight processing flags.

        This method should be overridden by specific components that need
        custom weight processing (e.g., QKV splitting, weight rearrangement).

        Args:
            fold_ln: Whether to fold layer norm weights
            center_writing_weights: Whether to center writing weights
            center_unembed: Whether to center unembedding weights
            fold_value_biases: Whether to fold value biases
            refactor_factored_attn_matrices: Whether to refactor factored attention matrices
        """
        # Base implementation does nothing - components override this
        pass

    def custom_weight_processing(
        self, hf_state_dict: Dict[str, torch.Tensor], component_prefix: str, **processing_kwargs
    ) -> Dict[str, torch.Tensor]:
        """Custom weight processing for component-specific transformations.

        This method allows components to perform heavy lifting weight processing
        directly on raw HF weights before general folding operations.

        Args:
            hf_state_dict: Raw HuggingFace state dict
            component_prefix: Prefix for this component's weights (e.g., "transformer.h.0.attn")
            **processing_kwargs: Additional processing arguments

        Returns:
            Dictionary of processed weights ready for general folding operations
        """
        # Base implementation returns empty dict - components can override
        return {}

    def get_processed_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the state dict after weight processing.

        Returns:
            Dictionary mapping parameter names to processed tensors
        """
        # Base implementation returns the standard state dict
        return self.state_dict()

    def get_expected_parameter_names(self, prefix: str = "") -> list[str]:
        """Get the expected TransformerLens parameter names for this component.

        This method should be overridden by specific components to return
        the parameter names they expect in the TransformerLens format.

        Args:
            prefix: Prefix to add to parameter names (e.g., "blocks.0.attn")

        Returns:
            List of expected parameter names in TransformerLens format
        """
        # Base implementation returns empty list - components should override
        return []

    def get_list_size(self) -> int:
        """Get the number of items if this is a list component.

        For components where is_list_item=True, this should return the number
        of items in the list (e.g., number of layers for blocks, number of experts
        for MoE experts).

        Subclasses should override this method to return the correct count
        based on their specific configuration attribute.

        Returns:
            Number of items in the list, or 0 if not a list component
        """
        if not self.is_list_item:
            return 0

        # Base implementation returns 0 - subclasses should override
        return 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Generic forward pass for bridge components with input/output hooks."""
        # Since we use add_module, the component is stored in _modules
        original_component = self._modules.get("_original_component", None)
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        # Try to find the main input
        input_arg_names = [
            "input",
            "hidden_states",
            "input_ids",
            "query_input",
            "x",
            "inputs_embeds",
        ]
        input_found = False
        # Try kwargs first
        for name in input_arg_names:
            if name in kwargs:
                kwargs[name] = self.hook_in(kwargs[name])
                input_found = True
                break
        # If not in kwargs, try first positional arg
        if not input_found and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            args = (hooked_input,) + args[1:]
            input_found = True
        # Call the original component's forward
        output = original_component(*args, **kwargs)

        # Handle tuple outputs from transformer components
        if isinstance(output, tuple):
            # Apply hook to first element (hidden states) and preserve the rest
            hooked_first = self.hook_out(output[0])
            output = (hooked_first,) + output[1:]
        else:
            # Pass output through hook_out
            output = self.hook_out(output)

        return output

    def __getattr__(self, name: str) -> Any:
        # Only called if attribute not found through normal lookup
        # First check if it's a module attribute (like hook_in, hook_out)
        if hasattr(self, "_modules") and name in self._modules:
            return self._modules[name]

        # Only try to resolve aliases if compatibility mode is enabled
        if self.compatibility_mode == True:
            # Check if this is a deprecated hook alias
            resolved_hook = resolve_alias(self, name, self.hook_aliases)
            if resolved_hook is not None:
                return resolved_hook

            # Check if this is a deprecated property alias
            resolved_property = resolve_alias(self, name, self.property_aliases)
            if resolved_property is not None:
                return resolved_property

        # Avoid recursion by checking if we're looking for original_component
        if name == "original_component":
            # This should not happen since original_component is a property
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if this is a submodule that should be registered as a PyTorch module
        # but hasn't been yet. This prevents PyTorch's add_module from failing.
        if name in self.submodules:
            # Don't delegate to original component for submodules
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Try to get from original_component if it exists
        original_component = self._modules.get("_original_component", None)
        if original_component is not None:
            try:
                name_split = name.split(".")

                if len(name_split) > 1:
                    current = getattr(original_component, name_split[0])
                    for part in name_split[1:]:
                        current = getattr(current, part)
                    return current
                else:
                    return getattr(original_component, name)
            except AttributeError:
                pass

        # If we get here, the attribute wasn't found anywhere
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, with passthrough to original component for compatibility."""
        # Handle normal PyTorch module attributes and our own attributes

        # Check if this is a HookPoint being set
        if isinstance(value, HookPoint):
            self._register_hook(name, value)
            super().__setattr__(name, value)
            return

        if name.startswith("_") or name in [
            "name",
            "config",
            "submodules",
            "conversion_rule",
            "compatibility_mode",
            "disable_warnings",
        ]:
            super().__setattr__(name, value)
            return

        # Check if this is a property on our class - if so, try to set it normally
        class_attr = getattr(type(self), name, None)
        if class_attr is not None and isinstance(class_attr, property):
            if class_attr.fset is not None:
                super().__setattr__(name, value)
                return
            # If it's a property with no setter, try the original component instead

        # Try to set the attribute on the original component if we have one
        if hasattr(self, "_modules") and "_original_component" in self._modules:
            original_component = self._modules["_original_component"]
            # Check if the attribute exists on the original component before setting
            if hasattr(original_component, name):
                try:
                    setattr(original_component, name, value)
                    return
                except AttributeError:
                    pass

        # Fall back to normal attribute setting
        super().__setattr__(name, value)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict into the component, forwarding to the original component.

        Args:
            state_dict: Dictionary containing a whole state of the module
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function
            assign: Whether to assign items in the state dictionary to their corresponding keys in the module instead of copying them

        Returns:
            NamedTuple with missing_keys and unexpected_keys fields
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        # Forward the load_state_dict call to the original component
        return self.original_component.load_state_dict(state_dict, strict=strict, assign=assign)

    def has_bias(self) -> bool:
        """Check if the linear layer has a bias."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )

        if not hasattr(self.original_component, "bias"):
            return False
        return self.original_component.bias is not None
