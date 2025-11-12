"""Base class for generalized transformer components."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.hook_points import HookPoint


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
    # Values can be either a string (single target) or a list of strings (multiple fallback targets)
    hook_aliases: Dict[str, Union[str, List[str]]] = {}
    property_aliases: Dict[str, str] = {}

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, "GeneralizedComponent"]] = None,
        conversion_rule: Optional[BaseHookConversion] = None,
    ):
        """Initialize the generalized component.

        Args:
            name: The name of this component (None if component has no container in remote model)
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
        self._hook_alias_registry: Dict[
            str, Union[str, List[str]]
        ] = {}  # Permanent registry of hook aliases
        self._property_alias_registry: Dict[str, str] = {}  # Permanent registry of property aliases

        # Standardized hooks for all bridge components
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

        # Apply conversion rule to hooks if available
        if self.conversion_rule is not None:
            self.hook_in.hook_conversion = self.conversion_rule
            self.hook_out.hook_conversion = self.conversion_rule

        # Note: _register_aliases() is called later after enable_compatibility_mode()
        # to ensure aliases point to processed weights

    def _register_hook(self, name: str, hook: HookPoint) -> None:
        """Register a hook in the component's hook registry."""
        # Set the name on the HookPoint
        hook.name = name
        # Add to registry
        self._hook_registry[name] = hook

    def _register_aliases(self) -> None:
        """Register aliases from class-level dictionaries.

        This is called ONLY in enable_compatibility_mode() after weight processing.
        It creates actual Python attributes/properties that directly reference the target objects.

        Note: This should only be called when compatibility mode is enabled and after
        weight processing is complete to ensure property aliases point to processed weights.
        """
        # Register hook aliases by storing them in the registry
        if self.hook_aliases:
            self._hook_alias_registry.update(self.hook_aliases)

        # Register property aliases by storing them in the registry
        if self.property_aliases:
            self._property_alias_registry.update(self.property_aliases)

        # Create actual attribute references for hook aliases
        for alias_name, target_path in self._hook_alias_registry.items():
            try:
                # Resolve the target object (handles both single targets and lists)
                if isinstance(target_path, list):
                    # For list-based fallbacks, try each target until one works
                    for single_target in target_path:
                        try:
                            target_obj = self
                            for part in single_target.split("."):
                                target_obj = getattr(target_obj, part)
                            # Found it, set the alias
                            object.__setattr__(self, alias_name, target_obj)
                            break
                        except AttributeError:
                            continue
                else:
                    # Single target
                    target_obj = self
                    for part in target_path.split("."):
                        target_obj = getattr(target_obj, part)
                    object.__setattr__(self, alias_name, target_obj)
            except AttributeError:
                # Target doesn't exist yet, skip
                pass

        # Create actual attribute references for property aliases
        # This way accessing self.W_Q directly returns self.q.weight without any __getattr__ overhead
        for alias_name, target_path in self._property_alias_registry.items():
            try:
                # Check if we should use processed weights instead
                # If _use_processed_weights is True and _processed_{alias_name} exists, use that
                if hasattr(self, "_use_processed_weights") and self._use_processed_weights:
                    processed_attr = f"_processed_{alias_name}"
                    if hasattr(self, processed_attr):
                        target_obj = getattr(self, processed_attr)
                        object.__setattr__(self, alias_name, target_obj)
                        continue

                # Otherwise, resolve the target object from the path
                target_obj = self
                for part in target_path.split("."):
                    target_obj = getattr(target_obj, part)

                # Set the alias as a direct attribute reference
                # This creates a "real" attribute that points to the same object
                object.__setattr__(self, alias_name, target_obj)
            except AttributeError:
                # Target doesn't exist yet, skip
                pass

    def get_hooks(self) -> Dict[str, HookPoint]:
        """Get all hooks registered in this component."""
        hooks = self._hook_registry.copy()

        # Add hook aliases if compatibility mode is enabled
        # Since aliases are now real Python attributes, we can just use getattr
        if self.compatibility_mode and self._hook_alias_registry:
            for alias_name in self._hook_alias_registry.keys():
                if hasattr(self, alias_name):
                    target_hook = getattr(self, alias_name)
                    if isinstance(target_hook, HookPoint):
                        hooks[alias_name] = target_hook

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

        # Get the target dtype from the original component's parameters
        target_dtype = None
        try:
            target_dtype = next(original_component.parameters()).dtype
        except StopIteration:
            # Component has no parameters, keep inputs as-is
            pass

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
                hooked = self.hook_in(kwargs[name])
                # Cast to target dtype if needed and input is a float tensor
                if (
                    target_dtype is not None
                    and isinstance(hooked, torch.Tensor)
                    and hooked.is_floating_point()
                ):
                    hooked = hooked.to(dtype=target_dtype)
                kwargs[name] = hooked
                input_found = True
                break
        # If not in kwargs, try first positional arg
        if not input_found and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            # Cast to target dtype if needed and input is a float tensor
            if target_dtype is not None and hooked_input.is_floating_point():
                hooked_input = hooked_input.to(dtype=target_dtype)
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
        # OPTIMIZATION: Check most common case first (module attributes)
        # Avoid hasattr() which is expensive - use direct dict access
        modules = object.__getattribute__(self, "__dict__").get("_modules")
        if modules is not None and name in modules:
            return modules[name]

        # Avoid recursion by checking if we're looking for original_component
        if name == "original_component":
            # This should not happen since original_component is a property
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # OPTIMIZATION: Check submodules - use direct dict access instead of hasattr
        submodules = object.__getattribute__(self, "__dict__").get("submodules")
        if submodules is not None and name in submodules:
            # Don't delegate to original component for submodules
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Try to get from original_component if it exists
        if modules is not None:
            original_component = modules.get("_original_component")
            if original_component is not None:
                try:
                    # OPTIMIZATION: Check for dots first to avoid split if not needed
                    if "." in name:
                        name_split = name.split(".")
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
