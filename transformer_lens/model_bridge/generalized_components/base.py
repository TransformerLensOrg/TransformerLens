"""Base class for generalized transformer components."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)
from transformer_lens.hook_points import HookPoint


class GeneralizedComponent(nn.Module):
    """Base class for generalized transformer components.

    This class provides a standardized interface for transformer components
    and handles hook registration and execution.
    """

    is_list_item: bool = False
    compatibility_mode: bool = False
    disable_warnings: bool = False
    hook_aliases: Dict[str, Union[str, List[str]]] = {}
    property_aliases: Dict[str, str] = {}

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, "GeneralizedComponent"]] = None,
        conversion_rule: Optional[BaseTensorConversion] = None,
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
        self._hook_registry: Dict[str, HookPoint] = {}
        self._hook_alias_registry: Dict[str, Union[str, List[str]]] = {}
        self._property_alias_registry: Dict[str, str] = {}
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()
        # real_components maps TL keys to (remote_path, actual_instance) tuples
        # For list components, actual_instance will be a list of component instances
        self.real_components: Dict[str, tuple] = {}
        if self.conversion_rule is not None:
            self.hook_in.hook_conversion = self.conversion_rule
            self.hook_out.hook_conversion = self.conversion_rule

    def _register_hook(self, name: str, hook: HookPoint) -> None:
        """Register a hook in the component's hook registry."""
        hook.name = name
        self._hook_registry[name] = hook

    def _register_aliases(self) -> None:
        """Register aliases from class-level dictionaries.

        This is called ONLY in enable_compatibility_mode() after weight processing.
        It creates actual Python attributes/properties that directly reference the target objects.

        Note: This should only be called when compatibility mode is enabled and after
        weight processing is complete to ensure property aliases point to processed weights.
        """
        if self.hook_aliases:
            self._hook_alias_registry.update(self.hook_aliases)
        if self.property_aliases:
            self._property_alias_registry.update(self.property_aliases)
        for alias_name, target_path in self._hook_alias_registry.items():
            try:
                if isinstance(target_path, list):
                    for single_target in target_path:
                        try:
                            target_obj = self
                            for part in single_target.split("."):
                                target_obj = getattr(target_obj, part)
                            object.__setattr__(self, alias_name, target_obj)
                            break
                        except AttributeError:
                            continue
                else:
                    target_obj = self
                    for part in target_path.split("."):
                        target_obj = getattr(target_obj, part)
                    object.__setattr__(self, alias_name, target_obj)
            except AttributeError:
                pass
        for alias_name, target_path in self._property_alias_registry.items():
            try:
                target_obj = self
                for part in target_path.split("."):
                    target_obj = getattr(target_obj, part)
                object.__setattr__(self, alias_name, target_obj)
            except AttributeError:
                pass

    def get_hooks(self) -> Dict[str, HookPoint]:
        """Get all hooks registered in this component."""
        hooks = self._hook_registry.copy()
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

    def set_processed_weights(
        self, weights: Dict[str, torch.Tensor], verbose: bool = False
    ) -> None:
        """Set the processed weights for use in compatibility mode.

        This method stores processed weights as attributes on the component so they can be
        used directly in the forward pass without modifying the original component.

        Components should override this method to handle their specific weight structure.
        The weights dict contains keys like "weight", "bias", "W_in", "W_out", etc.

        If this component has submodules, this method will automatically distribute the
        weights to those subcomponents using ProcessWeights.distribute_weights_to_components.

        Args:
            weights: Dictionary of processed weight tensors
            verbose: If True, print detailed information about weight setting
        """
        if verbose:
            print(
                f"\n  set_processed_weights: {self.__class__.__name__} (name={getattr(self, 'name', 'unknown')})"
            )
            print(f"    Received {len(weights)} weight keys")

        # First, handle single-part keys (keys without ".") by setting them as parameters
        # on the original component
        if self.original_component is not None:
            for key, weight_tensor in weights.items():
                # Only process keys without "." (single-part keys)
                if "." not in key:
                    # Try to set the parameter on the original component
                    if hasattr(self.original_component, key):
                        param = getattr(self.original_component, key)
                        if param is not None and isinstance(param, torch.nn.Parameter):
                            # Check that shapes match
                            if param.shape != weight_tensor.shape:
                                raise ValueError(
                                    f"Shape mismatch when setting weight '{key}' in {type(self.original_component).__name__}: "
                                    f"existing param shape {param.shape} != new tensor shape {weight_tensor.shape}"
                                )
                            if verbose:
                                print(f"    Setting weight: {key} (shape: {weight_tensor.shape})")
                            # break tying by creating a new param
                            new_param = nn.Parameter(weight_tensor)
                            setattr(self.original_component, key, new_param)
                        elif param is None:
                            # Parameter exists but is None (e.g., bias=False in nn.Linear)
                            # Create a new parameter from the weight tensor
                            if verbose:
                                print(
                                    f"    Creating weight: {key} (shape: {weight_tensor.shape}) - was None"
                                )
                            new_param = nn.Parameter(weight_tensor)
                            setattr(self.original_component, key, new_param)

        # If this component has submodules, distribute weights to them
        if self.real_components:
            from transformer_lens.weight_processing import ProcessWeights

            if verbose:
                print(f"    Has {len(self.real_components)} subcomponents, distributing weights...")

            ProcessWeights.distribute_weights_to_components(
                state_dict=weights,
                component_mapping=self.real_components,
                verbose=verbose,
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Generic forward pass for bridge components with input/output hooks."""
        original_component = self._modules.get("_original_component", None)
        if original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. Call set_original_component() first."
            )
        target_dtype = None
        try:
            target_dtype = next(original_component.parameters()).dtype
        except StopIteration:
            pass
        input_arg_names = [
            "input",
            "hidden_states",
            "input_ids",
            "query_input",
            "x",
            "inputs_embeds",
        ]
        input_found = False
        for name in input_arg_names:
            if name in kwargs:
                hooked = self.hook_in(kwargs[name])
                if (
                    target_dtype is not None
                    and isinstance(hooked, torch.Tensor)
                    and hooked.is_floating_point()
                ):
                    hooked = hooked.to(dtype=target_dtype)
                kwargs[name] = hooked
                input_found = True
                break
        if not input_found and len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked_input = self.hook_in(args[0])
            if target_dtype is not None and hooked_input.is_floating_point():
                hooked_input = hooked_input.to(dtype=target_dtype)
            args = (hooked_input,) + args[1:]
            input_found = True
        output = original_component(*args, **kwargs)
        if isinstance(output, tuple):
            hooked_first = self.hook_out(output[0])
            output = (hooked_first,) + output[1:]
        else:
            output = self.hook_out(output)
        return output

    def __getattr__(self, name: str) -> Any:
        modules = object.__getattribute__(self, "__dict__").get("_modules")
        if modules is not None and name in modules:
            return modules[name]
        if name == "original_component":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        submodules = object.__getattribute__(self, "__dict__").get("submodules")
        if submodules is not None and name in submodules:
            # Don't return submodule here - it should be accessed via _modules after add_module()
            # Raising AttributeError allows PyTorch's add_module() to work correctly
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if modules is not None:
            original_component = modules.get("_original_component")
            if original_component is not None:
                try:
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
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, with passthrough to original component for compatibility."""
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
        class_attr = getattr(type(self), name, None)
        if class_attr is not None and isinstance(class_attr, property):
            if class_attr.fset is not None:
                super().__setattr__(name, value)
                return
        if hasattr(self, "_modules") and "_original_component" in self._modules:
            original_component = self._modules["_original_component"]
            if hasattr(original_component, name):
                try:
                    setattr(original_component, name, value)
                    return
                except AttributeError:
                    pass
        super().__setattr__(name, value)
