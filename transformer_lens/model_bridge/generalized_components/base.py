"""Base class for generalized transformer components."""
from __future__ import annotations

import contextlib
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from accelerate.utils import align_module_device

from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)
from transformer_lens.hook_points import HookPoint


def align_offloaded_subtree(module: nn.Module) -> contextlib.ExitStack:
    """Materialize every Accelerate-offloaded descendant of ``module`` for the
    caller's duration (an ``ExitStack`` of ``align_module_device`` contexts).

    Accelerate attaches offload hooks at leaf level - whichever submodule
    directly owns the Parameter (e.g. ``c_attn``, ``c_proj``) - not on
    container modules like an attention block as a whole. ``align_module_device``
    on a container alone is therefore a no-op even though its descendants are
    offloaded. Walking every descendant and entering each one's
    ``align_module_device`` (a cheap no-op for any module that has no hook of
    its own) covers both a leaf ``original_component`` and a multi-level
    container uniformly, without needing to know in advance which specific
    descendant a given architecture adapter's code actually reads from.
    """
    stack = contextlib.ExitStack()
    for submodule in module.modules():
        stack.enter_context(align_module_device(submodule))
    return stack


class CloneOutputUnderGradMixin(nn.Module):
    """Clone the forward output so HF's in-place mutation cannot corrupt it.

    Under grad, autograd forbids in-place writes to backward-hook views; under
    no_grad, cached hook_out tensors alias the storage HF then rewrites.
    Mix in ahead of a bridge class: ``class X(CloneOutputUnderGradMixin, LinearBridge)``.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = super().forward(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            out = out.clone()
        return out


# Bumped whenever any component rebinds its hook_aliases (MoEBridge's dense/
# sparse dispatch, OlmoHybrid's per-layer selection). Alias caches key on it:
# a rebind can leave the hook REGISTRY unchanged while changing what the
# aliases point at, so size-based cache keys cannot see it.
_ALIAS_GENERATION = 0


def alias_generation() -> int:
    """Current global alias-rebind generation."""
    return _ALIAS_GENERATION


def _bump_alias_generation() -> None:
    global _ALIAS_GENERATION
    _ALIAS_GENERATION += 1


class GeneralizedComponent(nn.Module):
    """Base class for generalized transformer components.

    This class provides a standardized interface for transformer components
    and handles hook registration and execution.
    """

    is_list_item: bool = False
    hook_out_is_single_residual_stream: bool = False
    compatibility_mode: bool = False
    disable_warnings: bool = False
    hook_aliases: Dict[str, Union[str, List[str]]] = {}

    # Projection-hook protocol between container bridges (MLPBridge) and the
    # projection bridges they wrap (LinearBridge / Conv1DBridge): the wrapper
    # records that its hook_out fired so the container only re-fires it as a
    # bypass fallback, and the container suppresses the wrapper's next hook_in
    # after pre-firing it itself. See MLPBridge.forward.
    _fired_hook_out: bool = False
    _suppress_next_hook_in: bool = False
    property_aliases: Dict[str, str] = {}

    def __init__(
        self,
        name: Optional[str],
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, "GeneralizedComponent"]] = None,
        conversion_rule: Optional[BaseTensorConversion] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
        optional: bool = False,
    ):
        """Initialize the generalized component.

        Args:
            name: The name of this component (None if component has no container in remote model)
            config: Optional configuration object for the component
            submodules: Dictionary of GeneralizedComponent submodules to register
            conversion_rule: Optional conversion rule for this component's hooks
            hook_alias_overrides: Optional dictionary to override default hook aliases.
                For example, {"hook_attn_out": "ln1_post.hook_out"} will make hook_attn_out
                point to ln1_post.hook_out instead of the default value in self.hook_aliases.
            optional: If True, setup skips this subtree when absent (hybrid architectures).
        """
        super().__init__()
        self.name = name
        self.config = config
        self.submodules = submodules or {}
        self.conversion_rule = conversion_rule
        self.optional = optional
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

        # Copy class-level hook_aliases and apply any overrides
        if hook_alias_overrides is not None:
            self.hook_aliases = self.__class__.hook_aliases.copy()
            self.hook_aliases.update(hook_alias_overrides)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run forward(), materializing the wrapped component's params first if offloaded.

        Bridge components read the wrapped module's raw parameters directly
        (self.weight / self.original_component.bias / etc. via __getattr__ or
        direct attribute access) rather than exclusively calling the wrapped
        module's own forward(). Under an Accelerate CPU/disk device_map,
        Accelerate only swaps a meta placeholder for the real, materialized
        tensor around the wrapped module's OWN forward() call (its pre/post
        forward hooks) - a raw attribute read outside that window silently
        sees the meta placeholder instead, with no error. align_module_device
        wraps this call in the same pre/post-forward hook Accelerate would
        have run, so every read during this call - whichever way the code
        reaches it - gets real data. It's a no-op (bare yield) when the
        wrapped module has no offload hook, which covers the common case of
        no device_map, a single device, or a multi-GPU split with no CPU/disk
        offload involved.

        Deliberately just original_component, not align_offloaded_subtree's
        whole-subtree walk: Accelerate hooks the leaf modules that actually own
        Parameters (e.g. NormalizationBridge/LinearBridge wrap one directly), so
        materializing exactly that leaf for exactly its own call keeps the same
        one-leaf-at-a-time memory footprint Accelerate's native per-module hooks
        would give a plain (non-bridge) forward pass. A component whose
        original_component is a container of several separately-hooked leaves
        (e.g. an attention module wrapping distinct q/k/v/o projections) doesn't
        need this to also materialize here - each of ITS own sub-bridges calls
        into its own leaf the same way. align_offloaded_subtree is for the
        narrower case of code that reads a specific descendant's raw params
        directly during setup, without going through that descendant's own
        bridge __call__ at all (see JointQKVAttentionBridge.set_original_component).
        """
        original_component = self._modules.get("_original_component")
        if original_component is None:
            return super().__call__(*args, **kwargs)
        with align_module_device(original_component):
            return super().__call__(*args, **kwargs)

    def _register_hook(self, name: str, hook: HookPoint) -> None:
        """Register a hook in the component's hook registry."""
        hook.name = name
        self._hook_registry[name] = hook

    def _register_aliases(self) -> None:
        """Register aliases from class-level dictionaries.

        Called unconditionally at bridge init (see bridge.py); compatibility mode
        additionally re-registers after weight processing.
        It creates actual Python attributes/properties that directly reference the target objects.

        Note: Re-registration expects to run after
        weight processing is complete to ensure property aliases point to processed weights.
        """
        if self.hook_aliases:
            self._hook_alias_registry.update(self.hook_aliases)
        if self.property_aliases:
            self._property_alias_registry.update(self.property_aliases)
        for alias_name, target_path in self._hook_alias_registry.items():
            resolved = False
            if isinstance(target_path, list):
                for single_target in target_path:
                    try:
                        target_obj = self
                        for part in single_target.split("."):
                            target_obj = getattr(target_obj, part)
                        object.__setattr__(self, alias_name, target_obj)
                        resolved = True
                        break
                    except AttributeError:
                        continue
            else:
                try:
                    target_obj = self
                    for part in target_path.split("."):
                        target_obj = getattr(target_obj, part)
                    object.__setattr__(self, alias_name, target_obj)
                    resolved = True
                except AttributeError:
                    pass
            if not resolved:
                # Surface drops instead of silently swallowing — some aliases are
                # legitimately conditional on optional submodules, but an author
                # needs to see which ones dropped at bridge-init.
                warnings.warn(
                    f"Hook alias '{alias_name}' -> '{target_path}' on "
                    f"{type(self).__name__}(name={getattr(self, 'name', None)!r}) "
                    f"did not resolve; this hook will not be accessible.",
                    stacklevel=2,
                )
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
        # An opaque wrapper (created with config=None) shadows the wrapped
        # module's own config. HF forwards sometimes read a submodule's config
        # directly (e.g. Qwen2Audio's forward does create_bidirectional_mask(
        # config=self.audio_tower.config)), so inherit the real config to avoid
        # exposing None. Components given an explicit config keep it.
        if self.config is None:
            self.config = getattr(original_component, "config", None)

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
            self.hook_in.remove_hooks(dir="both")
            self.hook_out.remove_hooks(dir="both")
        elif hook_name == "output":
            self.hook_out.remove_hooks(dir="both")
        elif hook_name == "input":
            self.hook_in.remove_hooks(dir="both")
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
        # Skip non-fp params: quantized weights (bnb uint8/int8, GPTQ/AWQ int32,
        # HQQ, torchao) are stored in integer dtypes and dequantized internally
        # during matmul. The compute dtype must come from a fp parameter; casting
        # fp inputs to an integer storage dtype destroys precision.
        target_dtype = None
        for p in original_component.parameters():
            if not p.dtype.is_floating_point:
                continue
            target_dtype = p.dtype
            break
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
        elif not isinstance(output, torch.Tensor) and isinstance(
            getattr(output, "last_hidden_state", None), torch.Tensor
        ):
            # ModelOutput-returning components (e.g. vision/audio towers).
            output.last_hidden_state = self.hook_out(output.last_hidden_state)
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
        if name == "hook_aliases":
            # Any alias rebind invalidates alias caches downstream.
            _bump_alias_generation()
        if name.startswith("_") or name in [
            "name",
            "config",
            "submodules",
            "conversion_rule",
            "compatibility_mode",
            "disable_warnings",
            "optional",
            # Components rebind these per layer at bind time (MoEBridge's
            # dense/sparse dispatch). Without the carve-out the assignment is
            # forwarded to the wrapped HF module whenever it happens to expose
            # the attribute — the rebind then silently vanishes.
            "hook_aliases",
            "property_aliases",
            # train()/eval() set self.training; redirecting it to the original
            # component leaves the wrapper stuck in training mode (dropout at
            # inference). Recursion still reaches the original via _modules.
            "training",
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
