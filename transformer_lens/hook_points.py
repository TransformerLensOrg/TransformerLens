from __future__ import annotations

"""Hook Points.

Helpers to access activations in models.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import torch
import torch.nn as nn
import torch.utils.hooks as hooks
from torch import Tensor

# Import BaseTensorConversion from the new location
from transformer_lens.conversion_utils.conversion_steps.base_tensor_conversion import (
    BaseTensorConversion,
)


@dataclass
class LensHandle:
    """Dataclass that holds information about a PyTorch hook."""

    hook: hooks.RemovableHandle
    """Reference to the Hook's Removable Handle."""

    is_permanent: bool = False
    """Indicates if the Hook is Permanent."""

    context_level: Optional[int] = None
    """Context level associated with the hooks context manager for the given hook."""

    user_hook: Optional[Callable] = None
    """The original hook callable, before ``add_hook`` wraps it."""


# Define type aliases
NamesFilter = Optional[Union[Callable[[str], bool], Sequence[str], str]]


class _ScaledGradientTensor:
    """Wrapper around gradient tensors that applies backward_scale to sum operations.

    This works around a PyTorch bug/behavior where multiplying gradient tensors
    element-wise in backward hooks gives incorrect sums.
    """

    def __init__(self, tensor: Tensor, scale: float):
        self._tensor = tensor
        self._scale = scale

    def sum(self, *args, **kwargs):
        """Override sum to apply scaling to the result, not the tensor."""
        result = self._tensor.sum(*args, **kwargs)
        if isinstance(result, Tensor) and result.numel() == 1:
            # Scalar result - apply scale
            return result * self._scale
        return result

    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped tensor."""
        return getattr(self._tensor, name)

    def __repr__(self):
        return f"ScaledGradientTensor({self._tensor}, scale={self._scale})"


@runtime_checkable
class _HookFunctionProtocol(Protocol):
    """Protocol for hook functions."""

    def __call__(self, tensor: Tensor, *, hook: "HookPoint") -> Union[Any, None]:
        ...


HookFunction = _HookFunctionProtocol  # Callable[..., _HookFunctionProtocol]

DeviceType = Optional[torch.device]
_grad_t = Union[tuple[Tensor, ...], Tensor]


class _AliasedHookPoint:
    """
    A lightweight wrapper that represents a HookPoint with an aliased name.

    This is used when a hook is registered with multiple names (e.g., in compatibility mode
    where both canonical and legacy names should trigger the hook). Instead of modifying
    the original HookPoint's name, we create this wrapper that delegates to the original
    HookPoint but presents a different name to the user's hook function.
    """

    def __init__(self, alias_name: str, target: "HookPoint"):
        """
        Create an aliased view of a HookPoint.

        Args:
            alias_name: The name to present to the hook function
            target: The original HookPoint to delegate to
        """
        self._alias_name = alias_name
        self._target = target

    @property
    def name(self) -> Optional[str]:
        """Return the alias name."""
        return self._alias_name

    @property
    def ctx(self) -> dict:
        """Delegate to the target's context."""
        return self._target.ctx

    @property
    def hook_conversion(self):
        """Delegate to the target's hook conversion."""
        return self._target.hook_conversion

    def layer(self) -> int:
        """
        Extract layer index from the alias name.

        Returns the layer index for hook names like 'blocks.0.attn.hook_pattern' -> 0
        """
        if self._alias_name is None:
            raise ValueError("Name cannot be None")
        split_name = self._alias_name.split(".")
        return int(split_name[1])


class HookPoint(nn.Module):
    """
    A helper class to access intermediate activations in a PyTorch model (inspired by Garcon).

    HookPoint is a dummy module that acts as an identity function by default. By wrapping any
    intermediate activation in a HookPoint, it provides a convenient way to add PyTorch hooks.
    """

    def __init__(self):
        super().__init__()
        self.fwd_hooks: list[LensHandle] = []
        self.bwd_hooks: list[LensHandle] = []
        self.ctx = {}

        # A variable giving the hook's name (from the perspective of the root
        # module) - this is set by the root module at setup.
        self.name: Optional[str] = None

        # Hook conversion for input and output transformations
        self.hook_conversion: Optional[BaseTensorConversion] = None

        # Backward gradient scale factor (for compatibility between architectures)
        # This scales the SUM of gradients, not element-wise (to avoid PyTorch bugs)
        self.backward_scale: float = 1.0

    def __repr__(self) -> str:
        bits = [f"name={self.name!r}"] if self.name is not None else []
        if self.fwd_hooks:
            bits.append(f"{len(self.fwd_hooks)} fwd")
        if self.bwd_hooks:
            bits.append(f"{len(self.bwd_hooks)} bwd")
        return f"HookPoint({', '.join(bits)})" if bits else "HookPoint()"

    def add_perma_hook(self, hook: HookFunction, dir: Literal["fwd", "bwd"] = "fwd") -> None:
        self.add_hook(hook, dir=dir, is_permanent=True)

    def add_hook(
        self,
        hook: HookFunction,
        dir: Literal["fwd", "bwd"] = "fwd",
        is_permanent: bool = False,
        level: Optional[int] = None,
        prepend: bool = False,
        alias_names: Optional[list[str]] = None,
    ) -> None:
        """
        Hook format is fn(activation, hook_name)
        Change it into PyTorch hook format (this includes input and output,
        which are the same for a HookPoint)
        If prepend is True, add this hook before all other hooks
        If alias_names is provided, the hook will be called once for each alias name,
        receiving a temporary HookPoint-like object with that name instead of self
        (useful for compatibility mode aliases)
        """

        def full_hook(
            module: torch.nn.Module,
            module_input: Any,
            module_output: Any,
        ):
            if (
                dir == "bwd"
            ):  # For a backwards hook, module_output is a tuple of (grad,) - I don't know why.
                module_output = module_output[0]

                # Apply backward scaling if needed (wrap tensor to scale sum operations)
                if self.backward_scale != 1.0:
                    module_output = _ScaledGradientTensor(module_output, self.backward_scale)

            # Apply input conversion if hook_conversion exists
            if self.hook_conversion is not None:
                module_output = self.hook_conversion.convert(module_output)

            # Apply the hook for each name (or just once with canonical name)
            if alias_names is not None:
                # Call the hook once for each alias name
                # Create a simple wrapper that acts like a HookPoint but with a different name
                hook_result = None
                for alias_name in alias_names:
                    # Create a view of this HookPoint with the alias name
                    hook_with_alias = _AliasedHookPoint(alias_name, self)
                    # Apply the hook
                    hook_result = hook(module_output, hook=hook_with_alias)  # type: ignore[arg-type]

                    # If the hook modified the output, use that for subsequent calls
                    if hook_result is not None:
                        module_output = hook_result
            else:
                # Call the hook once with the canonical name (self)
                hook_result = hook(module_output, hook=self)

            # Apply output reversion if hook_conversion exists and hook returned a value
            if hook_result is not None and self.hook_conversion is not None:
                hook_result = self.hook_conversion.revert(hook_result)

            # For backward hooks, PyTorch expects the return to be a tuple of (grad,)
            if dir == "bwd" and hook_result is not None:
                return (
                    hook_result
                    if isinstance(hook_result, tuple) and len(hook_result) == 1
                    else (hook_result,)
                )

            return hook_result

        # annotate the `full_hook` with the string representation of the `hook` function
        if isinstance(hook, partial):
            # partial.__repr__() can be extremely slow if arguments contain large objects, which
            # is common when caching tensors.
            full_hook.__name__ = f"partial({hook.func.__repr__()},...)"
        else:
            full_hook.__name__ = hook.__repr__()

        if dir == "fwd":
            pt_handle = self.register_forward_hook(full_hook, prepend=prepend)
            visible_hooks = self.fwd_hooks
        elif dir == "bwd":
            # Wrap full_hook's bare Tensor return in tuple for PyTorch's backward API
            def _bwd_hook_wrapper(
                module: torch.nn.Module,
                grad_input: Any,
                grad_output: Any,
            ):
                result = full_hook(module, grad_input, grad_output)
                if result is None:
                    return None
                if isinstance(result, tuple):
                    return result
                return (result,)

            if isinstance(hook, partial):
                _bwd_hook_wrapper.__name__ = f"partial({hook.func.__repr__()},...)"
            else:
                _bwd_hook_wrapper.__name__ = hook.__repr__()
            pt_handle = self.register_full_backward_hook(_bwd_hook_wrapper, prepend=prepend)
            visible_hooks = self.bwd_hooks
        else:
            raise ValueError(f"Invalid direction {dir}")

        handle = LensHandle(pt_handle, is_permanent, level, user_hook=hook)

        if prepend:
            # we could just pass this as an argument in PyTorch 2.0, but for now we manually do this...
            visible_hooks.insert(0, handle)

        else:
            visible_hooks.append(handle)

    def has_hooks(
        self,
        dir: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = True,
        level: Optional[int] = None,
    ) -> bool:
        """Check if this HookPoint has any active hooks.

        Args:
            dir: Direction of hooks to check ("fwd", "bwd", or "both")
            including_permanent: Whether to include permanent hooks in the check
            level: Only check hooks at this context level (None for all levels)

        Returns:
            True if any matching hooks are found, False otherwise
        """

        def _has_hooks_in_direction(handles: list[LensHandle]) -> bool:
            for handle in handles:
                # Check if this hook matches our criteria
                if not including_permanent and handle.is_permanent:
                    continue
                if level is not None and handle.context_level != level:
                    continue
                return True
            return False

        if dir == "fwd":
            return _has_hooks_in_direction(self.fwd_hooks)
        elif dir == "bwd":
            return _has_hooks_in_direction(self.bwd_hooks)
        elif dir == "both":
            return _has_hooks_in_direction(self.fwd_hooks) or _has_hooks_in_direction(
                self.bwd_hooks
            )
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(
        self,
        dir: Literal["fwd", "bwd", "both"] = "fwd",
        including_permanent: bool = False,
        level: Optional[int] = None,
    ) -> None:
        def _remove_hooks(handles: list[LensHandle]) -> list[LensHandle]:
            output_handles = []
            for handle in handles:
                if including_permanent:
                    handle.hook.remove()
                elif (not handle.is_permanent) and (level is None or handle.context_level == level):
                    handle.hook.remove()
                else:
                    output_handles.append(handle)
            return output_handles

        if dir == "fwd" or dir == "both":
            self.fwd_hooks = _remove_hooks(self.fwd_hooks)
        if dir == "bwd" or dir == "both":
            self.bwd_hooks = _remove_hooks(self.bwd_hooks)
        if dir not in ["fwd", "bwd", "both"]:
            raise ValueError(f"Invalid direction {dir}")

    def clear_context(self):
        del self.ctx
        self.ctx = {}

    def enable_reshape(
        self,
        hook_conversion: Optional[BaseTensorConversion] = None,
    ) -> None:
        """
        Enable reshape functionality for this hook point using a BaseTensorConversion.

        Args:
            hook_conversion: BaseTensorConversion instance to handle input/output transformations.
                           The convert() method will be used for input transformation,
                           and the revert() method will be used for output transformation.
        """
        self.hook_conversion = hook_conversion

    def forward(self, x: Tensor) -> Tensor:
        return x

    def layer(self):
        # Returns the layer index if the name has the form 'blocks.{layer}.{...}'
        # Helper function that's mainly useful on HookedTransformer
        # If it doesn't have this form, raises an error -
        if self.name is None:
            raise ValueError("Name cannot be None")
        split_name = self.name.split(".")
        return int(split_name[1])


# %%
class HookIntrospectionMixin:
    """``list_hooks()`` mixin for any class exposing a ``hook_dict``.

    Accessed via ``getattr`` so subclasses can provide ``hook_dict`` as either
    an instance attribute (``HookedRootModule``) or a ``@property`` (``TransformerBridge``).
    """

    def list_hooks(
        self,
        name_filter: NamesFilter = None,
        dir: Literal["fwd", "bwd", "both"] = "both",
        including_permanent: bool = True,
    ) -> dict[str, list[LensHandle]]:
        """Return attached hooks grouped by HookPoint name; empty HookPoints are omitted.

        Args:
            name_filter: A hook name, list of names, or predicate. ``None`` matches all.
            dir: Restrict to forward, backward, or both directions.
            including_permanent: If False, drop permanent hooks from the result.
        """
        if name_filter is None:
            matches: Callable[[str], bool] = lambda _: True
        elif callable(name_filter):
            matches = name_filter
        elif isinstance(name_filter, str):
            target = name_filter
            matches = lambda n: n == target
        else:
            allowed = set(name_filter)
            matches = lambda n: n in allowed

        out: dict[str, list[LensHandle]] = {}
        hook_dict: dict[str, HookPoint] = getattr(self, "hook_dict")
        for name, hp in hook_dict.items():
            if not matches(name):
                continue
            handles: list[LensHandle] = []
            if dir in ("fwd", "both"):
                handles.extend(hp.fwd_hooks)
            if dir in ("bwd", "both"):
                handles.extend(hp.bwd_hooks)
            if not including_permanent:
                handles = [h for h in handles if not h.is_permanent]
            if handles:
                out[name] = handles
        return out


# HookedRootModule moved to transformer_lens.HookedRootModule (3.0). Import it from
# its dedicated module — there is no re-export here.


# %%
