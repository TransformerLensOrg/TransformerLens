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

# Layer-type variant submodule names. Tuple for deterministic iteration order.
# Extend here when adding new hybrid variant types.
VARIANT_SUBMODULE_NAMES: tuple[str, ...] = ("attn", "linear_attn", "mamba", "mixer", "ssm")
_VARIANT_SUBMODULE_SET: frozenset[str] = frozenset(VARIANT_SUBMODULE_NAMES)

# Infrastructure modules excluded from submodule introspection.
_BLOCK_INTERNAL_MODULES: frozenset[str] = frozenset({"hook_in", "hook_out", "_original_component"})

# Norm-module prefixes excluded from layer_types() labels.
_NORM_PREFIXES: tuple[str, ...] = ("ln", "layer_norm", "norm", "rms")


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
        # ln1_post/ln2_post redirect attn_out/mlp_out to match HookedTransformer's
        # placement (hook fires after the post-norm, not before).
        auto_overrides = {}
        if submodules is not None:
            if "ln1_post" in submodules:
                auto_overrides["hook_attn_out"] = "ln1_post.hook_out"
            if "ln2_post" in submodules:
                auto_overrides["hook_mlp_out"] = "ln2_post.hook_out"
        merged_overrides = {**auto_overrides, **(hook_alias_overrides or {})}

        # Guard against the C15 bug class: sequential transformer block (attn +
        # mlp) with no ln2 would silently point hook_resid_mid at the wrong
        # tensor. Use ParallelBlockBridge for parallel-residual architectures.
        # Skip the check on generic-container / attn-only uses (no mlp).
        has_attn_like = submodules is not None and any(
            k in submodules for k in _VARIANT_SUBMODULE_SET
        )
        has_mlp = submodules is not None and "mlp" in submodules
        has_ln2 = submodules is not None and "ln2" in submodules
        if has_attn_like and has_mlp and not has_ln2 and type(self) is BlockBridge:
            raise ValueError(
                f"BlockBridge at '{name}': 'ln2' submodule not declared. "
                f"Either declare ln2, or use ParallelBlockBridge for a "
                f"parallel-residual architecture."
            )

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


class ParallelBlockBridge(BlockBridge):
    """Block where attn and MLP both read the pre-attention residual.

    For GPT-J, NeoX, Pythia, Phi, Cohere, CodeGen, and some Falcon variants,
    output = resid_pre + attn_out + mlp_out — no distinct post-attention
    residual exists. Matches legacy HookedTransformer which omits hook_resid_mid
    when ``cfg.parallel_attn_mlp=True``. Type-level distinction means a reader
    of the adapter sees ``ParallelBlockBridge`` and knows the hook is absent.
    """

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        hook_alias_overrides: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name,
            config=config,
            submodules=submodules,
            hook_alias_overrides=hook_alias_overrides,
        )
        # Ensure instance-level copy before mutating; base may have left the
        # class-level dict shared when no overrides were passed.
        if self.hook_aliases is BlockBridge.hook_aliases:
            self.hook_aliases = dict(self.hook_aliases)
        self.hook_aliases.pop("hook_resid_mid", None)
