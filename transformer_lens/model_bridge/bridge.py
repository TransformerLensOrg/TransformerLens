"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
from torch import nn

from transformer_lens import utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint


class StopAtLayerException(Exception):
    """Exception to stop forward pass at a specific layer."""

    def __init__(self, tensor, layer_idx):
        self.tensor = tensor
        self.layer_idx = layer_idx
        self.layer_output = tensor  # Add the missing layer_output attribute
        super().__init__(f"Stopped at layer {layer_idx}")


def collect_aliases_recursive(hook_dict, prefix=""):
    """Recursively collect hook aliases from a nested hook dictionary."""
    aliases = {}
    for key, value in hook_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            aliases.update(collect_aliases_recursive(value, full_key))
        elif hasattr(value, "name"):
            aliases[full_key] = value.name
    return aliases


from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_setup import set_original_components
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.get_params_util import get_bridge_params
from transformer_lens.model_bridge.hook_point_wrapper import HookPointWrapper
from transformer_lens.model_bridge.types import ComponentMapping
from transformer_lens.utilities.aliases import resolve_alias

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache


class TransformerBridge(nn.Module):
    """Bridge between HuggingFace and HookedTransformer models.

    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the HookedTransformer and HuggingFace model structures.
    """

    # Top-level hook aliases for legacy TransformerLens names
    # Placing these on the main bridge ensures aliases like 'hook_embed' are available
    hook_aliases: Dict[str, Union[str, List[str]]] = {
        "hook_embed": "embed.hook_out",
        # rotary style models use rotary_emb.hook_out, but gpt2-style models use pos_embed.hook_out
        "hook_pos_embed": ["pos_embed.hook_out", "rotary_emb.hook_out"],
        "hook_unembed": "unembed.hook_out",
    }

    def __init__(
        self,
        model: nn.Module,
        adapter: ArchitectureAdapter,
        tokenizer: Any,
    ):
        """Initialize the bridge.

        Args:
            model: The model to bridge (must be a PyTorch nn.Module or PreTrainedModel)
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
        """
        super().__init__()
        # Set original_model directly in __dict__ to avoid any property issues
        self.__dict__["original_model"] = model
        self.adapter = adapter
        self.cfg = adapter.cfg

        self.tokenizer = tokenizer
        self.compatibility_mode = False
        self._hook_cache = None  # Cache for hook discovery results
        self._hook_registry: Dict[
            str, HookPoint
        ] = {}  # Dynamic registry of hook names to HookPoints
        self._hook_registry_initialized = False  # Track if registry has been initialized

        # Add device information to config from the loaded model
        if not hasattr(self.cfg, "device") or self.cfg.device is None:
            try:
                self.cfg.device = str(next(self.original_model.parameters()).device)
            except StopIteration:
                self.cfg.device = "cpu"

        if not hasattr(adapter, "component_mapping") or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")

        # Set original components on the pre-created bridge components
        # Access original_model directly from __dict__ to avoid __getattr__ issues
        original_model = self.__dict__["original_model"]
        set_original_components(self, self.adapter, original_model)

        # Initialize hook registry after components are set up
        self._initialize_hook_registry()

        # Intiialize dictionary containing hooks that will be cached
        self._initialize_hooks_to_cache()

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to track HookPoint objects dynamically."""
        # Call parent setattr first
        super().__setattr__(name, value)

        # Check if this is a HookPoint being set
        if isinstance(value, HookPoint):
            # Set the name on the HookPoint
            value.name = name
            # Add to registry
            self._hook_registry[name] = value
        elif isinstance(value, HookPointWrapper):
            # Handle HookPointWrapper objects
            hook_in_name = f"{name}.hook_in"
            hook_out_name = f"{name}.hook_out"
            value.hook_in.name = hook_in_name
            value.hook_out.name = hook_out_name
            self._hook_registry[hook_in_name] = value.hook_in
            self._hook_registry[hook_out_name] = value.hook_out
        elif hasattr(value, "get_hooks") and callable(getattr(value, "get_hooks")):
            # This is a GeneralizedComponent being set
            # We need to register its hooks with the appropriate prefix
            component_hooks = value.get_hooks()
            for hook_name, hook in component_hooks.items():
                full_name = f"{name}.{hook_name}"
                hook.name = full_name
                self._hook_registry[full_name] = hook

    def _initialize_hook_registry(self) -> None:
        """Initialize the hook registry by scanning existing components."""
        if self._hook_registry_initialized:
            return

        # Scan existing components for hooks
        self._scan_existing_hooks(self, "")

        # Add bridge aliases if compatibility mode is enabled
        if self.compatibility_mode:
            self._add_aliases_to_hooks(self._hook_registry)

        self._hook_registry_initialized = True

    def _collect_component_aliases(self, component_mapping, prefix=""):
        """Recursively collect aliases from components."""
        aliases = {}

        # Handle dict of components (like component_mapping)
        if isinstance(component_mapping, dict):
            for name, component in component_mapping.items():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                aliases.update(self._collect_component_aliases(component, sub_prefix))
        else:
            # Handle individual component
            if hasattr(component_mapping, "hook_aliases") and component_mapping.hook_aliases:
                for alias_name, target in component_mapping.hook_aliases.items():
                    full_alias = f"{prefix}.{alias_name}" if prefix else alias_name
                    full_target = f"{prefix}.{target}" if prefix else target
                    aliases[full_alias] = full_target

            # Recursively collect from submodules
            if hasattr(component_mapping, "submodules") and component_mapping.submodules:
                for sub_name, sub_component in component_mapping.submodules.items():
                    sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
                    aliases.update(self._collect_component_aliases(sub_component, sub_prefix))

        return aliases

    def _collect_hook_aliases_from_registry(self):
        """Collect aliases based on existing hooks in the registry."""
        aliases = {}

        # Get component aliases from the adapter
        if hasattr(self.adapter, "component_mapping"):
            component_aliases = self._collect_component_aliases(self.adapter.component_mapping)

            # Apply component aliases to all existing hooks
            for hook_name in self._hook_registry.keys():
                # Check if this hook matches any component alias pattern
                for alias_pattern, target_pattern in component_aliases.items():
                    # Handle dynamic block patterns (blocks.0, blocks.1, etc.)
                    if "blocks." in target_pattern and "blocks." in hook_name:
                        # Extract the block number from the hook name
                        import re

                        block_match = re.search(r"blocks\.(\d+)", hook_name)
                        if block_match:
                            block_num = block_match.group(1)
                            # Replace generic patterns with actual block numbers
                            dynamic_alias_pattern = alias_pattern.replace(
                                "blocks.", f"blocks.{block_num}."
                            )
                            dynamic_target_pattern = target_pattern.replace(
                                "blocks.", f"blocks.{block_num}."
                            )

                            # Check if this hook name matches the target pattern
                            if hook_name.endswith(dynamic_target_pattern):
                                # Create the alias name by replacing the target with the alias
                                alias_name = hook_name.replace(
                                    dynamic_target_pattern, dynamic_alias_pattern
                                )
                                aliases[alias_name] = hook_name
                    else:
                        # Handle non-block patterns
                        if hook_name.endswith(target_pattern):
                            # Create the alias name by replacing the target with the alias
                            alias_name = hook_name.replace(target_pattern, alias_pattern)
                            aliases[alias_name] = hook_name

        return aliases

    def _add_aliases_to_hooks(self, hooks: Dict[str, HookPoint]) -> None:
        """Add aliases to hooks in place."""

        # Collect component aliases and merge with bridge aliases
        component_aliases = self._collect_hook_aliases_from_registry()

        # Merge component aliases with bridge aliases
        all_aliases = {**self.hook_aliases, **component_aliases}

        # If no aliases, do nothing
        if not all_aliases:
            return

        for alias_name, target in all_aliases.items():
            # Use the existing alias system to resolve the target hook
            # Convert to Dict[str, str] for resolve_alias if target_name is a list
            if isinstance(target, list):
                # For list targets, try each one until one works
                for single_target in target:
                    target_hook = resolve_alias(self, alias_name, {alias_name: single_target})
                    if target_hook is not None:
                        hooks[alias_name] = target_hook
                        break
                    continue
            else:
                try:
                    target_hook = resolve_alias(self, alias_name, {alias_name: target})
                    if target_hook is not None:
                        hooks[alias_name] = target_hook
                except AttributeError:
                    # Skip this alias if it can't be resolved (e.g., during initialization)
                    continue

    def _scan_existing_hooks(self, module: nn.Module, prefix: str = "") -> None:
        """Scan existing modules for hooks and add them to registry."""
        visited = set()

        def scan_module(mod: nn.Module, path: str = "") -> None:
            obj_id = id(mod)
            if obj_id in visited:
                return
            visited.add(obj_id)

            # Check if this is a GeneralizedComponent with its own hook registry
            if hasattr(mod, "get_hooks") and callable(getattr(mod, "get_hooks")):
                # Use the component's own hook registry
                component_hooks = mod.get_hooks()  # type: ignore
                if isinstance(component_hooks, dict):
                    # Type cast to help mypy understand this is a dict of hooks
                    hooks_dict = cast(Dict[str, HookPoint], component_hooks)  # type: ignore
                    for hook_name, hook in hooks_dict.items():  # type: ignore
                        full_name = f"{path}.{hook_name}" if path else hook_name
                        hook.name = full_name
                        self._hook_registry[full_name] = hook

            # Always scan attributes for additional hooks and submodules
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                if attr_name == "original_component" or attr_name == "original_model":
                    continue

                # Skip properties that might not be ready during initialization
                if attr_name in [
                    "OV",
                    "QK",
                    "W_V",
                    "W_O",
                    "W_Q",
                    "W_K",
                    "W_in",
                    "W_gate",
                    "W_out",
                    "b_V",
                    "b_O",
                    "b_Q",
                    "b_K",
                    "b_in",
                    "b_out",
                ]:
                    continue

                attr = getattr(mod, attr_name)

                name = f"{path}.{attr_name}" if path else attr_name

                if isinstance(attr, HookPoint):
                    attr.name = name
                    self._hook_registry[name] = attr
                elif isinstance(attr, HookPointWrapper):
                    hook_in_name = f"{name}.hook_in"
                    hook_out_name = f"{name}.hook_out"
                    attr.hook_in.name = hook_in_name
                    attr.hook_out.name = hook_out_name
                    self._hook_registry[hook_in_name] = attr.hook_in
                    self._hook_registry[hook_out_name] = attr.hook_out

            # Check named children
            for child_name, child_module in mod.named_children():
                if (
                    child_name == "original_component"
                    or child_name == "_original_component"
                    or child_name == "original_model"
                ):
                    continue
                child_path = f"{path}.{child_name}" if path else child_name
                scan_module(child_module, child_path)

        scan_module(module, prefix)

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        """Get all HookPoint objects in the model for compatibility with HookedTransformer."""
        hooks = self._hook_registry.copy()

        # Add aliases if compatibility mode is enabled
        if self.compatibility_mode:
            self._add_aliases_to_hooks(hooks)

        return hooks

    def _discover_hooks(self) -> dict[str, HookPoint]:
        """Get all HookPoint objects from the registry (deprecated, use hook_dict)."""
        return self._hook_registry.copy()

    def clear_hook_cache(self) -> None:
        """Clear the cached hook discovery results (deprecated, kept for compatibility)."""
        pass  # No longer needed since we don't use caching

    def clear_hook_registry(self) -> None:
        """Clear the hook registry and force re-initialization."""
        self._hook_registry.clear()
        self._hook_registry_initialized = False

    def _initialize_hooks_to_cache(self) -> None:
        """Initialize the hooks to cache when running the model with cache."""
        self.hooks_to_cache = {}

        default_cached_hooks_names = [
            "embed.hook_in",
            "embed.hook_out",
            "pos_embed.hook_in",
            "pos_embed.hook_out",
            "rotary_embed.hook_in",
            "rotary_embed.hook_out",
            "ln_final.hook_in",
            "ln_final.hook_scale",
            "ln_final.hook_normalized",
            "ln_final.hook_out",
            "unembed.hook_in",
            "unembed.hook_out",
        ]

        for block_idx in range(self.cfg.n_layers):
            default_cached_hooks_names.append(f"blocks.{block_idx}.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1.hook_scale")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1.hook_normalized")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1_post.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1_post.hook_scale")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1_post.hook_normalized")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln1_post.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.q.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.q.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.q_norm.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.q_norm.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.k.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.k.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.k_norm.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.k_norm.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.v.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.v.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.o.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.o.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_attn_scores")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_pattern")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_hidden_states")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_scale")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_normalized")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_scale")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_normalized")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.in.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.in.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.out.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.out.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.gate.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.gate.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.hook_out")

        for hook_name in default_cached_hooks_names:
            if hook_name in self._hook_registry:
                self.hooks_to_cache[hook_name] = self._hook_registry[hook_name]

    def set_hooks_to_cache(
        self, hook_names: Optional[List[str]] = None, include_all: bool = False
    ) -> None:
        """Set the hooks to cache when running the model with cache.

        You can specify hook names that were only available in the old HookedTransformer,
        but in this case you need to make sure to enable compatibility mode.

        Args:
            hook_names (Optional[List[str]]): List of hook names to cache
            include_all (bool): Whether to cache all hooks
        """
        hooks_to_cache = {}

        if self.compatibility_mode:
            aliases = collect_aliases_recursive(self.hook_dict)

        if include_all:
            self.hooks_to_cache = self.hook_dict
            return

        if hook_names is not None:
            for hook_name in hook_names:
                if hook_name in self._hook_registry:
                    hooks_to_cache[hook_name] = self._hook_registry[hook_name]
                else:
                    raise ValueError(
                        f"Hook {hook_name} does not exist. If you are using a hook name used with the old HookedTransformer, make sure to enable compatibility mode."
                    )
        else:
            raise ValueError("hook_names must be provided if include_all is False")

        self.hooks_to_cache = hooks_to_cache

    def __getattr__(self, name: str) -> Any:
        """Provide a clear error message for missing attributes."""
        # First check if the attribute is in __dict__ (direct attributes)
        if name in self.__dict__:
            return self.__dict__[name]

        # Check if this is a hook alias when compatibility mode is enabled
        if self.compatibility_mode:
            resolved_hook = resolve_alias(self, name, self.hook_aliases)
            if resolved_hook is not None:
                return resolved_hook

        # Try to get from original_model if it exists
        if "original_model" in self.__dict__ and self.__dict__["original_model"] is not None:
            try:
                name_split = name.split(".")
                if len(name_split) > 1:
                    current = getattr(self.__dict__["original_model"], name_split[0])
                    for part in name_split[1:]:
                        current = getattr(current, part)
                    return current
                else:
                    return getattr(self.__dict__["original_model"], name)
            except AttributeError:
                pass

        # If we get here, the attribute wasn't found anywhere
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _get_nested_attr(self, path: str) -> Any:
        """Get a nested attribute using dot notation."""
        obj = self
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj

    def _format_single_component(self, name: str, path: str, indent: int = 0) -> str:
        """Format a single component's string representation.

        Args:
            name: The name of the component
            path: The path to get the component
            indent: The indentation level

        Returns:
            A formatted string for the component
        """
        indent_str = "  " * indent
        try:
            comp = self.adapter.get_component(self.original_model, path)
            if hasattr(comp, "original_component"):
                if comp.original_component is None:
                    return f"{indent_str}{name}: <error: original component not set>"
                return f"{indent_str}{name}: {type(comp).__name__}({type(comp.original_component).__name__})"
            return f"{indent_str}{name}: {type(comp).__name__}"
        except Exception as e:
            return f"{indent_str}{name}: <error: {e}>"

    def _format_component_mapping(
        self, mapping: ComponentMapping, indent: int = 0, prepend: str | None = None
    ) -> list[str]:
        """Format a component mapping dictionary.

        Args:
            mapping: The component mapping dictionary
            indent: The indentation level
            prepend: Optional path to prepend to component names (e.g. "blocks.0")

        Returns:
            A list of formatted strings
        """
        lines = []
        for name, value in mapping.items():
            path = f"{prepend}.{name}" if prepend else name

            if hasattr(value, "_modules") and hasattr(value, "name"):
                # This is a bridge component instance
                lines.append(self._format_single_component(name, path, indent))

                # Check if it has submodules (like BlockBridge)
                submodules = value.submodules

                if submodules:
                    # For list items (like blocks), add .0 to the path to indicate the first item
                    subpath = f"{path}.0" if value.is_list_item else path
                    # Recursively format submodules
                    sub_lines = self._format_component_mapping(submodules, indent + 1, subpath)
                    lines.extend(sub_lines)

            else:
                # For other types, use prepend if provided
                lines.append(self._format_single_component(name, path, indent))
        return lines

    def __str__(self) -> str:
        """Get a string representation of the bridge.

        Returns:
            A string describing the bridge's components
        """
        lines = ["TransformerBridge:"]
        mapping = self.adapter.get_component_mapping()
        lines.extend(self._format_component_mapping(mapping, indent=1))
        return "\n".join(lines)

    def enable_compatibility_mode(
        self, disable_warnings: bool = False, no_processing: bool = False
    ) -> None:
        """Enable compatibility mode for the bridge.

        This sets up the bridge to work with legacy HookedTransformer components/hooks.
        It will also disable warnings about the usage of legacy components/hooks if specified.

        Args:
            disable_warnings: Whether to disable warnings about legacy components/hooks
            no_processing: Whether to disable pre-processing steps of the model (e.g. folding layer norm weights, folding value biases)
        """
        # Avoid circular import
        from transformer_lens.utilities.bridge_components import (
            apply_fn_to_all_components,
        )

        self.compatibility_mode = True

        def set_compatibility_mode(component: Any) -> None:
            """Set compatibility mode on a component."""
            component.compatibility_mode = True
            component.disable_warnings = disable_warnings

        apply_fn_to_all_components(self, set_compatibility_mode)

        # Re-initialize the hook registry to include aliases from components
        self.clear_hook_registry()
        self._initialize_hook_registry()

        if not no_processing:
            # Apply weight processing using the centralized ProcessWeights class
            self.process_weights(
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,  # Keep unfactored format to match HuggingFace
            )

    def process_weights(
        self,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Apply weight processing transformations using the centralized ProcessWeights class.

        This method extracts weights from the original HuggingFace model and applies weight processing
        using the centralized ProcessWeights class with the architecture adapter to handle parameter
        name translation from TransformerLens format to HuggingFace format.
        """
        # import torch
        # import torch.nn as nn

        from transformer_lens.weight_processing import ProcessWeights

        # # Step 1: Extract HuggingFace weights from original model
        hf_state_dict = self._extract_hf_weights()

        # # Step 2: Apply centralized weight processing with architecture adapter
        # # The adapter will translate TransformerLens parameter names to HuggingFace parameter names
        processed_hf_state_dict = ProcessWeights.process_weights(
            hf_state_dict,
            self.cfg,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            adapter=self.adapter,
        )

        # # Step 3: Replace LayerNorm components with LayerNormPre-like operations if fold_ln is True
        # This is equivalent to what HookedTransformer does when it replaces LayerNorm with LayerNormPre
        if fold_ln:
            self._replace_layer_norm_with_identity(self.original_model)

        # # Step 4: Load processed weights into the original model using the bridge's load_state_dict method
        # This handles the key mapping between clean keys and _original_component keys
        # Use strict=False because weight processing may remove some keys (e.g., individual Q,K,V -> combined QKV)
        self.load_state_dict(processed_hf_state_dict, strict=False, assign=True)

    def _extract_hf_weights(self):
        """Extract weights from the original HuggingFace model."""
        # Use the bridge's clean state_dict method which automatically filters out _original_component
        hf_state_dict = self.state_dict()

        # Remove separate Q, K, V weights if combined QKV weights exist
        # This prevents the adapter from processing the same combined weight multiple times
        for layer_idx in range(self.cfg.n_layers):
            combined_qkv_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
            combined_qkv_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"

            if combined_qkv_key in hf_state_dict:
                # Remove separate Q, K, V weights since we have combined QKV
                separate_keys_to_remove = [
                    f"transformer.h.{layer_idx}.attn.q.weight",
                    f"transformer.h.{layer_idx}.attn.q.bias",
                    f"transformer.h.{layer_idx}.attn.k.weight",
                    f"transformer.h.{layer_idx}.attn.k.bias",
                    f"transformer.h.{layer_idx}.attn.v.weight",
                    f"transformer.h.{layer_idx}.attn.v.bias",
                ]

                for key_to_remove in separate_keys_to_remove:
                    if key_to_remove in hf_state_dict:
                        del hf_state_dict[key_to_remove]

        return hf_state_dict

    def _add_identity_layer_norm_params(self, processed_hf_state_dict):
        """Add identity LayerNorm parameters after folding.

        After folding LayerNorm into other layers, HuggingFace models still expect
        LayerNorm parameters to exist. Set them to identity (weight=1, bias=0).
        """
        import torch

        for layer_idx in range(self.cfg.n_layers):
            ln1_weight_key = f"transformer.h.{layer_idx}.ln_1.weight"
            ln1_bias_key = f"transformer.h.{layer_idx}.ln_1.bias"
            ln2_weight_key = f"transformer.h.{layer_idx}.ln_2.weight"
            ln2_bias_key = f"transformer.h.{layer_idx}.ln_2.bias"

            # Always add identity LayerNorm parameters (they were deleted by ProcessWeights.fold_layer_norm)
            processed_hf_state_dict[ln1_weight_key] = torch.ones(self.cfg.d_model)
            processed_hf_state_dict[ln1_bias_key] = torch.zeros(self.cfg.d_model)
            processed_hf_state_dict[ln2_weight_key] = torch.ones(self.cfg.d_model)
            processed_hf_state_dict[ln2_bias_key] = torch.zeros(self.cfg.d_model)

        ln_final_weight_key = "transformer.ln_f.weight"
        ln_final_bias_key = "transformer.ln_f.bias"

        # Always add identity final LayerNorm parameters (they were deleted by ProcessWeights.fold_layer_norm)
        processed_hf_state_dict[ln_final_weight_key] = torch.ones(self.cfg.d_model)
        processed_hf_state_dict[ln_final_bias_key] = torch.zeros(self.cfg.d_model)

    def _replace_layer_norm_with_identity(self, model):
        """Replace LayerNorm components with LayerNormPre-like operations to maintain mathematical equivalence.

        After folding LayerNorm into other layers, we need to replace the LayerNorm components
        with operations that only apply normalization (centering and scaling) without learnable parameters.
        This is equivalent to what HookedTransformer does when it replaces LayerNorm with LayerNormPre components.
        """
        import torch.nn as nn

        # Import the proper LayerNormPre from HookedTransformer
        from transformer_lens.components.layer_norm_pre import LayerNormPre
        from transformer_lens.config.HookedTransformerConfig import (
            HookedTransformerConfig,
        )

        # Create a compatible HookedTransformerConfig from the bridge config
        hooked_config = HookedTransformerConfig(
            d_model=self.cfg.d_model,
            d_vocab=self.cfg.d_vocab,
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
            d_head=self.cfg.d_head,
            d_mlp=self.cfg.d_mlp,
            eps=self.cfg.eps,
            n_ctx=1024,  # Default context length
            device=self.cfg.device,
            act_fn="relu",  # GPT-2 uses ReLU activation
            attn_only=getattr(self.cfg, 'attn_only', False),
        )

        # Replace LayerNorm components in each layer
        for layer_idx in range(self.cfg.n_layers):
            # Replace ln_1 and ln_2 with LayerNormPre using proper constructor
            model.transformer.h[layer_idx].ln_1 = LayerNormPre(hooked_config)
            model.transformer.h[layer_idx].ln_2 = LayerNormPre(hooked_config)

        # Replace final LayerNorm with LayerNormPre
        model.transformer.ln_f = LayerNormPre(hooked_config)

    def _load_processed_weights(self, processed_state_dict):
        """Load processed weights back into the TransformerBridge.

        Args:
            processed_state_dict: Dictionary of processed weights in TransformerLens format
        """
        # Load embedding weights
        if "embed.W_E" in processed_state_dict:
            self.embed.weight.data = processed_state_dict["embed.W_E"]
        if "pos_embed.W_pos" in processed_state_dict:
            self.pos_embed.weight.data = processed_state_dict["pos_embed.W_pos"]

        # Load layer weights
        for layer_idx in range(self.cfg.n_layers):
            if layer_idx >= len(self.blocks):
                continue

            block = self.blocks[layer_idx]

            # Load attention weights
            if f"blocks.{layer_idx}.attn.W_Q" in processed_state_dict:
                # The processed weights are in [n_heads, d_model, d_head] format
                # Need to reshape back to the bridge's expected format
                w_q = processed_state_dict[f"blocks.{layer_idx}.attn.W_Q"]
                w_k = processed_state_dict[f"blocks.{layer_idx}.attn.W_K"]
                w_v = processed_state_dict[f"blocks.{layer_idx}.attn.W_V"]
                w_o = processed_state_dict[f"blocks.{layer_idx}.attn.W_O"]

                # Reshape from TL format to bridge format and load
                if hasattr(block.attn, "q") and hasattr(block.attn.q, "weight"):
                    # For separate Q/K/V components, reshape from [n_heads, d_model, d_head] to [d_model, d_model]
                    if w_q.dim() == 3:  # [n_heads, d_model, d_head]
                        block.attn.q.weight.data = w_q.reshape(-1, w_q.shape[1])
                        block.attn.k.weight.data = w_k.reshape(-1, w_k.shape[1])
                        block.attn.v.weight.data = w_v.reshape(-1, w_v.shape[1])
                    else:
                        block.attn.q.weight.data = w_q
                        block.attn.k.weight.data = w_k
                        block.attn.v.weight.data = w_v

                if hasattr(block.attn, "o") and hasattr(block.attn.o, "weight"):
                    # For output weights, reshape from [n_heads, d_head, d_model] to [d_model, d_model]
                    if w_o.dim() == 3:  # [n_heads, d_head, d_model]
                        block.attn.o.weight.data = w_o.reshape(w_o.shape[1] * w_o.shape[0], -1)
                    else:
                        block.attn.o.weight.data = w_o

            # Load attention biases if they exist
            for bias_name in ["b_Q", "b_K", "b_V", "b_O"]:
                param_key = f"blocks.{layer_idx}.attn.{bias_name}"
                if param_key in processed_state_dict:
                    bridge_attr = bias_name[2:].lower()  # b_Q -> q, b_K -> k, etc.
                    if bridge_attr == "o":
                        bridge_attr = "o"
                    if hasattr(block.attn, bridge_attr):
                        attn_component = getattr(block.attn, bridge_attr)
                        if hasattr(attn_component, "bias") and attn_component.bias is not None:
                            bias_data = processed_state_dict[param_key]
                            if bias_data.dim() > 1:  # [n_heads, d_head] -> [n_heads * d_head]
                                bias_data = bias_data.reshape(-1)
                            attn_component.bias.data = bias_data

            # Load MLP weights
            if hasattr(block, "mlp"):
                mlp_weight_keys = ["W_in", "W_out", "W_gate"]
                mlp_bias_keys = ["b_in", "b_out", "b_gate"]

                for weight_key in mlp_weight_keys:
                    param_key = f"blocks.{layer_idx}.mlp.{weight_key}"
                    if param_key in processed_state_dict:
                        bridge_attr = weight_key[2:].lower()  # W_in -> in, W_out -> out
                        if bridge_attr == "in":
                            bridge_attr = "input"  # GPT-2 uses 'input' instead of 'in'
                        if hasattr(block.mlp, bridge_attr):
                            mlp_component = getattr(block.mlp, bridge_attr)
                            if hasattr(mlp_component, "weight"):
                                mlp_component.weight.data = processed_state_dict[param_key]

                for bias_key in mlp_bias_keys:
                    param_key = f"blocks.{layer_idx}.mlp.{bias_key}"
                    if param_key in processed_state_dict:
                        bridge_attr = bias_key[2:].lower()  # b_in -> in, b_out -> out
                        if bridge_attr == "in":
                            bridge_attr = "input"  # GPT-2 uses 'input' instead of 'in'
                        if hasattr(block.mlp, bridge_attr):
                            mlp_component = getattr(block.mlp, bridge_attr)
                            if hasattr(mlp_component, "bias") and mlp_component.bias is not None:
                                mlp_component.bias.data = processed_state_dict[param_key]

            # Load LayerNorm weights
            for ln_name in ["ln1", "ln2"]:
                for param_type in ["w", "b"]:
                    param_key = f"blocks.{layer_idx}.{ln_name}.{param_type}"
                    if param_key in processed_state_dict:
                        if hasattr(block, ln_name):
                            ln_component = getattr(block, ln_name)
                            attr_name = "weight" if param_type == "w" else "bias"
                            if hasattr(ln_component, attr_name):
                                param_tensor = getattr(ln_component, attr_name)
                                if param_tensor is not None:
                                    param_tensor.data = processed_state_dict[param_key]

        # Load final LayerNorm weights
        for param_type in ["w", "b"]:
            param_key = f"ln_final.{param_type}"
            if param_key in processed_state_dict:
                if hasattr(self, "ln_final"):
                    attr_name = "weight" if param_type == "w" else "bias"
                    if hasattr(self.ln_final, attr_name):
                        param_tensor = getattr(self.ln_final, attr_name)
                        if param_tensor is not None:
                            param_tensor.data = processed_state_dict[param_key]

        # Load unembedding weights
        if "unembed.W_U" in processed_state_dict:
            # Processed weights are in [d_model, d_vocab] format, bridge expects [d_vocab, d_model]
            unembed_weight = processed_state_dict["unembed.W_U"]
            if hasattr(self, "unembed") and hasattr(self.unembed, "weight"):
                self.unembed.weight.data = unembed_weight.T  # Transpose back

    # ==================== TOKENIZATION METHODS ====================

    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        """Converts a string to a tensor of tokens.

        Args:
            input: The input to tokenize
            prepend_bos: Whether to prepend the BOS token
            padding_side: Which side to pad on
            move_to_device: Whether to move to model device
            truncate: Whether to truncate to model context length

        Returns:
            Token tensor of shape [batch, pos]
        """
        # Handle prepend_bos logic
        if prepend_bos is None:
            prepend_bos = getattr(self.cfg, "default_prepend_bos", True)

        # Handle padding_side logic
        if padding_side is None:
            padding_side = getattr(self.tokenizer, "padding_side", "right")

        # Use the pre-calculated tokenizer_prepends_bos configuration
        tokenizer_prepends_bos = getattr(self.cfg, "tokenizer_prepends_bos", True)

        if prepend_bos and not tokenizer_prepends_bos:
            # We want to prepend bos but the tokenizer doesn't automatically do it, so we add it manually
            input = utils.get_input_with_manually_prepended_bos(self.tokenizer.bos_token, input)

        if isinstance(input, str):
            input = [input]

        # Tokenize
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )["input_ids"]

        if not prepend_bos and tokenizer_prepends_bos:
            # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
            tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)

        if move_to_device:
            tokens = tokens.to(self.cfg.device)

        return tokens

    # ==================== PAST KV CACHE HELPERS ====================

    def get_pos_offset(self, past_kv_cache, batch_size: int) -> int:
        """Compute position offset from a TransformerLensKeyValueCache-like object.

        Mirrors HookedTransformer.get_pos_offset behavior for compatibility.
        """
        if past_kv_cache is None:
            return 0
        cached_batch_size, cache_ctx_length, num_heads_in_cache, d_head_in_cache = past_kv_cache[
            0
        ].past_keys.shape
        assert cached_batch_size == batch_size
        if getattr(self.cfg, "n_key_value_heads", None) is None:
            assert num_heads_in_cache == self.cfg.n_heads
        else:
            assert num_heads_in_cache == getattr(self.cfg, "n_key_value_heads")
        assert d_head_in_cache == self.cfg.d_head
        return cache_ctx_length

    def to_string(
        self,
        tokens: Union[List[int], torch.Tensor, np.ndarray],
    ) -> Union[str, List[str]]:
        """Convert tokens to string(s).

        Args:
            tokens: Tokens to convert

        Returns:
            Decoded string(s)
        """
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens)

        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")

    def to_str_tokens(
        self,
        input: Union[str, torch.Tensor, np.ndarray, List],
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
    ) -> Union[List[str], List[List[str]]]:
        """Map text or tokens to a list of tokens as strings.

        Args:
            input: The input to convert
            prepend_bos: Whether to prepend BOS token
            padding_side: Which side to pad on

        Returns:
            List of token strings
        """
        if isinstance(input, list):
            # Use cast to help mypy understand the recursive return type
            return cast(
                List[List[str]],
                [self.to_str_tokens(item, prepend_bos, padding_side) for item in input],
            )
        elif isinstance(input, str):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)[0]
        elif isinstance(input, torch.Tensor):
            tokens = input.squeeze()
            if tokens.dim() == 0:
                tokens = tokens.unsqueeze(0)
            assert (
                tokens.dim() == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens.shape}"
        elif isinstance(input, np.ndarray):
            tokens_np = input.squeeze()
            if tokens_np.ndim == 0:
                tokens_np = np.expand_dims(tokens_np, axis=0)
            assert (
                tokens_np.ndim == 1
            ), f"Invalid tokens input to to_str_tokens, has shape: {tokens_np.shape}"
            tokens = torch.tensor(tokens_np)
        else:
            raise ValueError(f"Invalid input type to to_str_tokens: {type(input)}")

        str_tokens = self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        return str_tokens

    def to_single_token(self, string: str) -> int:
        """Map a string that makes up a single token to the id for that token.

        Args:
            string: The string to convert

        Returns:
            Token ID

        Raises:
            AssertionError: If string is not a single token
        """
        token = self.to_tokens(string, prepend_bos=False).squeeze()
        if token.numel() != 1:
            raise AssertionError(f"Input string: {string} is not a single token!")
        return int(token.item())

    def get_token_position(
        self,
        single_token: Union[str, int],
        input: Union[str, torch.Tensor],
        mode="first",
        prepend_bos: Optional[Union[bool, None]] = None,
        padding_side: Optional[Union[Literal["left", "right"], None]] = None,
    ):
        """Get the position of a single_token in a string or sequence of tokens.

        Raises an error if the token is not present.

        Args:
            single_token (Union[str, int]): The token to search for. Can
                be a token index, or a string (but the string must correspond to a single token).
            input (Union[str, torch.Tensor]): The sequence to
                search in. Can be a string or a rank 1 tensor of tokens or a rank 2 tensor of tokens
                with a dummy batch dimension.
            mode (str, optional): If there are multiple matches, which match to return. Supports
                "first" or "last". Defaults to "first".
            prepend_bos (bool, optional): Whether to prepend the BOS token to the input
                (only applies when input is a string). Defaults to None, using the bridge's default.
            padding_side (Union[Literal["left", "right"], None], optional): Specifies which side to pad when tokenizing multiple
                strings of different lengths.
        """
        if isinstance(input, str):
            # If the input is a string, convert to tensor
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input

        if len(tokens.shape) == 2:
            # If the tokens have shape [1, seq_len], flatten to [seq_len]
            assert (
                tokens.shape[0] == 1
            ), f"If tokens are rank two, they must have shape [1, seq_len], not {tokens.shape}"
            tokens = tokens[0]

        if isinstance(single_token, str):
            # If the single token is a string, convert to an integer
            single_token = self.to_single_token(single_token)
        elif isinstance(single_token, torch.Tensor):
            single_token = single_token.item()

        indices = torch.arange(len(tokens), device=tokens.device)[tokens == single_token]
        assert len(indices) > 0, "The token does not occur in the prompt"
        if mode == "first":
            return indices[0].item()
        elif mode == "last":
            return indices[-1].item()
        else:
            raise ValueError(f"mode must be 'first' or 'last', not {mode}")

    def to_single_str_token(self, int_token: int) -> str:
        """Get the single token corresponding to an int in string form.

        Args:
            int_token: The token ID

        Returns:
            The token string
        """
        assert isinstance(int_token, int)
        token = self.to_str_tokens(torch.tensor([int_token]))
        if isinstance(token, list) and len(token) == 1:
            return str(token[0])
        raise AssertionError("Expected a single string token.")

    @property
    def W_K(self) -> torch.Tensor:
        """Stack the key weights across all layers."""
        weights = []
        for block in self.blocks:
            w_k = block.attn.W_K
            # Reshape from [d_model, d_model] to [n_heads, d_model, d_head]
            if w_k.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_k = w_k.reshape(self.cfg.n_heads, self.cfg.d_model, d_head)
            weights.append(w_k)
        return torch.stack(weights, dim=0)

    @property
    def W_Q(self) -> torch.Tensor:
        """Stack the query weights across all layers."""
        weights = []
        for block in self.blocks:
            w_q = block.attn.W_Q
            # Reshape from [d_model, d_model] to [n_heads, d_model, d_head]
            if w_q.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_q = w_q.reshape(self.cfg.n_heads, self.cfg.d_model, d_head)
            weights.append(w_q)
        return torch.stack(weights, dim=0)

    @property
    def W_V(self) -> torch.Tensor:
        """Stack the value weights across all layers."""
        weights = []
        for block in self.blocks:
            w_v = block.attn.W_V
            # Reshape from [d_model, d_model] to [n_heads, d_model, d_head]
            if w_v.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_v = w_v.reshape(self.cfg.n_heads, self.cfg.d_model, d_head)
            weights.append(w_v)
        return torch.stack(weights, dim=0)

    @property
    def W_O(self) -> torch.Tensor:
        """Stack the attn output weights across all layers."""
        weights = []
        for block in self.blocks:
            w_o = block.attn.W_O
            # Reshape from [d_model, d_model] to [n_heads, d_head, d_model]
            if w_o.shape == (self.cfg.d_model, self.cfg.d_model):
                d_head = self.cfg.d_model // self.cfg.n_heads
                w_o = w_o.reshape(self.cfg.n_heads, d_head, self.cfg.d_model)
            weights.append(w_o)
        return torch.stack(weights, dim=0)

    @property
    def W_in(self) -> torch.Tensor:
        """Stack the MLP input weights across all layers."""
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_gate(self) -> Union[torch.Tensor, None]:
        """Stack the MLP gate weights across all layers.

        Only works for models with gated MLPs.
        """
        if getattr(self.cfg, "gated_mlp", False):
            return torch.stack([block.mlp.W_gate for block in self.blocks], dim=0)
        else:
            return None

    @property
    def W_out(self) -> torch.Tensor:
        """Stack the MLP output weights across all layers."""
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> torch.Tensor:
        """Stack the key biases across all layers."""
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> torch.Tensor:
        """Stack the query biases across all layers."""
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> torch.Tensor:
        """Stack the value biases across all layers."""
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> torch.Tensor:
        """Stack the attn output biases across all layers."""
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> torch.Tensor:
        """Stack the MLP input biases across all layers."""
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> torch.Tensor:
        """Stack the MLP output biases across all layers."""
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self):
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self):
        return FactoredMatrix(self.W_V, self.W_O)

    def params(self):
        """Property access to model parameters in the format expected by SVDInterpreter."""
        return self.get_params()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Return named parameters in the same format as HookedTransformer.

        This ensures compatibility with tools like SVDInterpreter that expect
        parameter names like 'blocks.0.attn.W_Q' instead of the raw model names.
        """
        params_dict = self.get_params()
        for name, param in params_dict.items():
            yield name, param

    # ==================== FORWARD PASS METHODS ====================

    def forward(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_type: str = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        past_kv_cache: Optional[TransformerLensKeyValueCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_at_layer: int = 0,
        **kwargs,
    ) -> Any:
        """Forward pass through the model.

        Args:
            input: Input to the model
            return_type: Type of output to return ('logits', 'loss', 'both', None)
            loss_per_token: Whether to return loss per token
            prepend_bos: Whether to prepend BOS token
            padding_side: Which side to pad on
            past_kv_cache: Optional TransformerLensKeyValueCache for generation
            start_at_layer: Layer to start forward pass from
            **kwargs: Additional arguments passed to model

        Returns:
            Model output based on return_type
        """
        # Handle string input
        if isinstance(input, (str, list)):
            input_ids = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            input_ids = input

        # Handle explicit attention mask
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        # Handle KV cache if provided
        if past_kv_cache is not None:
            # Convert TransformerLensKeyValueCache to backend format
            # Create a list of tuples (keys, values) for each layer in backend format
            backend_cache = []
            for entry in past_kv_cache.entries:
                if entry.past_keys.numel() > 0:  # Only add if there are cached values
                    # Convert from TL format [batch, pos, n_heads, d_head] to backend format [batch, n_heads, pos, d_head]
                    cached_keys = entry.past_keys.transpose(1, 2)  # [batch, n_heads, pos, d_head]
                    cached_values = entry.past_values.transpose(
                        1, 2
                    )  # [batch, n_heads, pos, d_head]
                    backend_cache.append((cached_keys, cached_values))
                # Note: We skip empty entries rather than adding (None, None) to maintain type consistency

            kwargs["past_key_values"] = backend_cache

            # Handle attention mask from the cache
            if hasattr(past_kv_cache, "previous_attention_mask"):
                # Build attention mask that includes past context
                batch_size = input_ids.shape[0]
                current_length = input_ids.shape[1]
                past_length = past_kv_cache.previous_attention_mask.shape[1]

                # Use explicit attention mask if provided, otherwise create one for current tokens
                if attention_mask is not None:
                    current_mask = attention_mask
                else:
                    current_mask = torch.ones(
                        batch_size, current_length, dtype=torch.long, device=input_ids.device
                    )

                # Combine with past attention mask
                if past_length > 0:
                    full_attention_mask = torch.cat(
                        [past_kv_cache.previous_attention_mask, current_mask], dim=1
                    )
                else:
                    full_attention_mask = current_mask

                kwargs["attention_mask"] = full_attention_mask

            # Enable caching for the underlying model
            kwargs["use_cache"] = True
        elif "use_past_kv_cache" in kwargs and kwargs["use_past_kv_cache"]:
            # If use_past_kv_cache is True but no cache provided, enable caching
            kwargs["use_cache"] = True

        # Store reference to original TransformerLensKeyValueCache for updating
        original_tl_cache = past_kv_cache

        # Run model
        if hasattr(self.original_model, "forward"):
            # Pass labels for loss calculation if needed
            if return_type in ["loss", "both"]:
                kwargs["labels"] = input_ids
            output = self.original_model.forward(input_ids, **kwargs)
        else:
            if return_type in ["loss", "both"]:
                kwargs["labels"] = input_ids
            output = self.original_model(input_ids, **kwargs)

        # Update TransformerLensKeyValueCache if it was provided and model returned new cache
        if (
            original_tl_cache is not None
            and hasattr(output, "past_key_values")
            and output.past_key_values is not None
        ):
            # Convert backend cache format back to TransformerLens format
            backend_cache = output.past_key_values
            for i, (cached_keys, cached_values) in enumerate(backend_cache):
                if i < len(original_tl_cache.entries) and cached_keys is not None:
                    # Convert from backend format [batch, n_heads, pos, d_head] to TL format [batch, pos, n_heads, d_head]
                    tl_keys = cached_keys.transpose(1, 2)
                    tl_values = cached_values.transpose(1, 2)
                    original_tl_cache.entries[i].past_keys = tl_keys
                    original_tl_cache.entries[i].past_values = tl_values

            # Update attention mask for next iteration
            if attention_mask is not None:
                original_tl_cache.previous_attention_mask = kwargs.get(
                    "attention_mask", attention_mask
                )
            elif hasattr(original_tl_cache, "previous_attention_mask"):
                # Extend the previous mask with ones for the new tokens
                batch_size, current_length = input_ids.shape
                new_mask = torch.ones(
                    batch_size, current_length, dtype=torch.long, device=input_ids.device
                )
                if original_tl_cache.previous_attention_mask is not None:
                    original_tl_cache.previous_attention_mask = torch.cat(
                        [original_tl_cache.previous_attention_mask, new_mask], dim=1
                    )
                else:
                    original_tl_cache.previous_attention_mask = new_mask

        # Extract logits from output
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, tuple) and len(output) > 0:
            logits = output[0]
        else:
            logits = output

        # Handle different return types
        if return_type == "logits":
            return logits
        elif return_type == "loss":
            if hasattr(output, "loss") and output.loss is not None:
                return output.loss
            else:
                # Calculate loss manually
                return self.loss_fn(logits, input_ids, per_token=loss_per_token)
        elif return_type == "both":
            loss = None
            if hasattr(output, "loss") and output.loss is not None:
                loss = output.loss
            else:
                loss = self.loss_fn(logits, input_ids, per_token=loss_per_token)
            return logits, loss
        elif return_type is None:
            return output
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

    def loss_fn(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        per_token: bool = False,
    ) -> torch.Tensor:
        """Calculate cross-entropy loss.

        Args:
            logits: Model logits
            tokens: Target tokens
            per_token: Whether to return per-token loss

        Returns:
            Loss tensor
        """
        # Simple cross-entropy loss implementation
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)

        # Shift logits and tokens for next-token prediction
        target_tokens = tokens[:, 1:].contiguous()  # Remove first token (typically BOS)
        pred_logits = logits[:, :-1]

        loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            target_tokens.reshape(-1),
            reduction="none",
        )

        if per_token:
            return loss.reshape(target_tokens.shape)
        else:
            return loss.mean()

    # ==================== CACHING METHODS ====================

    @overload
    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: Literal[True] = True,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Any, ActivationCache]:
        """Run with cache - placeholder implementation."""
        pass

    @overload
    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: Literal[False],
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run with cache - placeholder implementation."""
        pass

    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        names_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        stop_at_layer: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Any, Union[ActivationCache, Dict[str, torch.Tensor]]]:
        """Run the model and cache all activations.

        Args:
            input: Input to the model
            return_cache_object: Whether to return ActivationCache object
            remove_batch_dim: Whether to remove batch dimension
            names_filter: Filter for which activations to cache (str, list of str, or callable)
            stop_at_layer: Layer to stop forward pass at (not yet fully implemented)
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, cache)
        """
        # Process names_filter to create a callable that handles legacy hook names
        # Collect all aliases from bridge components (both hook and cache aliases)
        aliases = collect_aliases_recursive(self.hook_dict)

        def create_names_filter_fn(filter_input):
            if filter_input is None:
                return lambda name: True
            elif isinstance(filter_input, str):
                # Check if this is a legacy hook name that needs mapping
                mapped_name = aliases.get(filter_input, None)
                if mapped_name:
                    return lambda name: name == mapped_name or name == filter_input
                else:
                    return lambda name: name == filter_input
            elif isinstance(filter_input, list):
                # Map all legacy names in the list to new names
                mapped_list = []
                for item in filter_input:
                    mapped_list.append(item)  # Keep original
                    mapped_name = aliases.get(item, None)
                    if mapped_name:
                        mapped_list.append(mapped_name)
                return lambda name: name in mapped_list
            elif callable(filter_input):
                return filter_input
            else:
                raise ValueError("names_filter must be a string, list of strings, or callable")

        names_filter_fn = create_names_filter_fn(names_filter)

        cache: Dict[str, torch.Tensor] = {}
        hooks: List[Tuple[HookPoint, str]] = []
        visited: set[int] = set()

        def make_cache_hook(name: str):
            def cache_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                # Handle different types of outputs from bridge components
                if tensor is None:
                    cache[name] = None
                elif isinstance(tensor, torch.Tensor):
                    cache[name] = tensor.detach().cpu()
                elif isinstance(tensor, tuple):
                    # For tuple outputs, cache the first element (usually hidden states)
                    if len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                        cache[name] = tensor[0].detach().cpu()
                    else:
                        # If tuple doesn't contain tensors, don't cache it
                        pass
                else:
                    # For other types, try to convert to tensor, otherwise skip
                    try:
                        if hasattr(tensor, "detach"):
                            cache[name] = tensor.detach().cpu()
                        # If it's not a tensor-like object, don't cache it
                    except:
                        # If conversion fails, don't cache it
                        pass
                return tensor

            return cache_hook

        # Use hook dictionary to get all available hooks
        hook_dict = self.hook_dict

        # Filter hooks based on names_filter
        for hook_name, hook in hook_dict.items():
            # Only add hook if it passes the names filter
            if names_filter_fn(hook_name):
                hooks.append((hook, hook_name))

        # Register hooks
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))

            processed_args = [input]
            # Handle string input whether passed positionally or as a kwarg
            if processed_args and isinstance(processed_args[0], str):
                assert self.tokenizer is not None, "Tokenizer must be set to pass string input."
                input_ids = self.to_tokens(processed_args[0])
                input_ids = input_ids.to(next(self.original_model.parameters()).device)
                kwargs["input_ids"] = input_ids
                processed_args = processed_args[1:]
            elif "input" in kwargs and isinstance(kwargs["input"], str):
                assert self.tokenizer is not None, "Tokenizer must be set to pass string input."
                input_ids = self.to_tokens(kwargs["input"])
                input_ids = input_ids.to(next(self.original_model.parameters()).device)
                kwargs["input_ids"] = input_ids
                del kwargs["input"]

            # Add stop_at_layer hook if specified
            if stop_at_layer is not None:
                # stop_at_layer is exclusive, so stop_at_layer=1 means run layer 0 and stop before layer 1
                # We need to hook the output of the last layer to be processed (stop_at_layer - 1)
                last_layer_to_process = stop_at_layer - 1
                if (
                    hasattr(self, "blocks")
                    and last_layer_to_process >= 0
                    and last_layer_to_process < len(self.blocks)
                ):

                    def stop_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                        raise StopAtLayerException(tensor, stop_at_layer)

                    # Add hook to the output of the last layer to be processed
                    block_hook_name = f"blocks.{last_layer_to_process}.hook_out"
                    hook_dict = self.hook_dict
                    if block_hook_name in hook_dict:
                        hook_dict[block_hook_name].add_hook(stop_hook)
                        hooks.append((hook_dict[block_hook_name], block_hook_name))

            # Run the underlying model's forward method
            # Handle device parameter properly - move model to device if specified
            filtered_kwargs = kwargs.copy()
            target_device = filtered_kwargs.pop("device", None)  # Remove device from kwargs

            if target_device is not None:
                # Ensure model is on the target device
                self.original_model = self.original_model.to(target_device)
                # Also move processed_args to the same device if needed
                if processed_args and isinstance(processed_args[0], torch.Tensor):
                    processed_args = [processed_args[0].to(target_device)] + list(
                        processed_args[1:]
                    )
                # Move any tensor kwargs to the target device
                for key, value in filtered_kwargs.items():
                    if isinstance(value, torch.Tensor):
                        filtered_kwargs[key] = value.to(target_device)

            try:
                # For caching, we want attention weights to be available for hooks
                # Add output_attentions=True if not already specified
                if "output_attentions" not in filtered_kwargs:
                    filtered_kwargs["output_attentions"] = True

                # Call forward with the input as the first argument
                if processed_args:
                    output = self.forward(processed_args[0], **filtered_kwargs)
                elif "input_ids" in filtered_kwargs:
                    # If we have input_ids but no processed_args, use the input_ids as input
                    output = self.forward(
                        filtered_kwargs["input_ids"],
                        **{k: v for k, v in filtered_kwargs.items() if k != "input_ids"},
                    )
                else:
                    output = self.forward(**filtered_kwargs)
                # Extract logits if output is a HuggingFace model output object
                if hasattr(output, "logits"):
                    output = output.logits
            except StopAtLayerException as e:
                # Return the intermediate output from the specified layer
                output = e.layer_output
            except Exception as e:
                # Re-raise any other exceptions
                raise e
            finally:
                for hp, _ in hooks:
                    hp.remove_hooks()

        if self.compatibility_mode == True:
            # If compatibility mode is enabled, we need to handle aliases
            # Create duplicate cache entries for TransformerLens compatibility
            # Use the aliases collected from components (reverse mapping: new -> old)
            # Handle the case where some alias values might be lists
            reverse_aliases = {}
            for old_name, new_name in aliases.items():
                if isinstance(new_name, list):
                    # For list values, create a mapping for each item in the list
                    for single_new_name in new_name:
                        reverse_aliases[single_new_name] = old_name
                else:
                    reverse_aliases[new_name] = old_name

            # Create duplicate entries in cache
            cache_items_to_add = {}
            for cache_name, cached_value in cache.items():
                # Check if this cache name should have an alias
                for new_name, old_name in reverse_aliases.items():
                    if cache_name == new_name:
                        cache_items_to_add[old_name] = cached_value
                        break

            # Add the aliased entries to the cache
            cache.update(cache_items_to_add)

            # Add cache entries for all aliases (both hook and cache aliases)
            for alias_name, target_name in aliases.items():
                # Handle both string and list target names
                if isinstance(target_name, list):
                    # For list targets, find the first one that exists in cache
                    for single_target in target_name:
                        if single_target in cache and alias_name not in cache:
                            cache[alias_name] = cache[single_target]
                            break
                else:
                    if target_name in cache and alias_name not in cache:
                        cache[alias_name] = cache[target_name]

        if return_cache_object:
            from transformer_lens.ActivationCache import ActivationCache

            activation_cache = ActivationCache(cache, self)
            return output, activation_cache
        else:
            return output, cache

    def run_with_hooks(
        self,
        input: Union[str, List[str], torch.Tensor],
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        return_type: Optional[str] = "logits",
        names_filter: Optional[Union[str, List[str], Callable[[str], bool]]] = None,
        stop_at_layer: Optional[int] = None,
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Any:
        """Run the model with specified forward and backward hooks.

        Args:
            input: Input to the model
            fwd_hooks: Forward hooks to apply
            bwd_hooks: Backward hooks to apply
            reset_hooks_end: Whether to reset hooks at the end
            clear_contexts: Whether to clear hook contexts
            return_type: What to return ("logits", "loss", etc.)
            names_filter: Filter for hook names (not used directly, for compatibility)
            stop_at_layer: Layer to stop at (not yet fully implemented)
            remove_batch_dim: Whether to remove batch dimension from hook inputs (only works for batch_size==1)
            **kwargs: Additional arguments

        Returns:
            Model output
        """

        # Store hooks that we add so we can remove them later
        added_hooks: List[Tuple[HookPoint, str]] = []

        def add_hook_to_point(
            hook_point: HookPoint, hook_fn: Callable, name: str, dir: Literal["fwd", "bwd"] = "fwd"
        ):
            hook_point.add_hook(hook_fn, dir=dir)
            added_hooks.append((hook_point, name))

        # Add stop_at_layer hook if specified
        if stop_at_layer is not None:
            # stop_at_layer is exclusive, so stop_at_layer=1 means run layer 0 and stop before layer 1
            # We need to hook the output of the last layer to be processed (stop_at_layer - 1)
            last_layer_to_process = stop_at_layer - 1
            if (
                hasattr(self, "blocks")
                and last_layer_to_process >= 0
                and last_layer_to_process < len(self.blocks)
            ):

                def stop_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                    raise StopAtLayerException(tensor, stop_at_layer)

                # Add hook to the output of the last layer to be processed
                block_hook_name = f"blocks.{last_layer_to_process}.hook_out"
                hook_dict = self.hook_dict
                if block_hook_name in hook_dict:
                    add_hook_to_point(hook_dict[block_hook_name], stop_hook, block_hook_name, "fwd")

        # Helper function to apply hooks based on name or filter function
        def apply_hooks(hooks: List[Tuple[Union[str, Callable], Callable]], is_fwd: bool):
            direction: Literal["fwd", "bwd"] = "fwd" if is_fwd else "bwd"
            # Collect aliases for resolving legacy hook names
            aliases = collect_aliases_recursive(self.hook_dict)

            for hook_name_or_filter, hook_fn in hooks:
                # Wrap the hook function to handle remove_batch_dim if needed
                if remove_batch_dim:
                    original_hook_fn = hook_fn

                    def wrapped_hook_fn(tensor, hook):
                        # Remove batch dimension if it's size 1
                        if tensor.shape[0] == 1:
                            tensor_no_batch = tensor.squeeze(0)
                            result = original_hook_fn(tensor_no_batch, hook)
                            # Add batch dimension back if result doesn't have it
                            if result.dim() == tensor_no_batch.dim():
                                result = result.unsqueeze(0)
                            return result
                        else:
                            return original_hook_fn(tensor, hook)

                    hook_fn = wrapped_hook_fn

                if isinstance(hook_name_or_filter, str):
                    # Direct hook name - check for aliases first
                    hook_dict = self.hook_dict
                    actual_hook_name = hook_name_or_filter

                    # If this is an alias, resolve it to the actual hook name
                    if hook_name_or_filter in aliases:
                        actual_hook_name = aliases[hook_name_or_filter]

                    if actual_hook_name in hook_dict:
                        add_hook_to_point(
                            hook_dict[actual_hook_name], hook_fn, actual_hook_name, direction
                        )
                else:
                    # Filter function
                    hook_dict = self.hook_dict
                    for name, hook_point in hook_dict.items():
                        if hook_name_or_filter(name):
                            add_hook_to_point(hook_point, hook_fn, name, direction)

        try:
            # Apply forward hooks
            apply_hooks(fwd_hooks, True)

            # Apply backward hooks (though we don't fully support them yet)
            apply_hooks(bwd_hooks, False)

            # Run the model
            try:
                output = self.forward(input, return_type=return_type or "logits", **kwargs)
            except StopAtLayerException as e:
                # Return the intermediate output from the specified layer
                output = e.layer_output

            return output

        finally:
            if reset_hooks_end:
                # Remove all hooks we added
                for hook_point, name in added_hooks:
                    hook_point.remove_hooks()

    # ==================== GENERATION METHODS ====================

    def generate(
        self,
        input: Union[str, List[str], torch.Tensor] = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        return_type: Optional[str] = "input",
        verbose: bool = True,
    ) -> Union[str, List[str], torch.Tensor]:
        """Generate text from the model - placeholder implementation."""
        # Simplified implementation - just return input
        return input

    # ==================== DEVICE MANAGEMENT ====================

    def to(self, *args, **kwargs) -> "TransformerBridge":
        """Move model to device or change dtype.

        Args:
            args: Positional arguments for nn.Module.to
            kwargs: Keyword arguments for nn.Module.to

        Returns:
            Self for chaining
        """
        self.original_model = self.original_model.to(*args, **kwargs)
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> "TransformerBridge":
        """Move model to CUDA.

        Args:
            device: CUDA device

        Returns:
            Self for chaining
        """
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        elif device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def cpu(self) -> "TransformerBridge":
        """Move model to CPU.

        Returns:
            Self for chaining
        """
        return self.to(torch.device("cpu"))  # type: ignore

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get embeddings for tokens.

        Args:
            tokens: Input tokens

        Returns:
            Token embeddings
        """
        # Use the embed component if available
        if hasattr(self, "embed") and hasattr(self.embed, "weight"):
            return torch.nn.functional.embedding(tokens, self.embed.weight)
        else:
            # Fallback to using the underlying model's embedding layer
            if hasattr(self.original_model, "get_input_embeddings"):
                embedding_layer = self.original_model.get_input_embeddings()  # type: ignore[operator]
                return embedding_layer(tokens)
            else:
                raise NotImplementedError("No embedding method available")

    def mps(self) -> "TransformerBridge":
        """Move model to MPS.

        Returns:
            Self for chaining
        """
        return self.to(torch.device("mps"))  # type: ignore

    def add_hook(self, name: str, hook_fn, dir="fwd", is_permanent=False):
        """Add a hook to a specific component."""
        # Navigate to the hook point using the name
        component = self
        parts = name.split(".")

        for part in parts[:-1]:  # All but the last part
            if hasattr(component, part):
                component = getattr(component, part)
            else:
                raise AttributeError(f"Component path '{'.'.join(parts[:-1])}' not found")

        # The last part should be a hook name
        hook_name = parts[-1]
        if hasattr(component, hook_name):
            hook_point = getattr(component, hook_name)
            if isinstance(hook_point, HookPoint):
                hook_point.add_hook(hook_fn, dir=dir, is_permanent=is_permanent)
            else:
                raise AttributeError(
                    f"'{hook_name}' is not a hook point. Found object of type: {type(hook_point)} with value: {hook_point}"
                )
        else:
            raise AttributeError(f"Hook point '{hook_name}' not found on component")

    def reset_hooks(self, clear_contexts=True):
        """Remove all hooks from the model."""

        # Recursively remove hooks from all components
        def remove_hooks_recursive(module):
            if isinstance(module, GeneralizedComponent):
                module.remove_hooks()
            for child in module.children():
                remove_hooks_recursive(child)

        remove_hooks_recursive(self)

    def get_caching_hooks(
        self,
        names_filter=None,
        incl_bwd=False,
        device=None,
        remove_batch_dim=False,
        cache=None,
        pos_slice=None,
    ):
        """Creates hooks to cache activations."""
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif isinstance(names_filter, str):
            filter_str = names_filter
            names_filter = lambda name: filter_str in name
        elif callable(names_filter):
            pass  # Already a function
        else:
            raise ValueError("names_filter must be a string, callable, or None")

        def make_cache_hook(name):
            def cache_hook(tensor, hook):
                cache[name] = tensor.detach().clone()
                if remove_batch_dim and tensor.shape[0] == 1:
                    cache[name] = cache[name].squeeze(0)
                if device is not None:
                    cache[name] = cache[name].to(device)
                return tensor

            return cache_hook

        fwd_hooks: List[Tuple[str, Callable]] = []
        bwd_hooks: List[Tuple[str, Callable]] = []

        # Collect hooks from all HookPoint objects in the model
        def collect_hooks(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if hasattr(child, "add_hook") and names_filter(full_name):
                    fwd_hooks.append((full_name, make_cache_hook(full_name)))
                collect_hooks(child, full_name)

        collect_hooks(self)

        return cache, fwd_hooks, bwd_hooks

    def hooks(self, fwd_hooks=[], bwd_hooks=[], reset_hooks_end=True, clear_contexts=False):
        """Context manager for temporarily adding hooks."""

        @contextmanager
        def _hooks_context():
            added_hooks = []

            try:
                # Add forward hooks
                for hook_name, hook_fn in fwd_hooks:
                    try:
                        self.add_hook(hook_name, hook_fn, dir="fwd")
                        added_hooks.append((hook_name, hook_fn))
                    except Exception as e:
                        print(f"Warning: Failed to add forward hook {hook_name}: {e}")

                # Add backward hooks
                for hook_name, hook_fn in bwd_hooks:
                    try:
                        self.add_hook(hook_name, hook_fn, dir="bwd")
                        added_hooks.append((hook_name, hook_fn))
                    except Exception as e:
                        print(f"Warning: Failed to add backward hook {hook_name}: {e}")

                yield

            finally:
                if reset_hooks_end:
                    # Reset all hooks
                    self.reset_hooks()

        return _hooks_context()

    def set_use_attn_result(self, use_attn_result: bool):
        """Toggle whether to explicitly calculate and expose the result for each attention head.

        Useful for interpretability but can easily burn through GPU memory.
        """
        self.cfg.use_attn_result = use_attn_result

    def set_use_split_qkv_input(self, use_split_qkv_input: bool):
        """
        Toggles whether to allow editing of inputs to each attention head.
        """
        self.cfg.use_split_qkv_input = use_split_qkv_input

    def get_params(self):
        """Access to model parameters in the format expected by SVDInterpreter.

        For missing weights, returns zero tensors of appropriate shape instead of raising exceptions.
        This ensures compatibility across different model architectures.

        Returns:
            dict: Dictionary of parameter tensors with TransformerLens naming convention

        Raises:
            ValueError: If configuration is inconsistent (e.g., cfg.n_layers != len(blocks))
        """
        return get_bridge_params(self)
