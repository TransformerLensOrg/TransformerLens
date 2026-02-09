"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""
import re
from contextlib import contextmanager
from functools import lru_cache
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

import einops
import numpy as np
import torch
from torch import nn

from transformer_lens import utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_setup import set_original_components
from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.get_params_util import get_bridge_params
from transformer_lens.utilities.aliases import resolve_alias
from transformer_lens.utilities.devices import move_to_and_update_config

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache

_BLOCK_PATTERN = re.compile("blocks\\.(\\d+)")


def build_alias_to_canonical_map(hook_dict, prefix=""):
    """Build a mapping from alias hook names to their canonical names.

    Args:
        hook_dict: Dictionary mapping hook names to HookPoint objects
        prefix: Prefix for nested keys

    Returns:
        Dictionary mapping alias names to canonical names

    Example:
        If hook_dict contains:
        - "blocks.0.hook_q" -> HookPoint(name="blocks.0.attn.q.hook_out")

        Returns:
        - {"blocks.0.hook_q": "blocks.0.attn.q.hook_out"}
    """
    aliases = {}
    for key, value in hook_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            aliases.update(build_alias_to_canonical_map(value, full_key))
        elif hasattr(value, "name"):
            if key != value.name:
                aliases[full_key] = value.name
    return aliases


class TransformerBridge(nn.Module):
    """Bridge between HuggingFace and TransformerLens models.

    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the TransformerLens and HuggingFace model structures.
    """

    hook_aliases: Dict[str, Union[str, List[str]]] = {
        "hook_embed": "embed.hook_out",
        "hook_pos_embed": ["pos_embed.hook_out", "rotary_emb.hook_out"],
        "hook_unembed": "unembed.hook_out",
    }

    def __init__(self, model: nn.Module, adapter: ArchitectureAdapter, tokenizer: Any):
        """Initialize the bridge.

        Args:
            model: The model to bridge (must be a PyTorch nn.Module or PreTrainedModel)
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
        """
        super().__init__()
        self.__dict__["original_model"] = model
        self.adapter = adapter
        self.cfg = adapter.cfg
        self.tokenizer = tokenizer
        if self.cfg.d_vocab == -1:
            if hasattr(self.tokenizer, "get_vocab"):
                vocab = self.tokenizer.get_vocab()
                self.cfg.d_vocab = max(vocab.values()) + 1
            elif hasattr(self.tokenizer, "vocab"):
                self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
            else:
                self.cfg.d_vocab = getattr(self.tokenizer, "vocab_size", 50257)
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab
        self.compatibility_mode = False
        self._hook_cache = None
        self._hook_registry: Dict[str, HookPoint] = {}
        self._hook_registry_initialized = False
        self._hook_alias_registry: Dict[str, Union[str, List[str]]] = {}
        self._property_alias_registry: Dict[str, str] = {}
        # real_components maps TL keys to (remote_path, actual_instance) tuples
        # For list components, actual_instance will be a list of component instances
        self.real_components: Dict[str, tuple] = {}
        if not hasattr(self.cfg, "device") or self.cfg.device is None:
            try:
                self.cfg.device = str(next(self.original_model.parameters()).device)
            except StopIteration:
                self.cfg.device = "cpu"
        if not hasattr(adapter, "component_mapping") or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")
        original_model = self.__dict__["original_model"]
        set_original_components(self, self.adapter, original_model)
        self._initialize_hook_registry()
        self._register_aliases()
        self._register_all_aliases_recursive()
        self._setup_hook_compatibility()
        self._initialize_hooks_to_cache()

    @classmethod
    def boot_transformers(
        cls,
        model_name: str,
        hf_config_overrides: Optional[dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        tokenizer: Optional[Any] = None,
        load_weights: bool = True,
    ) -> "TransformerBridge":
        """Boot a model from HuggingFace (alias for sources.transformers.boot).

        Args:
            model_name: The name of the model to load.
            hf_config_overrides: Optional overrides applied to the HuggingFace config before model load.
            device: The device to use. If None, will be determined automatically.
            dtype: The dtype to use for the model.
            tokenizer: Optional pre-initialized tokenizer to use; if not provided one will be created.
            load_weights: If False, load model without weights (on meta device) for config inspection only.

        Returns:
            The bridge to the loaded model.
        """
        from transformer_lens.model_bridge.sources.transformers import boot

        return boot(
            model_name=model_name,
            hf_config_overrides=hf_config_overrides,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            load_weights=load_weights,
        )

    @property
    def original_model(self) -> nn.Module:
        """Get the original model."""
        if "original_model" not in self.__dict__:
            raise AttributeError("original_model has not been set")
        return self.__dict__["original_model"]

    @original_model.setter
    def original_model(self, value: nn.Module) -> None:
        """Set the original model."""
        self.__dict__["original_model"] = value

    def _register_aliases(self) -> None:
        """Register bridge-level aliases.

        This is called at the END of __init__ when all components are set up.
        It registers the top-level bridge aliases (hook_embed, hook_pos_embed, etc.)
        and creates direct attribute references.
        """
        if self.hook_aliases:
            self._hook_alias_registry.update(self.hook_aliases)
            for alias_name, target_path in self.hook_aliases.items():
                try:
                    if isinstance(target_path, list):
                        for single_target in target_path:
                            try:
                                target_obj = self
                                for part in single_target.split("."):
                                    target_obj = getattr(target_obj, part)
                                object.__setattr__(self, alias_name, target_obj)
                                if isinstance(target_obj, HookPoint):
                                    target_obj.name = alias_name
                                break
                            except AttributeError:
                                continue
                    else:
                        target_obj = self
                        for part in target_path.split("."):
                            target_obj = getattr(target_obj, part)
                        object.__setattr__(self, alias_name, target_obj)
                        if isinstance(target_obj, HookPoint):
                            target_obj.name = alias_name
                except AttributeError:
                    pass

    def _set_processed_weight_attributes(self) -> None:
        """Create 3D processed weight attributes for attention components.

        For each attention component, if it has 2D weights (q.weight, k.weight, v.weight),
        reshape them to 3D format [n_heads, d_model, d_head] and set as:
        - _processed_W_Q
        - _processed_W_K
        - _processed_W_V
        - _processed_b_Q
        - _processed_b_K
        - _processed_b_V

        This allows property aliases (W_Q, W_K, W_V) to return 3D format for
        HookedTransformer compatibility while keeping 2D format for calculations.
        """

        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_head
        d_model = self.cfg.d_model
        if not hasattr(self, "blocks"):
            return
        for block in self.blocks:
            if not hasattr(block, "attn"):
                continue
            attn = block.attn
            if not (hasattr(attn, "q") and hasattr(attn.q, "weight")):
                continue
            try:
                w_q_2d = attn.q.weight.data
                w_k_2d = attn.k.weight.data
                w_v_2d = attn.v.weight.data
                attn._processed_W_Q = einops.rearrange(
                    w_q_2d, "m (i h) -> i m h", i=n_heads, h=d_head
                )
                attn._processed_W_K = einops.rearrange(
                    w_k_2d, "m (i h) -> i m h", i=n_heads, h=d_head
                )
                attn._processed_W_V = einops.rearrange(
                    w_v_2d, "m (i h) -> i m h", i=n_heads, h=d_head
                )
                if hasattr(attn.q, "bias") and attn.q.bias is not None:
                    b_q_2d = attn.q.bias.data
                    b_k_2d = attn.k.bias.data
                    b_v_2d = attn.v.bias.data
                    attn._processed_b_Q = einops.rearrange(
                        b_q_2d, "(i h) -> i h", i=n_heads, h=d_head
                    )
                    attn._processed_b_K = einops.rearrange(
                        b_k_2d, "(i h) -> i h", i=n_heads, h=d_head
                    )
                    attn._processed_b_V = einops.rearrange(
                        b_v_2d, "(i h) -> i h", i=n_heads, h=d_head
                    )
                if hasattr(attn, "o") and hasattr(attn.o, "weight"):
                    w_o_2d = attn.o.weight.data
                    w_o_transposed = w_o_2d.T
                    attn._processed_W_O = einops.rearrange(
                        w_o_transposed, "m (i h) -> i h m", i=n_heads, h=d_head
                    )
                    if hasattr(attn.o, "bias") and attn.o.bias is not None:
                        attn._processed_b_O = attn.o.bias.data
            except Exception:
                pass

    def _register_all_aliases_recursive(self) -> None:
        """Recursively register aliases on all bridge components.

        This walks through all components and calls _register_aliases() on each one.
        Used after weight processing to ensure aliases point to processed weights.
        """
        if hasattr(self, "_register_aliases"):
            self._register_aliases()
        for module in self.modules():
            if module is not self and hasattr(module, "_register_aliases"):
                getattr(module, "_register_aliases")()

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to track HookPoint objects dynamically."""
        super().__setattr__(name, value)
        if isinstance(value, HookPoint):
            value.name = name
            self._hook_registry[name] = value
        elif hasattr(value, "get_hooks") and callable(getattr(value, "get_hooks")):
            component_hooks = value.get_hooks()
            for hook_name, hook in component_hooks.items():
                full_name = f"{name}.{hook_name}"
                hook.name = full_name
                self._hook_registry[full_name] = hook

    def _initialize_hook_registry(self) -> None:
        """Initialize the hook registry by scanning existing components."""
        if self._hook_registry_initialized:
            return
        self._scan_existing_hooks(self, "")
        self._hook_registry_initialized = True

    def _collect_component_aliases(self, component_mapping, prefix=""):
        """Recursively collect aliases from components."""
        aliases = {}
        if isinstance(component_mapping, dict):
            for name, component in component_mapping.items():
                sub_prefix = f"{prefix}.{name}" if prefix else name
                aliases.update(self._collect_component_aliases(component, sub_prefix))
        else:
            if hasattr(component_mapping, "hook_aliases") and component_mapping.hook_aliases:
                for alias_name, target in component_mapping.hook_aliases.items():
                    full_alias = f"{prefix}.{alias_name}" if prefix else alias_name
                    full_target = f"{prefix}.{target}" if prefix else target
                    aliases[full_alias] = full_target
            if hasattr(component_mapping, "submodules") and component_mapping.submodules:
                for sub_name, sub_component in component_mapping.submodules.items():
                    sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
                    aliases.update(self._collect_component_aliases(sub_component, sub_prefix))
        return aliases

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_hook_aliases_cached(
        hook_names_tuple: Tuple[str, ...], component_aliases_tuple: Tuple[Tuple[str, str], ...]
    ) -> Tuple[Tuple[str, str], ...]:
        """Cached computation of hook aliases. Takes immutable inputs for caching."""
        aliases = {}
        component_aliases = dict(component_aliases_tuple)
        for hook_name in hook_names_tuple:
            for alias_pattern, target_pattern in component_aliases.items():
                if "blocks." in target_pattern and "blocks." in hook_name:
                    block_match = _BLOCK_PATTERN.search(hook_name)
                    if block_match:
                        block_num = block_match.group(1)
                        dynamic_alias_pattern = alias_pattern.replace(
                            "blocks.", f"blocks.{block_num}."
                        )
                        dynamic_target_pattern = target_pattern.replace(
                            "blocks.", f"blocks.{block_num}."
                        )
                        if hook_name.endswith(dynamic_target_pattern):
                            target_len = len(dynamic_target_pattern)
                            alias_name = hook_name[:-target_len] + dynamic_alias_pattern
                            aliases[alias_name] = hook_name
                elif hook_name.endswith(target_pattern):
                    target_len = len(target_pattern)
                    alias_name = hook_name[:-target_len] + alias_pattern
                    aliases[alias_name] = hook_name
        return tuple(aliases.items())

    def _collect_hook_aliases_from_registry(self):
        """Collect aliases based on existing hooks in the registry."""
        if hasattr(self.adapter, "component_mapping"):
            component_aliases = self._collect_component_aliases(self.adapter.component_mapping)
            hook_names_tuple = tuple(sorted(self._hook_registry.keys()))
            component_aliases_tuple = tuple(sorted(component_aliases.items()))  # type: ignore[operator]
            aliases_tuple = self._compute_hook_aliases_cached(
                hook_names_tuple, component_aliases_tuple
            )
            return dict(aliases_tuple)
        return {}

    def _add_aliases_to_hooks(self, hooks: Dict[str, HookPoint]) -> None:
        """Add aliases to hooks in place."""
        component_aliases = self._collect_hook_aliases_from_registry()
        all_aliases = {**self.hook_aliases, **component_aliases}
        if not all_aliases:
            return
        aliased_hook_ids = set()
        for alias_name, target in all_aliases.items():
            if isinstance(target, list):
                for single_target in target:
                    try:
                        target_hook = resolve_alias(self, alias_name, {alias_name: single_target})
                        if target_hook is not None:
                            hooks[alias_name] = target_hook
                            if isinstance(target_hook, HookPoint):
                                hook_id = id(target_hook)
                                if hook_id not in aliased_hook_ids:
                                    target_hook.name = alias_name
                                    aliased_hook_ids.add(hook_id)
                            break
                    except AttributeError:
                        continue
            else:
                try:
                    target_hook = resolve_alias(self, alias_name, {alias_name: target})
                    if target_hook is not None:
                        hooks[alias_name] = target_hook
                        if isinstance(target_hook, HookPoint):
                            hook_id = id(target_hook)
                            if hook_id not in aliased_hook_ids:
                                target_hook.name = alias_name
                                aliased_hook_ids.add(hook_id)
                except AttributeError:
                    continue

    def _scan_existing_hooks(self, module: nn.Module, prefix: str = "") -> None:
        """Scan existing modules for hooks and add them to registry."""
        visited = set()

        def scan_module(mod: nn.Module, path: str = "") -> None:
            obj_id = id(mod)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if hasattr(mod, "get_hooks") and callable(getattr(mod, "get_hooks")):
                component_hooks = mod.get_hooks()  # type: ignore[operator]
                if isinstance(component_hooks, dict):
                    hooks_dict = cast(Dict[str, HookPoint], component_hooks)
                    for hook_name, hook in hooks_dict.items():
                        full_name = f"{path}.{hook_name}" if path else hook_name
                        hook.name = full_name
                        self._hook_registry[full_name] = hook
            for attr_name in dir(mod):
                if attr_name.startswith("_"):
                    continue
                if attr_name == "original_component" or attr_name == "original_model":
                    continue
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
                try:
                    attr = getattr(mod, attr_name)
                except (AttributeError, NameError, RuntimeError, TypeError):
                    continue
                name = f"{path}.{attr_name}" if path else attr_name
                if isinstance(attr, HookPoint):
                    attr.name = name
                    self._hook_registry[name] = attr
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
        """Get all HookPoint objects in the model for compatibility with TransformerLens."""
        hooks = self._hook_registry.copy()
        self._add_aliases_to_hooks(hooks)
        return hooks

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
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_pattern")  # type: ignore[operator]
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_hidden_states")
            default_cached_hooks_names.append(f"blocks.{block_idx}.attn.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_scale")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_normalized")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_in")  # type: ignore[operator]
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_scale")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_normalized")
            default_cached_hooks_names.append(f"blocks.{block_idx}.ln2_post.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.hook_in")  # type: ignore[operator]
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.in.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.in.hook_out")  # type: ignore[operator]
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.out.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.out.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.gate.hook_in")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.gate.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.mlp.hook_out")
            default_cached_hooks_names.append(f"blocks.{block_idx}.hook_out")
        for hook_name in default_cached_hooks_names:
            if hook_name in self._hook_registry:
                self.hooks_to_cache[hook_name] = self._hook_registry[hook_name]  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        """Provide a clear error message for missing attributes."""
        if name in self.__dict__:  # type: ignore[arg-type]
            return self.__dict__[name]
        # Use direct __dict__ access instead of hasattr to avoid recursion risk
        if "_modules" in self.__dict__ and name in self.__dict__["_modules"]:  # type: ignore[arg-type]
            return self.__dict__["_modules"][name]
        if "original_model" in self.__dict__ and self.__dict__["original_model"] is not None:
            try:
                name_split = name.split(".")
                if len(name_split) > 1:
                    current = getattr(self.__dict__["original_model"], name_split[0])
                    for part in name_split[1:]:  # type: ignore[operator]
                        current = getattr(current, part)
                    return current
                else:
                    return getattr(self.__dict__["original_model"], name)
            except AttributeError:
                pass  # type: ignore[operator,assignment]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        """Get a string representation of the bridge.
        # type: ignore[operator]
               Returns:
                   A string describing the bridge's components # type: ignore[operator]
        """
        lines = ["TransformerBridge:"]
        mapping = self.adapter.get_component_mapping()
        lines.extend(self._format_component_mapping(mapping, indent=1))
        return "\n".join(lines)

    def enable_compatibility_mode(
        self,
        disable_warnings: bool = False,
        no_processing: bool = False,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Enable compatibility mode for the bridge.

        This sets up the bridge to work with legacy TransformerLens components/hooks.
        It will also disable warnings about the usage of legacy components/hooks if specified.

        Args:
            disable_warnings: Whether to disable warnings about legacy components/hooks
            no_processing: Whether to disable ALL pre-processing steps of the model.
                If True, overrides fold_ln, center_writing_weights, and center_unembed to False.
            fold_ln: Whether to fold layer norm weights into the subsequent linear layers.
                Default: True. Ignored if no_processing=True.
            center_writing_weights: Whether to center the writing weights (W_out in attention and MLPs).
                Default: True. Ignored if no_processing=True.
            center_unembed: Whether to center the unembedding matrix.
                Default: True. Ignored if no_processing=True.
            fold_value_biases: Whether to fold value biases into output bias.
                Default: True. Ignored if no_processing=True.
            refactor_factored_attn_matrices: Whether to refactor factored attention matrices.
                Default: False. Ignored if no_processing=True.
        """
        from transformer_lens.utilities.bridge_components import (
            apply_fn_to_all_components,
        )

        self.compatibility_mode = True

        def set_compatibility_mode(component: Any) -> None:
            """Set compatibility mode on a component."""
            component.compatibility_mode = True
            component.disable_warnings = disable_warnings

        apply_fn_to_all_components(self, set_compatibility_mode)
        self.clear_hook_registry()
        if not no_processing:
            self.process_weights(
                fold_ln=fold_ln,
                center_writing_weights=center_writing_weights,
                center_unembed=center_unembed,
                fold_value_biases=fold_value_biases,
                refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            )
        self._initialize_hook_registry()
        self._setup_hook_compatibility()
        self._register_all_aliases_recursive()

    def _setup_hook_compatibility(self) -> None:
        """Setup hook compatibility transformations to match HookedTransformer behavior.

        This method sets up hook conversions and wrappers that ensure Bridge hooks
        have the same shapes and behavior as HookedTransformer hooks. This includes:
        1. hook_z reshaping from [batch, seq, d_model] to [batch, seq, n_heads, d_head]
        2. Wrapping HF attention forward to inject position embeddings/attention masks
        3. Architecture-specific setup (e.g., rotary embedding references)

        This is called during __init__ and should always be run, regardless of whether
        compatibility mode or weight processing is enabled.

        Note: This method is idempotent - can be called multiple times safely.
        """
        if hasattr(self.adapter, "setup_hook_compatibility"):
            self.adapter.setup_hook_compatibility(self)
        elif hasattr(self.adapter, "setup_no_processing_hooks"):
            self.adapter.setup_no_processing_hooks(self)
        blocks_to_process = []
        if hasattr(self, "blocks"):
            blocks_to_process.extend(self.blocks)
        if hasattr(self, "encoder_blocks"):
            blocks_to_process.extend(self.encoder_blocks)
        if hasattr(self, "decoder_blocks"):
            blocks_to_process.extend(self.decoder_blocks)
        for block in blocks_to_process:
            for attn_name in ["attn", "self_attn", "cross_attn"]:
                if hasattr(block, attn_name):
                    attn = getattr(block, attn_name)
                    if hasattr(attn, "setup_hook_compatibility"):
                        attn.setup_hook_compatibility()
                    elif hasattr(attn, "setup_no_processing_hooks"):
                        attn.setup_no_processing_hooks()

    def process_weights(
        self,
        verbose: bool = False,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process weights directly using ProcessWeights and architecture adapter.

        This method applies weight processing transformations to improve model interpretability
        without requiring a reference HookedTransformer model. Works with all architectures
        supported by TransformerBridge, including GPT-OSS and other new models.

        Args:
            verbose: If True, print detailed progress messages. Default: False
            fold_ln: Fold LayerNorm weights/biases into subsequent layers. Default: True
            center_writing_weights: Center weights that write to residual stream. Default: True
            center_unembed: Center unembedding weights (translation invariant). Default: True
            fold_value_biases: Fold value biases into output bias. Default: True
            refactor_factored_attn_matrices: Experimental QK/OV factorization. Default: False
        """
        from transformer_lens.weight_processing import ProcessWeights

        if verbose:
            print(f"Processing weights for {self.cfg.model_name}...")

        if verbose:
            print("  Extracting state dict from existing model...")
        state_dict = self.state_dict()
        adapter = self.adapter

        # Break weight tying between embed and unembed in the state dict
        # This is necessary for models like GPT-2 that share weights between lm_head and wte
        # We need to untie them so that center_unembed only affects unembed, not embed
        embed_key = "embed.weight"
        unembed_key = "unembed.weight"

        if embed_key in state_dict and unembed_key in state_dict:
            # Check if they point to the same tensor (weight tying)
            if state_dict[embed_key].data_ptr() == state_dict[unembed_key].data_ptr():
                if verbose:
                    print("  Breaking weight tying between embed and unembed in state dict...")
                # Clone the unembed weight to break the tie
                state_dict[unembed_key] = state_dict[unembed_key].clone()

        if adapter and hasattr(adapter, "preprocess_weights"):
            state_dict = adapter.preprocess_weights(state_dict)

        # Use unified ProcessWeights.process_weights() like HookedTransformer does
        if verbose:
            print("  Processing weights (fold_ln, center_writing_weights, etc.)...")
        state_dict = ProcessWeights.process_weights(
            state_dict,
            self.cfg,
            fold_ln=fold_ln,
            center_writing_weights=center_writing_weights,
            center_unembed=center_unembed,
            fold_value_biases=fold_value_biases,
            refactor_factored_attn_matrices=refactor_factored_attn_matrices,
            adapter=adapter,
        )

        # print("new", state_dict.keys())
        if verbose:
            print("  Distributing weights to generalized components...")
        ProcessWeights.distribute_weights_to_components(
            state_dict=state_dict,
            component_mapping=self.real_components,
        )

    def _calculate_loss(self, logits, tokens, loss_per_token=False):
        """Calculate cross-entropy loss."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none" if loss_per_token else "mean")
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        loss = loss_fct(flat_logits, flat_labels)
        if loss_per_token:
            return loss.view(shift_labels.shape)
        else:
            return loss

    def _extract_hf_weights(self):
        """Extract weights from the original HuggingFace model."""
        hf_state_dict = self.state_dict()
        for layer_idx in range(self.cfg.n_layers):
            combined_qkv_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
            combined_qkv_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"
            if combined_qkv_key in hf_state_dict:
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
        if prepend_bos is None:
            prepend_bos = getattr(self.cfg, "default_prepend_bos", True)
        if padding_side is None:
            padding_side = getattr(self.tokenizer, "padding_side", "right")
        tokenizer_prepends_bos = getattr(self.cfg, "tokenizer_prepends_bos", True)
        if prepend_bos and (not tokenizer_prepends_bos):
            input = utils.get_input_with_manually_prepended_bos(self.tokenizer.bos_token, input)
        if isinstance(input, str):
            input = [input]
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )["input_ids"]
        if not prepend_bos and tokenizer_prepends_bos:
            tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)
        if move_to_device:
            tokens = tokens.to(self.cfg.device)
        return tokens

    def get_pos_offset(self, past_kv_cache, batch_size: int) -> int:
        """Compute position offset from a TransformerLensKeyValueCache-like object.

        Mirrors TransformerLens.get_pos_offset behavior for compatibility.
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
        self, tokens: Union[List[int], torch.Tensor, np.ndarray]
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
        # In transformers v5, batch_decode treats a flat list as a single sequence,
        # not individual token IDs, so would return a single string. To maintain backward
        # compatibility with v4, we wrap each token to decode them individually.
        tokens_list = [[int(t)] for t in tokens.tolist()]
        str_tokens = self.tokenizer.batch_decode(tokens_list, clean_up_tokenization_spaces=False)
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
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input
        if len(tokens.shape) == 2:
            assert (
                tokens.shape[0] == 1
            ), f"If tokens are rank two, they must have shape [1, seq_len], not {tokens.shape}"
            tokens = tokens[0]
        if isinstance(single_token, str):
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

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Returns parameters following standard PyTorch semantics.

        This method delegates to the underlying HuggingFace model's parameters().
        For TransformerLens-style parameter generator, use tl_parameters() instead.

        Args:
            recurse: If True, yields parameters of this module and all submodules

        Returns:
            Iterator of nn.Parameter objects
        """
        return self.original_model.parameters(recurse=recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Returns named parameters following standard PyTorch semantics.

        This method delegates to the underlying HuggingFace model's named_parameters().
        For TransformerLens-style generator, use tl_named_parameters() instead.

        Args:
            prefix: Prefix to prepend to all parameter names
            recurse: If True, yields parameters of this module and all submodules
            remove_duplicate: If True, removes duplicate parameters

        Returns:
            Iterator of (name, parameter) tuples
        """
        return self.original_model.named_parameters(prefix, recurse, remove_duplicate)

    def tl_parameters(self) -> dict[str, torch.Tensor]:
        """Returns TransformerLens-style parameter dictionary.

        Parameter names follow TransformerLens conventions (e.g., 'blocks.0.attn.W_Q') and may
        include processed weights (non-leaf tensors). This format is expected by SVDInterpreter
        among other analysis tools.

        Returns:
            Dictionary mapping TransformerLens parameter names to tensors

        Example:
            >>> bridge = TransformerBridge.boot_transformers("gpt2")
            >>> tl_params = bridge.tl_parameters()
            >>> W_Q = tl_params["blocks.0.attn.W_Q"]  # Shape: [n_heads, d_model, d_head]
        """
        return self.get_params()

    def tl_named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Returns iterator of TransformerLens-style named parameters.

        This provides the same parameters as tl_parameters() but as an iterator
        for consistency with PyTorch's named_parameters() API pattern.

        Returns:
            Iterator of (name, tensor) tuples with TransformerLens naming conventions

        Example:
            >>> bridge = TransformerBridge.boot_transformers("gpt2")
            >>> for name, param in bridge.tl_named_parameters():
            ...     if "attn.W_Q" in name:
            ...         print(f"{name}: {param.shape}")  # doctest: +ELLIPSIS
            blocks.0.attn.W_Q: torch.Size([12, 768, 64])
            ...
        """
        return iter(self.get_params().items())

    def forward(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_type: Optional[str] = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        past_kv_cache: Optional[TransformerLensKeyValueCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
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
            stop_at_layer: Layer to stop forward pass at
            **kwargs: Additional arguments passed to model

        Returns:
            Model output based on return_type
        """

        # Set stop_at_layer flag on all blocks if requested
        if stop_at_layer is not None and hasattr(self, "blocks"):
            for block in self.blocks:
                block._stop_at_layer_idx = stop_at_layer

        try:
            if isinstance(input, (str, list)):
                input_ids = self.to_tokens(
                    input, prepend_bos=prepend_bos, padding_side=padding_side
                )
            else:
                input_ids = input
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if past_kv_cache is not None:
                backend_cache = []
                for entry in past_kv_cache.entries:
                    if entry.past_keys.numel() > 0:
                        cached_keys = entry.past_keys.transpose(1, 2)
                        cached_values = entry.past_values.transpose(1, 2)
                        backend_cache.append((cached_keys, cached_values))
                kwargs["past_key_values"] = backend_cache
                if hasattr(past_kv_cache, "previous_attention_mask"):
                    batch_size = input_ids.shape[0]
                    current_length = input_ids.shape[1]
                    past_length = past_kv_cache.previous_attention_mask.shape[1]
                    if attention_mask is not None:
                        current_mask = attention_mask
                    else:
                        current_mask = torch.ones(
                            batch_size, current_length, dtype=torch.long, device=input_ids.device
                        )
                    if past_length > 0:
                        full_attention_mask = torch.cat(
                            [past_kv_cache.previous_attention_mask, current_mask], dim=1
                        )
                    else:
                        full_attention_mask = current_mask
                    kwargs["attention_mask"] = full_attention_mask
                kwargs["use_cache"] = True
            elif "use_past_kv_cache" in kwargs and kwargs["use_past_kv_cache"]:
                kwargs["use_cache"] = True
            original_tl_cache = past_kv_cache
            if return_type in ["loss", "both"]:
                kwargs["labels"] = input_ids
            output = self.original_model(input_ids, **kwargs)
            if (
                original_tl_cache is not None
                and hasattr(output, "past_key_values")
                and (output.past_key_values is not None)
            ):
                backend_cache = output.past_key_values
                for i, (cached_keys, cached_values) in enumerate(backend_cache):
                    if i < len(original_tl_cache.entries) and cached_keys is not None:
                        tl_keys = cached_keys.transpose(1, 2)
                        tl_values = cached_values.transpose(1, 2)
                        original_tl_cache.entries[i].past_keys = tl_keys
                        original_tl_cache.entries[i].past_values = tl_values
                if attention_mask is not None:
                    original_tl_cache.previous_attention_mask = kwargs.get(
                        "attention_mask", attention_mask
                    )
                elif hasattr(original_tl_cache, "previous_attention_mask"):
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
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple) and len(output) > 0:
                logits = output[0]
            else:
                logits = output
            if return_type == "logits":
                return logits
            elif return_type == "loss":
                if hasattr(output, "loss") and output.loss is not None:
                    return output.loss
                else:
                    return self.loss_fn(logits, input_ids, per_token=loss_per_token)
            elif return_type == "both":
                loss = None  # type: ignore[operator]
                if hasattr(output, "loss") and output.loss is not None:
                    loss = output.loss
                else:
                    loss = self.loss_fn(logits, input_ids, per_token=loss_per_token)
                return (logits, loss)
            elif return_type is None:
                return None
            else:
                raise ValueError(f"Invalid return_type: {return_type}")
        except StopAtLayerException as e:
            # Execution stopped at the requested layer
            return e.layer_output
        finally:
            # Clean up the stop_at_layer flag on all blocks
            if stop_at_layer is not None and hasattr(self, "blocks"):
                for block in self.blocks:
                    block._stop_at_layer_idx = None

    def get_hook_point(self, hook_name: str) -> Optional[HookPoint]:
        """Get a hook point by name from the bridge's hook system."""
        if hook_name in self._hook_registry:
            return self._hook_registry[hook_name]
        try:
            parts = hook_name.split(".")
            current = self
            for part in parts:
                current = getattr(current, part)
            if isinstance(current, HookPoint):
                return current
        except AttributeError:
            pass
        return None

    def loss_fn(
        self, logits: torch.Tensor, tokens: torch.Tensor, per_token: bool = False
    ) -> torch.Tensor:
        """Calculate cross-entropy loss.

        Args:
            logits: Model logits
            tokens: Target tokens
            per_token: Whether to return per-token loss

        Returns:
            Loss tensor
        """
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        target_tokens = tokens[:, 1:].contiguous()
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
        # type: ignore[name-defined]
               Returns:
                   Tuple of (output, cache)
        """
        aliases = build_alias_to_canonical_map(self.hook_dict)

        def create_names_filter_fn(filter_input):
            if filter_input is None:
                return lambda name: True
            elif isinstance(filter_input, str):
                mapped_name = aliases.get(filter_input, None)
                if mapped_name:
                    return lambda name: name == mapped_name or name == filter_input
                else:
                    return lambda name: name == filter_input
            elif isinstance(filter_input, list):
                mapped_list = []
                for item in filter_input:
                    mapped_list.append(item)
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
                if tensor is None:
                    cache[name] = None
                elif isinstance(tensor, torch.Tensor):
                    cache[name] = tensor.detach().cpu()
                elif isinstance(tensor, tuple):
                    if len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
                        cache[name] = tensor[0].detach().cpu()
                    else:
                        pass
                else:
                    try:
                        if hasattr(tensor, "detach"):
                            cache[name] = tensor.detach().cpu()
                    except:
                        pass
                return tensor

            return cache_hook

        hook_dict = self.hook_dict
        effective_stop_layer = None
        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                effective_stop_layer = len(self.blocks) + stop_at_layer
            else:
                effective_stop_layer = stop_at_layer
        for hook_name, hook in hook_dict.items():
            if names_filter_fn(hook_name):
                if effective_stop_layer is not None:
                    if hook_name.startswith("blocks."):
                        try:
                            layer_num = int(hook_name.split(".")[1])
                            if layer_num >= effective_stop_layer:
                                continue
                        except (IndexError, ValueError):
                            pass
                hooks.append((hook, hook_name))
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))
        processed_args = [input]
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
        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                stop_at_layer = len(self.blocks) + stop_at_layer
            last_layer_to_process = stop_at_layer - 1

            def stop_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                raise StopAtLayerException(tensor)

            if stop_at_layer >= 0 and stop_at_layer < len(self.blocks):
                # Stop at the beginning of the specified block, not at the end of the previous block
                block_hook_name = f"blocks.{stop_at_layer}.hook_in"
                hook_dict = self.hook_dict
                if block_hook_name in hook_dict:
                    hook_dict[block_hook_name].add_hook(stop_hook)
                    hooks.append((hook_dict[block_hook_name], block_hook_name))
        filtered_kwargs = kwargs.copy()
        target_device = filtered_kwargs.pop("device", None)
        if target_device is not None:
            self.original_model = self.original_model.to(target_device)
            if processed_args and isinstance(processed_args[0], torch.Tensor):
                processed_args = [processed_args[0].to(target_device)] + list(processed_args[1:])
            for key, value in filtered_kwargs.items():
                if isinstance(value, torch.Tensor):
                    filtered_kwargs[key] = value.to(target_device)
        try:
            if "output_attentions" not in filtered_kwargs:
                filtered_kwargs["output_attentions"] = True
            if processed_args:
                output = self.forward(processed_args[0], **filtered_kwargs)
            elif "input_ids" in filtered_kwargs:
                output = self.forward(
                    filtered_kwargs["input_ids"],
                    **{k: v for k, v in filtered_kwargs.items() if k != "input_ids"},
                )
            else:
                output = self.forward(**filtered_kwargs)
            if hasattr(output, "logits"):
                output = output.logits
        except StopAtLayerException as e:
            output = e.layer_output
        except Exception as e:
            raise e
        finally:
            for hp, _ in hooks:
                hp.remove_hooks()
        if self.compatibility_mode == True:
            reverse_aliases = {}
            for old_name, new_name in aliases.items():
                if isinstance(new_name, list):
                    for single_new_name in new_name:
                        reverse_aliases[single_new_name] = old_name
                else:
                    reverse_aliases[new_name] = old_name
            cache_items_to_add = {}
            for cache_name, cached_value in cache.items():
                for new_name, old_name in reverse_aliases.items():
                    if cache_name == new_name:
                        cache_items_to_add[old_name] = cached_value
                        break
            cache.update(cache_items_to_add)
            for alias_name, target_name in aliases.items():
                if isinstance(target_name, list):
                    for single_target in target_name:
                        if single_target in cache and alias_name not in cache:
                            cache[alias_name] = cache[single_target]
                            break
                elif target_name in cache and alias_name not in cache:
                    cache[alias_name] = cache[target_name]
        if return_cache_object:
            activation_cache = ActivationCache(cache, self, has_batch_dim=True)
            if remove_batch_dim:
                activation_cache.remove_batch_dim()
            return (output, activation_cache)
        else:
            if remove_batch_dim:
                for key in cache:
                    if cache[key] is not None and isinstance(cache[key], torch.Tensor):
                        if cache[key].size(0) == 1:
                            cache[key] = cache[key][0]
            return (output, cache)

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
        added_hooks: List[Tuple[HookPoint, str]] = []
        effective_stop_layer = None
        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                effective_stop_layer = len(self.blocks) + stop_at_layer
            else:
                effective_stop_layer = stop_at_layer

        def add_hook_to_point(
            hook_point: HookPoint, hook_fn: Callable, name: str, dir: Literal["fwd", "bwd"] = "fwd"
        ):
            if effective_stop_layer is not None and name.startswith("blocks."):
                try:
                    layer_num = int(name.split(".")[1])
                    if layer_num >= effective_stop_layer:
                        return
                except (IndexError, ValueError):
                    pass
            if self.compatibility_mode and name != hook_point.name:
                alias_names_list: list[str] = []
                if hook_point.name is not None:
                    alias_names_list.append(hook_point.name)
                alias_names_list.append(name)
                hook_point.add_hook(hook_fn, dir=dir, alias_names=alias_names_list)
            else:
                hook_point.add_hook(hook_fn, dir=dir)
            added_hooks.append((hook_point, name))

        if stop_at_layer is not None and hasattr(self, "blocks"):
            if stop_at_layer < 0:
                stop_at_layer = len(self.blocks) + stop_at_layer
            last_layer_to_process = stop_at_layer - 1

            def stop_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                raise StopAtLayerException(tensor)

            if stop_at_layer >= 0 and stop_at_layer < len(self.blocks):
                # Stop at the beginning of the specified block, not at the end of the previous block
                block_hook_name = f"blocks.{stop_at_layer}.hook_in"
                hook_dict = self.hook_dict
                if block_hook_name in hook_dict:
                    add_hook_to_point(hook_dict[block_hook_name], stop_hook, block_hook_name, "fwd")

        def apply_hooks(hooks: List[Tuple[Union[str, Callable], Callable]], is_fwd: bool):
            direction: Literal["fwd", "bwd"] = "fwd" if is_fwd else "bwd"
            aliases = build_alias_to_canonical_map(self.hook_dict)
            for hook_name_or_filter, hook_fn in hooks:
                if remove_batch_dim:
                    original_hook_fn = hook_fn

                    # Use default argument to capture hook_fn by value, not reference
                    # This prevents all closures from using the last hook_fn in the loop
                    def wrapped_hook_fn(tensor, hook, _orig_fn=original_hook_fn):
                        if tensor.shape[0] == 1:
                            tensor_no_batch = tensor.squeeze(0)
                            result = _orig_fn(tensor_no_batch, hook)
                            if result.dim() == tensor_no_batch.dim():
                                result = result.unsqueeze(0)
                            return result
                        else:
                            return _orig_fn(tensor, hook)

                    hook_fn = wrapped_hook_fn
                if isinstance(hook_name_or_filter, str):
                    hook_dict = self.hook_dict
                    actual_hook_name = hook_name_or_filter
                    if hook_name_or_filter in aliases:
                        actual_hook_name = aliases[hook_name_or_filter]
                    if actual_hook_name in hook_dict:
                        add_hook_to_point(
                            hook_dict[actual_hook_name], hook_fn, actual_hook_name, direction
                        )
                else:
                    hook_dict = self.hook_dict
                    seen_hooks = set()
                    for name, hook_point in hook_dict.items():
                        if hook_name_or_filter(name):
                            hook_id = id(hook_point)
                            if hook_id in seen_hooks:
                                continue
                            seen_hooks.add(hook_id)
                            hook_name_to_use = hook_point.name if hook_point.name else name
                            add_hook_to_point(hook_point, hook_fn, hook_name_to_use, direction)

        try:
            apply_hooks(fwd_hooks, True)
            apply_hooks(bwd_hooks, False)
            try:
                output = self.forward(
                    input, return_type=return_type, stop_at_layer=stop_at_layer, **kwargs
                )
            except StopAtLayerException as e:
                output = e.layer_output
            return output
        finally:
            if reset_hooks_end:
                for hook_point, name in added_hooks:
                    hook_point.remove_hooks()

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
        output_logits: bool = False,
    ) -> str | list[str] | torch.Tensor | Any:  # Any for transformers.utils.ModelOutput
        # Using Any due to beartype's forward reference resolution limitations.
        # See: https://github.com/beartype/beartype/issues/546
        """Sample tokens from the model.

        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        This implementation is based on HookedTransformer.generate() to ensure consistent behavior.

        Args:
            input: Text string, list of strings, or tensor of tokens
            max_new_tokens: Maximum number of tokens to generate
            stop_at_eos: If True, stop generating tokens when the model outputs eos_token
            eos_token_id: The token ID to use for end of sentence
            do_sample: If True, sample from the model's output distribution. Otherwise, use greedy search
            top_k: Number of tokens to sample from. If None, sample from all tokens
            top_p: Probability mass to sample from. If 1.0, sample from all tokens
            temperature: Temperature for sampling. Higher values will make the model more random
            freq_penalty: Frequency penalty for sampling - how much to penalise previous tokens
            use_past_kv_cache: Not used in Bridge (kept for API compatibility)
            prepend_bos: Not used in Bridge (kept for API compatibility)
            padding_side: Not used in Bridge (kept for API compatibility)
            return_type: The type of output to return - 'input', 'str', or 'tokens'
            verbose: Not used in Bridge (kept for API compatibility)
            output_logits: If True, return a ModelOutput with sequences and logits tuple

        Returns:
            Generated sequence as string, list of strings, or tensor depending on input type and return_type.
            If output_logits=True, returns a ModelOutput-like object with 'sequences' and 'logits' attributes.
        """
        # Convert input to tokens
        if isinstance(input, str):
            input_tokens = self.tokenizer(
                input, return_tensors="pt", padding=False, truncation=False
            )["input_ids"].to(self.cfg.device)
            input_type = "str"
        elif isinstance(input, list):
            input_tokens = self.tokenizer(
                input, return_tensors="pt", padding=True, truncation=False
            )["input_ids"].to(self.cfg.device)
            input_type = "list"
        else:
            input_tokens = input.to(self.cfg.device)
            input_type = "tokens"

        # Determine return type
        if return_type == "input":
            if input_type in ["str", "list"]:
                return_type = "str"
            else:
                return_type = "tokens"

        batch_size = input_tokens.shape[0]

        # Setup EOS token handling
        stop_tokens = []
        eos_token_for_padding = 0
        if stop_at_eos:
            if eos_token_id is None:
                assert (
                    self.tokenizer.eos_token_id is not None
                ), "Must pass eos_token_id if stop_at_eos is True and tokenizer has no eos_token_id"
                eos_token_id = self.tokenizer.eos_token_id

            if isinstance(eos_token_id, int):
                stop_tokens = [eos_token_id]
                eos_token_for_padding = eos_token_id
            else:
                stop_tokens = list(eos_token_id)
                eos_token_for_padding = (
                    self.tokenizer.eos_token_id
                    if self.tokenizer.eos_token_id is not None
                    else eos_token_id[0]
                )

        # Track which sequences have finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.cfg.device)

        # Optionally collect logits at each generation step for downstream tooling/tests
        logits_seq_list: list[torch.Tensor] | None = [] if output_logits else None

        # Generate tokens
        current_tokens = input_tokens.clone()
        sampled_tokens_list = []

        for _ in range(max_new_tokens):
            # Get logits for next token
            with torch.no_grad():
                logits = self(current_tokens, return_type="logits")
                final_logits = logits[:, -1, :]

                # Collect logits if requested
                if logits_seq_list is not None:
                    logits_seq_list.append(final_logits.clone())

                # Sample next token
                if do_sample:
                    sampled_tokens = utils.sample_logits(
                        final_logits,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                        freq_penalty=freq_penalty,
                        tokens=current_tokens,
                    ).to(self.cfg.device)
                else:
                    sampled_tokens = final_logits.argmax(-1).to(self.cfg.device)

                sampled_tokens_list.append(sampled_tokens.unsqueeze(1))

                # Handle EOS tokens for finished sequences
                if stop_at_eos:
                    sampled_tokens[finished_sequences] = eos_token_for_padding
                    finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens.to(self.cfg.device),
                            torch.tensor(stop_tokens).to(self.cfg.device),
                        )
                    )

                # Append sampled token to current sequence
                current_tokens = torch.cat([current_tokens, sampled_tokens.unsqueeze(1)], dim=1)

                # Early stopping if all sequences finished
                if stop_at_eos and finished_sequences.all():
                    break

        # Concatenate all sampled tokens
        sampled_tokens = torch.cat(sampled_tokens_list, dim=1)
        output_tokens = torch.cat([input_tokens, sampled_tokens], dim=1)

        # Return ModelOutput if output_logits was requested
        if output_logits and logits_seq_list is not None:
            from transformers.utils import ModelOutput  # type: ignore

            def _logits_to_tuple(logits_list: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
                assert logits_list is not None
                # Convert list of [batch, vocab] tensors to tuple
                return tuple(logits_list)

            try:
                from transformers.generation.utils import GenerateDecoderOnlyOutput

                # Return a HF-compatible ModelOutput structure
                # GenerateDecoderOnlyOutput expects: sequences, scores (optional), logits (optional)
                return GenerateDecoderOnlyOutput(
                    sequences=cast(torch.LongTensor, output_tokens),
                    # HF's type hint says tuple[FloatTensor] but should be tuple[FloatTensor, ...]
                    # (variable-length tuple with one element per generated token)
                    logits=_logits_to_tuple(logits_seq_list),  # type: ignore[arg-type]
                )
            except (ImportError, AttributeError):
                # Fallback if GenerateDecoderOnlyOutput not available in this transformers version
                return ModelOutput(
                    sequences=output_tokens,
                    logits=_logits_to_tuple(logits_seq_list),
                )

        # Format output
        if return_type == "str":
            if input_type == "str":
                return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            else:
                decoded_texts = [
                    self.tokenizer.decode(tokens, skip_special_tokens=True)
                    for tokens in output_tokens
                ]
                return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts
        else:  # return_type == "tokens"
            return output_tokens

    def hf_generate(
        self,
        input: str | list[str] | torch.Tensor = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: int | None = None,
        do_sample: bool = True,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        use_past_kv_cache: bool = True,
        return_type: str | None = "input",
        **generation_kwargs,
    ) -> str | list[str] | torch.Tensor | Any:  # Any for HF ModelOutput types
        # Using Any due to beartype's forward reference resolution limitations.
        # See: https://github.com/beartype/beartype/issues/546
        """Generate text using the underlying HuggingFace model with full HF API support.

        This method provides direct access to HuggingFace's generation API, forwarding all
        generation parameters (including output_scores, output_logits, output_attentions,
        output_hidden_states) directly to the underlying HF model. Use this when you need
        full HuggingFace generation features not supported by the standard generate() method.

        For standard generation compatible with HookedTransformer, use generate() instead.

        Args:
            input: Text string, list of strings, or tensor of tokens
            max_new_tokens: Maximum number of tokens to generate
            stop_at_eos: If True, stop generating tokens when the model outputs eos_token
            eos_token_id: The token ID to use for end of sentence
            do_sample: If True, sample from the model's output distribution
            top_k: Number of tokens to sample from
            top_p: Probability mass to sample from
            temperature: Temperature for sampling
            use_past_kv_cache: If True, use KV caching for faster generation
            return_type: The type of output to return - 'input', 'str', or 'tokens'
            **generation_kwargs: Additional HuggingFace generation parameters including:
                - output_scores: Return generation scores
                - output_logits: Return generation logits
                - output_attentions: Return attention weights
                - output_hidden_states: Return hidden states
                - return_dict_in_generate: Return ModelOutput object
                - And any other HF generation parameters

        Returns:
            Generated sequence as string, list of strings, tensor, or HF ModelOutput
            depending on input type, return_type, and generation_kwargs.

        Example::

            # Get full HF ModelOutput with logits and attentions
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained("tiny-stories-1M")
            result = model.hf_generate(
                "Hello world",
                max_new_tokens=5,
                output_logits=True,
                output_attentions=True,
                return_dict_in_generate=True
            )
            print(result.sequences)  # Generated tokens
            print(result.logits)  # Logits for each generation step
            print(result.attentions)  # Attention weights
        """
        # Handle string input by tokenizing it
        if isinstance(input, str):
            inputs = self.tokenizer(input, return_tensors="pt", padding=False, truncation=False).to(
                self.cfg.device
            )
            input_ids = inputs["input_ids"]
            input_type = "str"
        elif isinstance(input, list):
            inputs = self.tokenizer(input, return_tensors="pt", padding=True, truncation=False).to(
                self.cfg.device
            )
            input_ids = inputs["input_ids"]
            input_type = "list"
        else:
            input_ids = input
            if input_ids.device != self.cfg.device:
                input_ids = input_ids.to(self.cfg.device)
            input_type = "tokens"

        # Build generation_kwargs from explicit args and kwargs
        generation_kwargs = dict(generation_kwargs) if generation_kwargs is not None else {}
        generation_kwargs.update(
            {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
        )

        if top_k is not None:
            generation_kwargs["top_k"] = top_k
        if top_p is not None:
            generation_kwargs["top_p"] = top_p
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        elif stop_at_eos and self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        if use_past_kv_cache:
            generation_kwargs["use_cache"] = True

        # HF dict flags that trigger ModelOutput returns
        hf_dict_flags = (
            "output_scores",
            "output_logits",
            "output_attentions",
            "output_hidden_states",
        )

        # If any HF-style output flags are provided, ensure return_dict_in_generate is set
        any_flag_set = False
        for f in hf_dict_flags:
            if generation_kwargs.get(f) is not None:
                generation_kwargs[f] = bool(generation_kwargs[f])
                any_flag_set = True

        if any_flag_set:
            generation_kwargs.setdefault("return_dict_in_generate", True)

        # Generate using the original HuggingFace model
        with torch.no_grad():
            outputs = self.original_model.generate(input_ids, **generation_kwargs)  # type: ignore[operator]

        # Check if output is a ModelOutput
        try:
            from transformers.utils import ModelOutput  # type: ignore

            is_model_output = isinstance(outputs, ModelOutput)
        except Exception:
            is_model_output = False

        # Return based on return_type and input format
        if return_type == "input" or return_type is None:
            if input_type == "str":
                # Decode the full output back to string
                if is_model_output and hasattr(outputs, "sequences"):
                    return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif input_type == "list":
                # Decode each sequence in the batch
                if is_model_output and hasattr(outputs, "sequences"):
                    return [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in outputs.sequences
                    ]
                return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
            else:
                # Return the full token sequence including input
                return outputs
        elif return_type == "tokens":
            return outputs
        else:
            # For other return types, default to the decoded text
            if input_type == "str":
                if is_model_output and hasattr(outputs, "sequences"):
                    return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif input_type == "list":
                if is_model_output and hasattr(outputs, "sequences"):
                    return [
                        self.tokenizer.decode(seq, skip_special_tokens=True)
                        for seq in outputs.sequences
                    ]
                return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
            else:
                return outputs

    def to(self, *args, **kwargs) -> "TransformerBridge":
        """Move model to device and/or change dtype.

        Args:
            args: Positional arguments for nn.Module.to
            kwargs: Keyword arguments for nn.Module.to
            print_details: Whether to print details about device/dtype changes (default: True)

        Returns:
            Self for chaining
        """
        # Extract print_details if provided
        print_details = kwargs.pop("print_details", True)

        # Handle both device and dtype changes
        # torch.nn.Module.to() supports: to(device), to(dtype), to(device, dtype),
        # to(device=...), to(dtype=...), to(device=..., dtype=...)
        target_device, target_dtype = None, None

        if len(args) >= 1:
            first_arg = args[0]
            if isinstance(first_arg, (torch.device, str)):
                target_device = first_arg
            elif isinstance(first_arg, torch.dtype):
                target_dtype = first_arg
        if len(args) >= 2:
            second_arg = args[1]
            if isinstance(second_arg, torch.dtype):
                target_dtype = second_arg

        # these override positional args
        if "device" in kwargs:
            target_device = kwargs["device"]
        if "dtype" in kwargs:
            target_dtype = kwargs["dtype"]

        if target_device is not None:
            move_to_and_update_config(self, target_device, print_details)
        if target_dtype is not None:
            move_to_and_update_config(self, target_dtype, print_details)

        # Move the original model with all original args/kwargs (with print_details removed)
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
        return self.to(torch.device("cpu"))

    def mps(self) -> "TransformerBridge":
        """Move model to MPS.

        Returns:
            Self for chaining
        """
        return self.to(torch.device("mps"))

    def add_hook(self, name: str, hook_fn, dir="fwd", is_permanent=False):
        """Add a hook to a specific component."""
        component = self
        parts = name.split(".")
        for part in parts[:-1]:
            if hasattr(component, part):
                component = getattr(component, part)
            else:
                raise AttributeError(f"Component path '{'.'.join(parts[:-1])}' not found")
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

        def remove_hooks_recursive(module):
            if isinstance(module, GeneralizedComponent):
                module.remove_hooks()
            for child in module.children():
                remove_hooks_recursive(child)

        remove_hooks_recursive(self)

    def hooks(self, fwd_hooks=[], bwd_hooks=[], reset_hooks_end=True, clear_contexts=False):
        """Context manager for temporarily adding hooks.

        Args:
            fwd_hooks: List of (hook_name, hook_fn) tuples for forward hooks
            bwd_hooks: List of (hook_name, hook_fn) tuples for backward hooks
            reset_hooks_end: If True, removes hooks when context exits
            clear_contexts: Unused (for compatibility with HookedTransformer)

        Example:
            with model.hooks(fwd_hooks=[("hook_embed", my_hook)]):
                output = model("Hello world")
        """

        @contextmanager
        def _hooks_context():
            added_hooks: List[Tuple[HookPoint, str]] = []

            def add_hook_to_point(
                hook_point: HookPoint,
                hook_fn: Callable,
                name: str,
                dir: Literal["fwd", "bwd"] = "fwd",
            ):
                if self.compatibility_mode and name != hook_point.name:
                    alias_names_list: list[str] = []
                    if hook_point.name is not None:
                        alias_names_list.append(hook_point.name)
                    alias_names_list.append(name)
                    hook_point.add_hook(hook_fn, dir=dir, alias_names=alias_names_list)
                else:
                    hook_point.add_hook(hook_fn, dir=dir)
                added_hooks.append((hook_point, name))

            def apply_hooks(hooks: List[Tuple[Union[str, Callable], Callable]], is_fwd: bool):
                direction: Literal["fwd", "bwd"] = "fwd" if is_fwd else "bwd"
                aliases = build_alias_to_canonical_map(self.hook_dict)
                for hook_name_or_filter, hook_fn in hooks:
                    if isinstance(hook_name_or_filter, str):
                        hook_dict = self.hook_dict
                        actual_hook_name = hook_name_or_filter
                        if hook_name_or_filter in aliases:
                            actual_hook_name = aliases[hook_name_or_filter]
                        if actual_hook_name in hook_dict:
                            add_hook_to_point(
                                hook_dict[actual_hook_name], hook_fn, actual_hook_name, direction
                            )
                    else:
                        hook_dict = self.hook_dict
                        seen_hooks = set()
                        for name, hook_point in hook_dict.items():
                            if hook_name_or_filter(name):
                                hook_id = id(hook_point)
                                if hook_id in seen_hooks:
                                    continue
                                seen_hooks.add(hook_id)
                                hook_name_to_use = hook_point.name if hook_point.name else name
                                add_hook_to_point(hook_point, hook_fn, hook_name_to_use, direction)

            try:
                apply_hooks(fwd_hooks, True)
                apply_hooks(bwd_hooks, False)
                yield self
            finally:
                if reset_hooks_end:
                    for hook_point, name in added_hooks:
                        hook_point.remove_hooks()

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

    def _is_valid_bridge_path(self, hf_path: str) -> bool:
        """Check if a HuggingFace path corresponds to a valid bridge component.

        This validates that the path follows the bridge component structure and doesn't
        contain nested HuggingFace components that should have been wrapped.

        Args:
            hf_path: HuggingFace path after removing _original_component

        Returns:
            True if the path is valid, False if it contains nested HF components
        """
        # Split the path into parts
        parts = hf_path.split(".")

        # Get the component mapping for validation
        component_mapping = self.adapter.component_mapping
        if not component_mapping:
            return True  # If no mapping, accept all keys

        # Walk through the path and check if each level is a registered bridge component
        # For example, transformer.h.0.mlp.in.weight should be valid
        # but transformer.h.0.mlp.c_fc.weight should be invalid (c_fc is nested HF component)

        # Start from the root
        current_component = None
        idx = 0

        # Find which top-level component this belongs to
        for tl_name, component in component_mapping.items():
            if component.name and hf_path.startswith(component.name + "."):
                current_component = component
                # Skip past the HF prefix
                remaining_path = hf_path[len(component.name) + 1 :]
                parts = remaining_path.split(".")
                idx = 0
                break

        if current_component is None:
            return True  # Path doesn't match any component, let it through

        # Special handling for blocks
        if hasattr(current_component, "is_list_item") and current_component.is_list_item:
            # Skip the layer index
            if idx < len(parts) and parts[idx].isdigit():
                idx += 1

        # Now validate the rest of the path against submodules
        while idx < len(parts):
            part = parts[idx]

            # If we hit 'weight' or 'bias', we're at a parameter - this is valid
            if part in ("weight", "bias"):
                return True

            # Check if this part is a registered submodule
            if hasattr(current_component, "submodules") and current_component.submodules:
                if part in current_component.submodules:
                    current_component = current_component.submodules[part]
                    idx += 1
                    continue
                else:
                    # This part is not a registered bridge component
                    # It's likely a nested HF component (like c_fc, c_proj, c_attn)
                    return False
            else:
                # No submodules to check, but not at a parameter yet
                # Check if next is weight/bias
                if idx + 1 < len(parts) and parts[idx + 1] in ("weight", "bias"):
                    return True
                # Otherwise this is likely a nested HF component
                return False

            idx += 1

        return True

    def _normalize_bridge_key_to_hf(self, key: str) -> str:
        """Normalize a key that uses bridge attribute names to use HF module names.

        PyTorch's state_dict uses the Python attribute names (e.g., 'ln1')
        but the conversion logic expects HF module names (e.g., 'ln_1'). This
        function only replaces non-nested component names, leaving bridge
        subcomponents (like 'in', 'out', 'q', 'k', 'v') unchanged since they're
        handled by the component structure.

        Args:
            key: Key that may use bridge attribute names

        Returns:
            Key with attribute names replaced by module names where needed
        """
        component_mapping = self.adapter.component_mapping
        if not component_mapping:
            return key

        # Build a mapping of only the direct module attribute names to HF names
        # We only care about top-level and block-level component names, NOT subcomponents
        attr_to_hf = {}

        # Map top-level components
        for tl_name, component in component_mapping.items():
            if component.name and tl_name != "blocks":
                attr_to_hf[tl_name] = component.name

        # Map block-level components (ln1, ln2, attn, mlp)
        blocks_component = component_mapping.get("blocks")
        if blocks_component and hasattr(blocks_component, "submodules"):
            for tl_subname, subcomponent in blocks_component.submodules.items():
                if subcomponent.name:
                    # Only map if the names differ (e.g., ln1 -> ln_1, but attn -> attn)
                    if tl_subname != subcomponent.name:
                        attr_to_hf[tl_subname] = subcomponent.name

        # Replace only these specific attribute names in the key
        # We need to be careful to only replace whole path components, not substrings
        parts = key.split(".")
        result_parts = []

        for part in parts:
            if part in attr_to_hf:
                result_parts.append(attr_to_hf[part])
            else:
                result_parts.append(part)

        return ".".join(result_parts)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Get state dict with TransformerLens format keys.

        Converts HuggingFace format keys to TransformerLens format and filters out
        _original_component references and nested HuggingFace components.

        This returns a clean state dict with only bridge component paths converted to TL format,
        excluding nested HF components (like c_fc, c_proj, c_attn) that exist inside
        original_component modules.

        Args:
            destination: Optional dict to store state dict in
            prefix: Optional prefix to add to all keys
            keep_vars: Whether to keep variables as Variables instead of tensors

        Returns:
            Dict containing the state dict with TransformerLens format keys
        """
        if destination is not None:
            raw_state_dict = self.original_model.state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars
            )
        else:
            raw_state_dict = self.original_model.state_dict(prefix=prefix, keep_vars=keep_vars)

        # Clean _original_component references and convert to TL format
        # Also filter out nested HuggingFace components that are wrapped by bridge components
        tl_state_dict = {}

        for key, value in raw_state_dict.items():
            # Skip _original_component keys
            if key == "_original_component" or key.startswith("_original_component."):
                continue

            # Remove all _original_component from the key
            clean_key = key.replace("._original_component", "")

            # Check if this is a valid bridge path (not a nested HF component)
            if not self._is_valid_bridge_path(clean_key):
                continue

            # Normalize bridge component names to HF names for conversion
            # (e.g., 'ln1' -> 'ln_1', 'mlp.in' -> 'mlp.c_fc')
            hf_key = self._normalize_bridge_key_to_hf(clean_key)

            # Convert to TL format - this uses the adapter's component_mapping
            tl_key = self.adapter.convert_hf_key_to_tl_key(hf_key)

            # Only add if we haven't seen this TL key yet (handles duplicates)
            if tl_key not in tl_state_dict:
                tl_state_dict[tl_key] = value

        return tl_state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load state dict into the model, handling both clean keys and original keys with _original_component references.

        Args:
            state_dict: Dictionary containing a whole state of the module
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function
            assign: Whether to assign items in the state dictionary to their corresponding keys in the module instead of copying them

        Returns:
            NamedTuple with missing_keys and unexpected_keys fields
        """
        current_state_dict = self.original_model.state_dict()
        clean_to_actual = {}
        actual_to_clean = {}
        for actual_key in current_state_dict.keys():
            if actual_key != "_original_component":
                clean_key = actual_key.replace("._original_component", "")
                clean_to_actual[clean_key] = actual_key
                actual_to_clean[actual_key] = clean_key
        mapped_state_dict = {}
        for input_key, value in state_dict.items():
            if input_key in current_state_dict:
                mapped_state_dict[input_key] = value
            else:
                if input_key in clean_to_actual:
                    actual_key = clean_to_actual[input_key]
                    mapped_state_dict[actual_key] = value
                else:
                    mapped_state_dict[input_key] = value
        effective_strict = strict and len(mapped_state_dict) == len(current_state_dict)
        return self.original_model.load_state_dict(
            mapped_state_dict, strict=effective_strict, assign=assign
        )

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
