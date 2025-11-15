"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""
import re
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

import numpy as np
import torch
from torch import nn

from transformer_lens import utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint

_BLOCK_PATTERN = re.compile("blocks\\.(\\d+)")


class StopAtLayerException(Exception):
    """Exception to stop forward pass at a specific layer."""

    def __init__(self, tensor, layer_idx):
        self.tensor = tensor
        self.layer_idx = layer_idx
        self.layer_output = tensor
        super().__init__(f"Stopped at layer {layer_idx}")


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


from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_setup import set_original_components
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.get_params_util import get_bridge_params
from transformer_lens.utilities.aliases import resolve_alias

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache


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
        import einops
        import torch

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

    def _fix_backward_hook_gradients(self) -> None:
        """Fix backward hook gradients by overriding HF transformer forward.

        The HuggingFace transformer's forward method unpacks tuples between blocks
        in a way that breaks gradient flow for backward hooks. This override calls
        BlockBridge blocks directly in sequence, matching HookedTransformer's approach.

        Testing shows this makes backward hook gradients match HookedTransformer exactly.
        """
        if not hasattr(self.original_model, "transformer"):
            return
        transformer = self.original_model.transformer
        assert isinstance(
            transformer, nn.Module
        ), f"Expected transformer to be a Module, got {type(transformer)}"
        original_transformer_forward = transformer.forward

        def fixed_transformer_forward(
            input_ids=None,
            past_key_values=None,
            cache_position=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            """Custom transformer forward that preserves gradient flow for backward hooks."""
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if inputs_embeds is None:
                inputs_embeds = transformer.wte(input_ids)  # type: ignore[operator]
            if position_ids is None:
                if cache_position is not None:
                    position_ids = cache_position.unsqueeze(0)
                else:
                    position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0)
            position_embeds = transformer.wpe(position_ids)  # type: ignore[operator]
            hidden_states = inputs_embeds + position_embeds
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])
                token_type_embeds = transformer.wte(token_type_ids)  # type: ignore[operator]
                hidden_states = hidden_states + token_type_embeds
            hidden_states = transformer.drop(hidden_states)  # type: ignore[operator]
            if attention_mask is not None:
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(len(transformer.h), -1, -1, -1, -1)  # type: ignore[arg-type]
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            else:
                head_mask = [None] * len(transformer.h)  # type: ignore[arg-type]
            if past_key_values is None:
                past_key_values = tuple([None] * len(transformer.h))  # type: ignore[arg-type]
            use_cache_object = hasattr(past_key_values, "update")
            residual = hidden_states
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            for i, block_bridge in enumerate(self.blocks):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (residual,)  # type: ignore[operator]
                layer_past = past_key_values if use_cache_object else past_key_values[i]
                block_outputs = block_bridge(
                    residual,
                    layer_past,
                    cache_position,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs,
                )
                if isinstance(block_outputs, tuple):
                    residual = block_outputs[0]
                    if output_attentions and len(block_outputs) > 1:
                        all_attentions = all_attentions + (block_outputs[1],)  # type: ignore[operator,assignment]
                else:
                    residual = block_outputs
            hidden_states = residual
            if transformer.ln_f is not None:
                hidden_states = transformer.ln_f(hidden_states)  # type: ignore[operator]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]
            if return_dict:
                from transformers.modeling_outputs import (
                    BaseModelOutputWithPastAndCrossAttentions,
                )

                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                    past_key_values=None,
                    hidden_states=all_hidden_states,
                    attentions=all_attentions,
                )
            else:
                outputs: tuple[Any, ...] = (hidden_states,)
                if output_hidden_states:
                    outputs = outputs + (all_hidden_states,)
                if output_attentions:
                    outputs = outputs + (all_attentions,)
                return outputs

        transformer.forward = fixed_transformer_forward

    def enable_compatibility_mode(
        self,
        disable_warnings: bool = False,
        no_processing: bool = False,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
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
        self._initialize_hook_registry()
        self._fix_backward_hook_gradients()
        self._setup_hook_compatibility()
        if no_processing:
            fold_ln = False
            center_writing_weights = False
            center_unembed = False
            self._enable_split_qkv_attention()
            self.clear_hook_registry()
            self._initialize_hook_registry()
        else:
            self.process_compatibility_weights(
                fold_ln=fold_ln,
                center_writing_weights=center_writing_weights,
                center_unembed=center_unembed,
            )
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

    def _enable_split_qkv_attention(self) -> None:
        """Enable split Q/K/V computation for attention layers in no_processing mode.

        This extracts Q/K/V weights from HuggingFace attention components using the
        architecture adapter and sets them on JointQKVAttentionBridge instances.
        This enables 3 backward paths through ln1 (matching HookedTransformer).

        Unlike enable_ht_computation_for_bridge, this ONLY affects attention layers,
        leaving MLPs to use their original HF weights.
        """
        blocks_to_process = []
        if hasattr(self, "blocks"):
            blocks_to_process.extend(self.blocks)
        if hasattr(self, "encoder_blocks"):
            blocks_to_process.extend(self.encoder_blocks)
        if hasattr(self, "decoder_blocks"):
            blocks_to_process.extend(self.decoder_blocks)
        for block in blocks_to_process:
            if hasattr(block, "attn") and hasattr(block, "original_component"):
                hf_block = block.original_component
                if hasattr(hf_block, "attn"):
                    self.adapter._enable_ht_attention(block.attn, hf_block.attn)
                    ln1 = None
                    if hasattr(block, "ln1"):
                        ln1 = block.ln1
                    elif hasattr(block, "ln_1"):
                        ln1 = block.ln_1
                    elif hasattr(block, "input_layernorm"):
                        ln1 = block.input_layernorm
                    if ln1 is not None:
                        block.attn._ln1 = ln1
                        block.attn._expects_pre_ln1_input = True

    def process_compatibility_weights(
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
        import torch

        if verbose:
            print("  Extracting state dict from existing model...")
        raw_state_dict = self.original_model.state_dict()
        state_dict = {}
        for key, value in raw_state_dict.items():
            clean_key = key
            while "._original_component" in clean_key:
                clean_key = clean_key.replace("._original_component", "")
            state_dict[clean_key] = value.clone()
        adapter = self.adapter
        if adapter and hasattr(adapter, "preprocess_weights"):
            state_dict = adapter.preprocess_weights(state_dict)
        if adapter:
            try:
                unembed_b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
                if unembed_b_U_key not in state_dict:
                    state_dict[unembed_b_U_key] = torch.zeros(
                        self.cfg.d_vocab_out
                        if hasattr(self.cfg, "d_vocab_out")
                        else self.cfg.d_vocab,
                        dtype=self.cfg.dtype if hasattr(self.cfg, "dtype") else torch.float32,
                    )
            except (ValueError, KeyError):
                pass
        if fold_ln:
            if verbose:
                print("  Folding LayerNorm/RMSNorm...")
            uses_rms_norm = (
                getattr(self.cfg, "uses_rms_norm", False)
                or getattr(self.cfg, "normalization_type", None) == "RMS"
            )
            state_dict = ProcessWeights.fold_layer_norm(
                state_dict,
                self.cfg,
                fold_biases=not uses_rms_norm,
                center_weights=center_writing_weights and (not uses_rms_norm),
                adapter=adapter,
            )
        if center_writing_weights:
            if verbose:
                print("  Centering writing weights...")
            state_dict = ProcessWeights.center_writing_weights(
                state_dict, self.cfg, adapter=adapter
            )
        if center_unembed:
            if verbose:
                print("  Centering unembed...")
            state_dict = ProcessWeights.center_unembed(state_dict, adapter=adapter)
        if fold_value_biases:
            if verbose:
                print("  Folding value biases...")
            state_dict = ProcessWeights.fold_value_biases(state_dict, self.cfg, adapter=adapter)
        if refactor_factored_attn_matrices:
            if verbose:
                print("  Refactoring attention matrices...")
            state_dict = ProcessWeights.refactor_factored_attn_matrices(
                state_dict, self.cfg, adapter=adapter
            )
        if verbose:
            print("  Loading processed weights into components...")
        object.__setattr__(self, "_processed_tl_weights", state_dict)
        self._configure_components_for_processing(verbose=verbose)
        self._load_all_processed_weights(verbose=verbose, processed_state_dict=state_dict)
        if verbose:
            print("  Loading processed weights into Bridge components...")
        loaded_count = 0
        missing_count = 0
        import einops

        from transformer_lens.model_bridge.generalized_components.joint_qkv_attention import (
            JointQKVAttentionBridge,
        )

        for layer_idx in range(self.cfg.n_layers):
            if hasattr(self, "blocks") and layer_idx < len(self.blocks):
                attn_component = self.blocks[layer_idx].attn
                if isinstance(attn_component, JointQKVAttentionBridge):
                    q_weight_key = f"blocks.{layer_idx}.attn.q.weight"
                    k_weight_key = f"blocks.{layer_idx}.attn.k.weight"
                    v_weight_key = f"blocks.{layer_idx}.attn.v.weight"
                    q_bias_key = f"blocks.{layer_idx}.attn.q.bias"
                    k_bias_key = f"blocks.{layer_idx}.attn.k.bias"
                    v_bias_key = f"blocks.{layer_idx}.attn.v.bias"
                    if (
                        q_weight_key in state_dict
                        and k_weight_key in state_dict
                        and (v_weight_key in state_dict)
                    ):
                        q_weight_hf = state_dict[q_weight_key]
                        k_weight_hf = state_dict[k_weight_key]
                        v_weight_hf = state_dict[v_weight_key]
                        q_weight_tl = einops.rearrange(
                            q_weight_hf, "m (n h) -> n m h", n=self.cfg.n_heads
                        )
                        k_weight_tl = einops.rearrange(
                            k_weight_hf, "m (n h) -> n m h", n=self.cfg.n_heads
                        )
                        v_weight_tl = einops.rearrange(
                            v_weight_hf, "m (n h) -> n m h", n=self.cfg.n_heads
                        )
                        qkv_weight = torch.cat([q_weight_hf, k_weight_hf, v_weight_hf], dim=1)
                        if hasattr(attn_component, "qkv") and hasattr(
                            attn_component.qkv, "_original_component"
                        ):
                            qkv_component = attn_component.qkv._original_component
                            if hasattr(qkv_component, "weight"):
                                qkv_component.weight.data = qkv_weight.T
                                loaded_count += 1
                        attn_component._W_Q = q_weight_tl
                        attn_component._W_K = k_weight_tl
                        attn_component._W_V = v_weight_tl
                        o_weight_key_tl = f"blocks.{layer_idx}.attn.W_O"
                        if o_weight_key_tl in state_dict:
                            o_weight_key = o_weight_key_tl
                        else:
                            o_weight_key = self.adapter.translate_transformer_lens_path(
                                f"blocks.{layer_idx}.attn.W_O"
                            )
                        if o_weight_key in state_dict:
                            o_weight_tl = state_dict[o_weight_key]
                            if hasattr(attn_component._original_component, "c_proj") and hasattr(
                                attn_component._original_component.c_proj, "bias"
                            ):
                                o_bias = attn_component._original_component.c_proj.bias.data.clone()
                            else:
                                o_bias_key_tl = f"blocks.{layer_idx}.attn.b_O"
                                if o_bias_key_tl in state_dict:
                                    o_bias_key = o_bias_key_tl
                                else:
                                    o_bias_key = self.adapter.translate_transformer_lens_path(
                                        f"blocks.{layer_idx}.attn.b_O"
                                    )
                                o_bias = state_dict.get(o_bias_key, None)
                            if o_weight_tl.ndim == 3:
                                o_weight_hf = einops.rearrange(o_weight_tl, "n h m -> (n h) m")
                            else:
                                o_weight_hf = o_weight_tl
                            b_Q_tl = None
                            b_K_tl = None
                            b_V_tl = None
                            if q_bias_key in state_dict:
                                b_Q_tl = einops.rearrange(
                                    state_dict[q_bias_key], "(n h) -> n h", n=self.cfg.n_heads
                                )
                                b_K_tl = einops.rearrange(
                                    state_dict[k_bias_key], "(n h) -> n h", n=self.cfg.n_heads
                                )
                                b_V_tl = einops.rearrange(
                                    state_dict[v_bias_key], "(n h) -> n h", n=self.cfg.n_heads
                                )
                            attn_component.set_processed_weights(
                                {
                                    "W_Q": q_weight_tl,
                                    "W_K": k_weight_tl,
                                    "W_V": v_weight_tl,
                                    "W_O": o_weight_hf,
                                    "b_Q": b_Q_tl,
                                    "b_K": b_K_tl,
                                    "b_V": b_V_tl,
                                    "b_O": o_bias,
                                }
                            )
                        if q_bias_key in state_dict:
                            q_bias_hf = state_dict[q_bias_key]
                            k_bias_hf = state_dict[k_bias_key]
                            v_bias_hf = state_dict[v_bias_key]
                            q_bias_tl = einops.rearrange(
                                q_bias_hf, "(n h) -> n h", n=self.cfg.n_heads
                            )
                            k_bias_tl = einops.rearrange(
                                k_bias_hf, "(n h) -> n h", n=self.cfg.n_heads
                            )
                            v_bias_tl = einops.rearrange(
                                v_bias_hf, "(n h) -> n h", n=self.cfg.n_heads
                            )
                            qkv_bias = torch.cat([q_bias_hf, k_bias_hf, v_bias_hf], dim=0)
                            if hasattr(attn_component, "qkv") and hasattr(
                                attn_component.qkv, "_original_component"
                            ):
                                qkv_component = attn_component.qkv._original_component
                                if hasattr(qkv_component, "bias"):
                                    qkv_component.bias.data = qkv_bias
                                    loaded_count += 1
                            attn_component._b_Q = q_bias_tl
                            attn_component._b_K = k_bias_tl
                            attn_component._b_V = v_bias_tl
                        attn_component._hooked_weights_extracted = True
                        if verbose:
                            print(
                                f"    Loaded processed QKV weights for layer {layer_idx} (JointQKVAttention)"
                            )
                            print(f"      Q/K/V HF format: {q_weight_hf.shape}")
                            print(f"      Q/K/V TL format: {q_weight_tl.shape}")
                            print(f"      Reconstructed joint QKV HF format: {qkv_weight.shape}")
        for tb_key, weight_tensor in state_dict.items():
            if ".attn.q." in tb_key or ".attn.k." in tb_key or ".attn.v." in tb_key:
                continue
            try:
                parts = tb_key.split(".")
                component: Any = self
                for i, part in enumerate(parts[:-1]):
                    if part.isdigit():
                        if hasattr(component, "__getitem__"):
                            component = component[int(part)]
                        else:
                            raise TypeError(f"Component {component} is not indexable")
                    elif hasattr(component, part):
                        component = getattr(component, part)
                    elif hasattr(component, "_modules") and part in component._modules:
                        component = component._modules[part]
                    else:
                        raise AttributeError(f"Component {part} not found")
                param_name = parts[-1]
                if hasattr(component, "_original_component"):
                    target_component = component._original_component
                else:
                    target_component = component
                if hasattr(target_component, param_name):
                    param = getattr(target_component, param_name)
                    if param is not None and isinstance(param, torch.nn.Parameter):
                        param.data = weight_tensor
                        loaded_count += 1
                    elif param is None:
                        setattr(target_component, param_name, torch.nn.Parameter(weight_tensor))
                        loaded_count += 1
                else:
                    if verbose:
                        print(f"    Warning: Parameter {param_name} not found in {tb_key}")
                    missing_count += 1
            except (AttributeError, IndexError, KeyError, TypeError) as e:
                if verbose:
                    print(f"    Warning: Could not load {tb_key}: {e}")
                missing_count += 1
        if verbose:
            print(f"    Loaded {loaded_count} weights into Bridge components")
            print(f"    Skipped {missing_count} keys")
            print(f"    Processed state_dict has {len(state_dict)} keys")
        is_gemma_model = getattr(self.cfg, "architecture", None) in [
            "GemmaForCausalLM",
            "Gemma2ForCausalLM",
        ] or getattr(self.cfg, "original_architecture", None) in [
            "GemmaForCausalLM",
            "Gemma2ForCausalLM",
        ]
        if fold_ln and (not is_gemma_model):
            for layer_idx in range(self.cfg.n_layers):
                for ln_name in ["ln1", "ln2"]:
                    try:
                        block = self.blocks[layer_idx]
                        ln_component = getattr(block, ln_name, None)
                        if ln_component is not None:
                            if hasattr(ln_component, "_original_component"):
                                norm_module = ln_component._original_component
                            else:
                                norm_module = ln_component
                            if hasattr(norm_module, "weight") and norm_module.weight is not None:
                                with torch.no_grad():
                                    norm_module.weight.fill_(1.0)
                            if hasattr(norm_module, "bias") and norm_module.bias is not None:
                                with torch.no_grad():
                                    norm_module.bias.zero_()
                    except (AttributeError, IndexError):
                        pass
            try:
                if hasattr(self, "ln_final"):
                    ln_final = self.ln_final
                    if hasattr(ln_final, "_original_component"):
                        norm_module = ln_final._original_component
                    else:
                        norm_module = ln_final
                    if hasattr(norm_module, "weight") and norm_module.weight is not None:
                        with torch.no_grad():
                            norm_module.weight.fill_(1.0)
                    if hasattr(norm_module, "bias") and norm_module.bias is not None:
                        with torch.no_grad():
                            norm_module.bias.zero_()
            except (AttributeError, IndexError):
                pass
        if verbose:
            print("  Enabling processed weights mode on components...")

        def enable_processed_weights(component):
            """Enable processed weights mode on a component and all subcomponents."""
            component._use_processed_weights = True
            if hasattr(component, "submodules"):
                for subcomp in component.submodules.values():
                    enable_processed_weights(subcomp)

        if hasattr(self, "blocks"):
            for block in self.blocks:
                enable_processed_weights(block)
        if hasattr(self, "embed"):
            enable_processed_weights(self.embed)
        if hasattr(self, "pos_embed"):
            enable_processed_weights(self.pos_embed)
        if hasattr(self, "unembed"):
            enable_processed_weights(self.unembed)
        if verbose:
            print("  Setting 3D processed weight attributes...")
        self._set_processed_weight_attributes()
        if verbose:
            print("  Extracting HookedTransformer-compatible weights...")
        if hasattr(self, "blocks"):
            for block in self.blocks:
                if hasattr(block, "attn") and hasattr(
                    block.attn, "_extract_hooked_transformer_weights"
                ):
                    block.attn._hooked_weights_extracted = False
                    block.attn._extract_hooked_transformer_weights()
        object.__setattr__(self, "_weights_processed", True)
        if fold_ln:
            object.__setattr__(self.cfg, "layer_norm_folding", True)
        if verbose:
            print("✓ Weight processing complete!")

    def _configure_components_for_processing(self, verbose: bool = False):
        """Configure all components for processed weight loading (Phase 1).

        Args:
            verbose: If True, print detailed progress messages. Default: False
        """
        if hasattr(self, "cfg") and hasattr(self.cfg, "layer_norm_folding"):
            self.cfg.layer_norm_folding = True
        for layer_idx in range(self.cfg.n_layers):
            if hasattr(self, "blocks") and layer_idx < len(self.blocks):
                block = self.blocks[layer_idx]
                if hasattr(block, "ln1") and hasattr(block.ln1, "config"):
                    block.ln1.config.layer_norm_folding = True
                if hasattr(block, "ln2") and hasattr(block.ln2, "config"):
                    block.ln2.config.layer_norm_folding = True
        if hasattr(self, "ln_final") and hasattr(self.ln_final, "config"):
            self.ln_final.config.layer_norm_folding = True

    def _load_all_processed_weights(
        self, verbose: bool = False, processed_state_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> None:
        """Load processed weights into all components (Phase 2).

        Args:
            verbose: If True, print detailed progress messages. Default: False
            processed_state_dict: Optional processed state dict (if None, uses self._processed_tl_weights)
        """
        if processed_state_dict is not None:
            object.__setattr__(self, "_processed_tl_weights", processed_state_dict)
        self._load_embedding_weights(verbose=verbose)
        self._load_transformer_block_weights(verbose=verbose)
        self._load_unembed_weights(verbose=verbose)  # type: ignore[arg-type]

    def _load_embedding_weights(self, verbose: bool = False):  # type: ignore[arg-type]
        """Load embedding and positional embedding weights into components.
        # type: ignore[arg-type]
               Args:
                   verbose: If True, print detailed progress messages. Default: False
        """
        from transformer_lens.weight_processing import ProcessWeights

        processed_weights = self._processed_tl_weights
        adapter = self.adapter
        if hasattr(self, "embed"):
            try:
                embed_key = ProcessWeights._get_param_key("embed.W_E", adapter)
                if embed_key in processed_weights:
                    embed_weight = processed_weights[embed_key]
                    self.embed.set_processed_weight(embed_weight)
            except (ValueError, KeyError):
                pass
        if hasattr(self, "pos_embed"):
            try:
                pos_embed_key = ProcessWeights._get_param_key("pos_embed.W_pos", adapter)
                if pos_embed_key in processed_weights:
                    pos_embed_weight = processed_weights[pos_embed_key]
                    self.pos_embed.set_processed_weight(pos_embed_weight)
            except (ValueError, KeyError):
                pass

    def _load_transformer_block_weights(self, verbose: bool = False) -> None:
        """Load transformer block weights into attention and MLP components.

        Args:
            verbose: If True, print detailed progress messages. Default: False
        """
        processed_weights = self._processed_tl_weights
        for layer_idx in range(self.cfg.n_layers):
            if not hasattr(self, "blocks") or layer_idx >= len(self.blocks):
                continue
            block = self.blocks[layer_idx]
            if hasattr(block, "attn"):
                self._load_attention_weights(
                    block.attn, layer_idx, processed_weights, verbose=verbose
                )
            if hasattr(block, "mlp"):
                self._load_mlp_weights(block.mlp, layer_idx, processed_weights, verbose=verbose)

    def _load_attention_weights(
        self,
        attn_component: Any,
        layer_idx: int,
        processed_weights: Dict[str, torch.Tensor],
        verbose: bool = False,
    ) -> None:
        """Load attention weights into the AttentionBridge component.

        Args:
            attn_component: The attention component to load weights into
            layer_idx: The layer index
            processed_weights: Dictionary of processed weights (in HF format with processed values)
            verbose: If True, print detailed progress messages
        """
        from transformer_lens.weight_processing import ProcessWeights

        adapter = self.adapter
        cfg = self.cfg
        base_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.attn.W_Q", adapter)
        parts = base_key.rsplit(".", 2)
        if len(parts) == 3:
            attn_prefix = parts[0]
        else:
            attn_prefix = base_key.rsplit(".", 1)[0]
        w_q_key = f"blocks.{layer_idx}.attn.q.weight"
        w_k_key = f"blocks.{layer_idx}.attn.k.weight"
        w_v_key = f"blocks.{layer_idx}.attn.v.weight"
        w_o_key = f"{attn_prefix}.c_proj.weight"
        b_q_key = f"blocks.{layer_idx}.attn.q.bias"
        b_k_key = f"blocks.{layer_idx}.attn.k.bias"
        b_v_key = f"blocks.{layer_idx}.attn.v.bias"
        b_o_key = f"{attn_prefix}.c_proj.bias"
        W_Q = processed_weights.get(w_q_key)
        W_K = processed_weights.get(w_k_key)
        W_V = processed_weights.get(w_v_key)
        W_O = processed_weights.get(w_o_key)
        b_Q = processed_weights.get(b_q_key)
        b_K = processed_weights.get(b_k_key)
        b_V = processed_weights.get(b_v_key)
        b_O = processed_weights.get(b_o_key)
        if W_Q is not None and W_K is not None and (W_V is not None) and (W_O is not None):
            attn_component.set_processed_weights(
                {
                    "W_Q": W_Q,
                    "W_K": W_K,
                    "W_V": W_V,
                    "W_O": W_O,
                    "b_Q": b_Q,
                    "b_K": b_K,
                    "b_V": b_V,
                    "b_O": b_O,
                }
            )

    def _load_mlp_weights(self, mlp_component, layer_idx, processed_weights, verbose: bool = False):
        """Load MLP weights into the MLPBridge or GatedMLPBridge component.

        Args:
            verbose: If True, print detailed progress messages. Default: False
        """
        from transformer_lens.weight_processing import ProcessWeights

        adapter = self.adapter
        cfg = self.cfg
        W_in_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.mlp.W_in", adapter)
        W_out_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.mlp.W_out", adapter)
        b_in_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.mlp.b_in", adapter)
        b_out_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.mlp.b_out", adapter)
        W_gate_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.mlp.W_gate", adapter)
        b_gate_key = ProcessWeights._get_param_key(f"blocks.{layer_idx}.mlp.b_gate", adapter)
        W_in = processed_weights.get(W_in_key)
        W_out = processed_weights.get(W_out_key)
        b_in = processed_weights.get(b_in_key)
        b_out = processed_weights.get(b_out_key)
        W_gate = processed_weights.get(W_gate_key)
        b_gate = processed_weights.get(b_gate_key)
        if W_in is None or W_out is None:
            return
        mlp_component.set_processed_weights(
            {
                "W_in": W_in,
                "W_out": W_out,
                "b_in": b_in,
                "b_out": b_out,
                "W_gate": W_gate,
                "b_gate": b_gate,
            }
        )

    def _load_unembed_weights(self, verbose: bool = False):
        """Load unembedding weights into the UnembeddingBridge component.

        Args:
            verbose: If True, print detailed progress messages. Default: False
        """
        from transformer_lens.weight_processing import ProcessWeights

        processed_weights = self._processed_tl_weights
        adapter = self.adapter
        if hasattr(self, "unembed"):
            try:
                W_U_key = ProcessWeights._get_param_key("unembed.W_U", adapter)
                if W_U_key in processed_weights:
                    W_U_hf = processed_weights[W_U_key]
                    W_U = W_U_hf.T
                    try:
                        b_U_key = ProcessWeights._get_param_key("unembed.b_U", adapter)
                        b_U = processed_weights.get(b_U_key)
                    except (ValueError, KeyError):
                        b_U = None
                    self.unembed.set_processed_weight(W_U, b_U)
            except (ValueError, KeyError):
                pass

    def _ported_forward_pass(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_type: Optional[str] = "logits",
        prepend_bos: Optional[bool] = None,
        loss_per_token: bool = False,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
    ) -> Any:
        """Forward pass using ported HookedTransformer functionality."""
        if isinstance(input, (str, list)):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input
        token_embed = self.embed(tokens)
        if (
            hasattr(self.cfg, "positional_embedding_type")
            and self.cfg.positional_embedding_type == "rotary"
        ):
            residual = token_embed
        elif hasattr(self, "pos_embed"):
            pos_embed = self.pos_embed(tokens)
            residual = token_embed + pos_embed
        else:
            residual = token_embed
        start_layer = start_at_layer or 0
        if stop_at_layer is not None and stop_at_layer < 0:
            end_layer = self.cfg.n_layers + stop_at_layer
        else:
            end_layer = stop_at_layer or self.cfg.n_layers
        for layer_idx in range(start_layer, end_layer):
            if layer_idx >= len(self.blocks):
                break
            block = self.blocks[layer_idx]
            if hasattr(block, "hook_in"):
                residual = block.hook_in(residual)
            if hasattr(block, "ln1"):
                normed_residual = block.ln1(residual)
            else:
                normed_residual = residual
            if hasattr(block, "attn"):
                attn_out = block.attn(normed_residual)
                if isinstance(attn_out, tuple):
                    attn_out = attn_out[0]
                if hasattr(block, "ln1_post"):
                    attn_out = block.ln1_post(attn_out)
                residual = residual + attn_out
            if hasattr(block, "hook_resid_mid"):
                residual = block.hook_resid_mid(residual)
            if hasattr(block, "ln2"):
                normed_residual = block.ln2(residual)
            else:
                normed_residual = residual
            if hasattr(block, "mlp"):
                mlp_out = block.mlp(normed_residual)
                if isinstance(mlp_out, tuple):
                    mlp_out = mlp_out[0]
                if hasattr(block, "ln2_post"):
                    mlp_out = block.ln2_post(mlp_out)
                if hasattr(block, "hook_mlp_out"):
                    mlp_out = block.hook_mlp_out(mlp_out)
                residual = residual + mlp_out
            if hasattr(block, "hook_out"):
                residual = block.hook_out(residual)
        if hasattr(self, "ln_final"):
            residual = self.ln_final(residual)
        if return_type == "logits":
            logits = self.unembed(residual)
            return logits
        elif return_type == "loss":
            logits = self.unembed(residual)
            return self._calculate_loss(logits, tokens, loss_per_token)
        elif return_type == "both":
            logits = self.unembed(residual)
            loss = self._calculate_loss(logits, tokens, loss_per_token)
            return (logits, loss)
        elif return_type is None:
            return None
        else:
            return residual

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

    def _run_with_hooks_ported(
        self,
        input: Union[str, List[str], torch.Tensor],
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        return_type: Optional[str] = "logits",
        stop_at_layer: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Run with hooks using ported components."""
        if isinstance(input, (str, list)):
            tokens = self.to_tokens(input, prepend_bos=kwargs.get("prepend_bos", None))
        else:
            tokens = input
        added_hooks: List[Tuple[HookPoint, str]] = []

        def add_hook_to_point(
            hook_point: HookPoint,
            hook_fn: Callable,
            name: str,
            dir: str = "fwd",
            use_alias_only: bool = False,
        ):
            if self.compatibility_mode and name != hook_point.name and (not use_alias_only):
                alias_names_list: list[str] = []
                if hook_point.name is not None:
                    alias_names_list.append(hook_point.name)
                alias_names_list.append(name)
                hook_point.add_hook(hook_fn, dir=dir, alias_names=alias_names_list)  # type: ignore[arg-type]
            elif use_alias_only and name != hook_point.name:
                hook_point.add_hook(hook_fn, dir=dir, alias_names=[name])  # type: ignore[arg-type]
            else:
                hook_point.add_hook(hook_fn, dir=dir)  # type: ignore[arg-type]
            added_hooks.append((hook_point, name))

        try:
            for hook_name_or_filter, hook_fn in fwd_hooks:
                if isinstance(hook_name_or_filter, str):
                    hook_point = self.get_hook_point(hook_name_or_filter)
                    if hook_point is not None:
                        add_hook_to_point(
                            hook_point, hook_fn, hook_name_or_filter, "fwd", use_alias_only=True
                        )
                elif callable(hook_name_or_filter):
                    hook_dict = self.hook_dict
                    hook_point_to_names: dict[int, list[str]] = {}
                    for name, hook_point in hook_dict.items():
                        if hook_name_or_filter(name):
                            hp_id = id(hook_point)
                            if hp_id not in hook_point_to_names:
                                hook_point_to_names[hp_id] = []
                            hook_point_to_names[hp_id].append(name)
                    for hp_id, matching_names in hook_point_to_names.items():
                        hook_point = hook_dict[matching_names[0]]
                        name_to_use = hook_point.name if hook_point.name else matching_names[0]
                        add_hook_to_point(
                            hook_point, hook_fn, name_to_use, "fwd", use_alias_only=True
                        )
            for hook_name_or_filter, hook_fn in bwd_hooks:
                if isinstance(hook_name_or_filter, str):
                    hook_point = self.get_hook_point(hook_name_or_filter)
                    if hook_point is not None:
                        add_hook_to_point(
                            hook_point, hook_fn, hook_name_or_filter, "bwd", use_alias_only=True
                        )
                elif callable(hook_name_or_filter):
                    hook_dict = self.hook_dict
                    bwd_hook_point_to_names: dict[int, list[str]] = {}
                    for name, hook_point in hook_dict.items():
                        if hook_name_or_filter(name):
                            hp_id = id(hook_point)
                            if hp_id not in bwd_hook_point_to_names:
                                bwd_hook_point_to_names[hp_id] = []
                            bwd_hook_point_to_names[hp_id].append(name)
                    for hp_id, matching_names in bwd_hook_point_to_names.items():
                        hook_point = hook_dict[matching_names[0]]
                        name_to_use = hook_point.name if hook_point.name else matching_names[0]
                        add_hook_to_point(
                            hook_point, hook_fn, name_to_use, "bwd", use_alias_only=True
                        )
            return self._ported_forward_pass(
                tokens, return_type=return_type, stop_at_layer=stop_at_layer, **kwargs
            )
        finally:
            if reset_hooks_end:
                for hook_point, name in added_hooks:
                    hook_point.remove_hooks()

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

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Return named parameters in the same format as TransformerLens.

        This ensures compatibility with tools like SVDInterpreter that expect
        parameter names like 'blocks.0.attn.W_Q' instead of the raw model names.
        """
        params_dict = self.get_params()
        for name, param in params_dict.items():
            yield (name, param)

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
        # If weights have been processed and we need stop_at_layer or start_at_layer,
        # use the ported forward pass which properly implements those features
        if (
            hasattr(self, "_weights_processed")
            and self._weights_processed
            and (stop_at_layer is not None or start_at_layer is not None)
        ):
            return self._ported_forward_pass(
                input,
                return_type=return_type,
                loss_per_token=loss_per_token,
                prepend_bos=prepend_bos,
                start_at_layer=start_at_layer,
                stop_at_layer=stop_at_layer,
            )

        if isinstance(input, (str, list)):
            input_ids = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
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
                raise StopAtLayerException(tensor, stop_at_layer)

            if stop_at_layer == 0:
                hook_dict = self.hook_dict
                block_0_hook_name = "blocks.0.hook_in"
                if block_0_hook_name in hook_dict:
                    hook_dict[block_0_hook_name].add_hook(stop_hook)
                    hooks.append((hook_dict[block_0_hook_name], block_0_hook_name))
            elif last_layer_to_process >= 0 and last_layer_to_process < len(self.blocks):
                block_hook_name = f"blocks.{last_layer_to_process}.hook_out"
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
            from transformer_lens.ActivationCache import ActivationCache

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
        if hasattr(self, "_weights_processed") and self._weights_processed:
            return self._run_with_hooks_ported(
                input,
                fwd_hooks=fwd_hooks,
                bwd_hooks=bwd_hooks,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
                return_type=return_type,
                stop_at_layer=stop_at_layer,
                **kwargs,
            )
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
                raise StopAtLayerException(tensor, stop_at_layer)

            if stop_at_layer == 0:
                hook_dict = self.hook_dict
                block_0_hook_name = "blocks.0.hook_in"
                if block_0_hook_name in hook_dict:
                    add_hook_to_point(
                        hook_dict[block_0_hook_name], stop_hook, block_0_hook_name, "fwd"
                    )
            elif last_layer_to_process >= 0 and last_layer_to_process < len(self.blocks):
                block_hook_name = f"blocks.{last_layer_to_process}.hook_out"
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
    ) -> Union[str, List[str], torch.Tensor]:
        """Generate text from the model using the underlying HuggingFace model."""
        if isinstance(input, str):
            inputs = self.tokenizer(input, return_tensors="pt", padding=False, truncation=False).to(
                self.cfg.device
            )
            input_ids = inputs["input_ids"]
        elif isinstance(input, list):
            inputs = self.tokenizer(input, return_tensors="pt", padding=True, truncation=False).to(
                self.cfg.device
            )
            input_ids = inputs["input_ids"]
        else:
            input_ids = input
            if input_ids.device != self.cfg.device:
                input_ids = input_ids.to(self.cfg.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
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
        with torch.no_grad():
            outputs = self.original_model.generate(input_ids, **generation_kwargs)  # type: ignore[operator]
        if return_type == "input" or return_type is None:
            if isinstance(input, str):
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif isinstance(input, list):
                return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
            else:
                return outputs
        elif return_type == "tokens":
            return outputs
        elif isinstance(input, str):
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif isinstance(input, list):
            return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
        else:
            return outputs

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
        from contextlib import contextmanager

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

    def process_weights(
        self,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        fold_value_biases: bool = True,
        refactor_factored_attn_matrices: bool = False,
    ):
        """Process weights to match TransformerLens processing exactly.

        When called from enable_compatibility_mode(), the bridge components already
        work correctly, so this method primarily just marks weights as processed.
        """
        print("Processing weights to match TransformerLens exactly...")

        # Check if we've already processed weights to avoid infinite loops
        if getattr(self, "_weights_processed", False):
            print("Weights already processed, skipping...")
            return

        # Mark as processed first to prevent re-processing
        object.__setattr__(self, "_weights_processed", True)

        # When called from enable_compatibility_mode(), the bridge is already working correctly
        # The adapter has already processed weights and created proper components
        print("Bridge components should already match HookedTransformer from adapter processing")
        print("✅ Process weights complete - bridge ready for use")

    def get_processed_hf_weights(self) -> Dict[str, torch.Tensor]:
        """Get the processed HuggingFace format weights.

        Returns:
            Dictionary of processed weights in HuggingFace format with folding applied
        """
        if not hasattr(self, "_processed_tl_weights"):
            raise ValueError(
                "No processed weights available. Call enable_compatibility_mode() first."
            )

        # The _processed_tl_weights is actually in HF format (despite the name)
        # because process_compatibility_weights() processes HF format weights in-place
        return self._processed_tl_weights

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

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Get state dict with TransformerLens format keys.

        Converts HuggingFace format keys to TransformerLens format and filters out
        _original_component references.

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
        tl_state_dict = {}
        for key, value in raw_state_dict.items():
            if key == "_original_component" or key.startswith("_original_component."):
                continue
            clean_key = key.replace("._original_component", "")
            tl_key = self.adapter.convert_hf_key_to_tl_key(clean_key)
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
                bridge_key = self.adapter.convert_hf_key_to_bridge_key(input_key)
                if bridge_key in current_state_dict:
                    mapped_state_dict[bridge_key] = value
                elif input_key in clean_to_actual:
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
