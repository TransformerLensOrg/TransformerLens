"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LayerNormBridge,
    MLPBridge,
    UnembeddingBridge,
)
from transformer_lens.architecture_adapter.generalized_components.block import (
    BlockBridge,
)


class TransformerBridge:
    """Bridge between HuggingFace and HookedTransformer models.
    
    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the HookedTransformer and HuggingFace model structures.
    """

    def __init__(self, model: PreTrainedModel, adapter: ArchitectureAdapter, tokenizer: Any):
        """Initialize the bridge.
        
        Args:
            model: The HuggingFace model to bridge
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
        """
        self.model = model
        self.adapter = adapter
        self.cfg = adapter.cfg
        self.tokenizer = tokenizer
        # DEBUG: Bypass all submodule wrapping for debugging
        # (Commented out all code that replaces or wraps submodules)
        # This will help determine if the bridge wrappers are causing generation issues.
        
        if not hasattr(adapter, 'component_mapping') or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")
        
        # Get and replace components in the model
        embed = adapter.get_component(model, "embed")
        if not isinstance(embed, EmbeddingBridge):
            embed = EmbeddingBridge(original_component=embed, name="embed", architecture_adapter=adapter)
            # Replace in model using component mapping
            path = adapter.translate_transformer_lens_path("embed")
            self._set_by_path(model, path, embed)
        self.embed = embed
        
        # Use num_hidden_layers for Hugging Face configs, fallback to n_layers
        n_layers = getattr(self.cfg, 'num_hidden_layers', getattr(self.cfg, 'n_layers', None))
        if n_layers is None:
            raise AttributeError('Config has neither num_hidden_layers nor n_layers')
            
        # Create ModuleList for blocks
        block_bridges = nn.ModuleList()
        
        # Get the blocks path and component
        blocks = adapter.get_component(model, "blocks")
        
        # Build blocks
        for i in range(n_layers):
            # Get block components
            ln1 = adapter.get_component(model, f"blocks.{i}.ln1")
            ln2 = adapter.get_component(model, f"blocks.{i}.ln2")
            # Wrap layer norms with bridge
            if not isinstance(ln1, LayerNormBridge):
                ln1 = LayerNormBridge(original_component=ln1, name=f"blocks.{i}.ln1", architecture_adapter=adapter)
                path = adapter.translate_transformer_lens_path(f"blocks.{i}.ln1")
                self._set_by_path(model, path, ln1)
            if not isinstance(ln2, LayerNormBridge):
                ln2 = LayerNormBridge(original_component=ln2, name=f"blocks.{i}.ln2", architecture_adapter=adapter)
                path = adapter.translate_transformer_lens_path(f"blocks.{i}.ln2")
                self._set_by_path(model, path, ln2)
            attn = adapter.get_component(model, f"blocks.{i}.attn")
            if not isinstance(attn, AttentionBridge):
                attn = AttentionBridge(original_component=attn, name=f"blocks.{i}.attn", architecture_adapter=adapter)
                path = adapter.translate_transformer_lens_path(f"blocks.{i}.attn")
                self._set_by_path(model, path, attn)
            mlp = adapter.get_component(model, f"blocks.{i}.mlp")
            if not isinstance(mlp, MLPBridge):
                mlp = MLPBridge(original_component=mlp, name=f"blocks.{i}.mlp", architecture_adapter=adapter)
                path = adapter.translate_transformer_lens_path(f"blocks.{i}.mlp")
                self._set_by_path(model, path, mlp)
                
            # Create block bridge with the actual block layer
            block_bridge = BlockBridge(original_component=blocks[i], name=f"blocks.{i}", architecture_adapter=adapter)
            block_bridges.append(block_bridge)
            
        path = adapter.translate_transformer_lens_path("blocks")
        self._set_by_path(model, path, block_bridges)
        
        # Get final components
        ln_final = adapter.get_component(model, "ln_final")
        if not isinstance(ln_final, LayerNormBridge):
            ln_final = LayerNormBridge(original_component=ln_final, name="ln_final", architecture_adapter=adapter)
            # Replace in model using component mapping
            path = adapter.translate_transformer_lens_path("ln_final")
            self._set_by_path(model, path, ln_final)
        self.ln_final = ln_final
        
        unembed = adapter.get_component(model, "unembed")
        if not isinstance(unembed, UnembeddingBridge):
            unembed = UnembeddingBridge(original_component=unembed, name="unembed", architecture_adapter=adapter)
            # Replace in model using component mapping
            path = adapter.translate_transformer_lens_path("unembed")
            self._set_by_path(model, path, unembed)
        self.unembed = unembed
        
    def _set_by_path(self, obj: Any, path: str, value: Any) -> None:
        """Set a value in an object by its path.
        
        Args:
            obj: The object to modify
            path: The dot-separated path to the attribute
            value: The value to set
        """
        parts = path.split(".")
        for part in parts[:-1]:
            if part == "model":
                obj = obj.model
            elif part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
        
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
            comp = self.adapter.get_component(self.model, path)
            if hasattr(comp, 'original_component'):
                return f"{indent_str}{name}: {type(comp).__name__}({type(comp.original_component).__name__})"
            return f"{indent_str}{name}: {type(comp).__name__}"
        except Exception as e:
            return f"{indent_str}{name}: <error: {e}>"

    def _format_component_mapping(self, mapping: dict, indent: int = 0, prepend: str | None = None) -> list[str]:
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
            if isinstance(value, tuple):
                # For tuple paths (like blocks), get the TransformerLens path and component mapping
                _, sub_mapping = value  # Unpack but ignore tl_path since it's not used
                # Format the main component with prepend if provided
                path = f"{path}.0"
                lines.append(self._format_single_component(name, path, indent))
                # Recursively format subcomponents with updated prepend
                sub_lines = self._format_component_mapping(sub_mapping, indent + 1, path)
                lines.extend(sub_lines)
            else:
                # For regular components, use prepend if provided
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
        
    def generate(
        self,
        input: Any = "",
        max_new_tokens: int = 10,
        stop_at_eos: bool = True,
        eos_token_id: int | None = None,
        do_sample: bool = True,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        use_past_kv_cache: bool = True,
        prepend_bos: bool | None = None,
        padding_side: str | None = None,
        return_type: str | None = "input",
        verbose: bool = True,
    ) -> Any:
        """Sample tokens from the model, ported from HookedTransformer.generate."""
        # Tokenize input if needed
        if isinstance(input, (str, list)):
            assert self.tokenizer is not None, "Tokenizer must be set to pass string input."
            input_ids = self.tokenizer(
                input,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )["input_ids"]
        else:
            input_ids = input
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(next(self.model.parameters()).device)

        # Set up generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
        }
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id
        # Note: freq_penalty and use_past_kv_cache are not always supported by all models
        if hasattr(self.model, "generate"):
            output_ids = self.model.generate(input_ids, **gen_kwargs)
        else:
            raise RuntimeError("Underlying model does not support generate method.")

        # Return type logic (match HookedTransformer)
        if return_type == "input":
            if isinstance(input, (str, list)):
                return_type = "str"
            elif input_ids.ndim == 2:
                return_type = "tokens"
            else:
                return_type = "embeds"
        if return_type == "str":
            decoded_texts = [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_ids]
            return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts
        elif return_type == "tokens":
            return output_ids
        else:
            return output_ids

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the underlying HuggingFace model, with all arguments passed through."""
        # Pre-processing (if needed)
        output = self.model(*args, **kwargs)
        # Post-processing (if needed)
        return output

    def run_with_cache(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, torch.Tensor]]:
        """Run the model and cache all activations at HookPoint objects.
        Returns (output, cache_dict)."""
        from transformer_lens.hook_points import HookPoint

        cache: dict[str, torch.Tensor] = {}
        hooks: list[tuple[HookPoint, str]] = []
        visited: set[int] = set()

        def make_cache_hook(name: str) -> Callable[[torch.Tensor, Any], torch.Tensor]:
            def cache_hook(tensor: torch.Tensor, hook: Any) -> torch.Tensor:
                cache[name] = tensor.detach().cpu()
                return tensor
            return cache_hook

        # Recursively collect all HookPoint objects and their names
        def collect_hookpoints(module: nn.Module, prefix: str = "") -> None:
            obj_id = id(module)
            if obj_id in visited:
                return
            visited.add(obj_id)
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue
                name = f"{prefix}.{attr_name}" if prefix else attr_name
                if isinstance(attr, HookPoint):
                    hooks.append((attr, name))
                elif isinstance(attr, nn.Module):
                    collect_hookpoints(attr, name)
                elif isinstance(attr, (list, tuple)):
                    for i, item in enumerate(attr):
                        if isinstance(item, nn.Module):
                            collect_hookpoints(item, f"{name}[{i}]")
        collect_hookpoints(self)

        # Register hooks
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))

        try:
            # Run the underlying model's forward method
            output = self.model(*args, **kwargs)
        finally:
            # Remove hooks
            for hp, _ in hooks:
                hp.remove_hooks()
        return output, cache 

    @property
    def blocks(self):
        # Use the adapter to get the blocks component, for flexibility
        return self.adapter.get_component(self.model, "blocks") 