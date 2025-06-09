"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""

from typing import Any, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_creation import (
    create_and_replace_components_from_mapping,
)


class TransformerBridge:
    """Bridge between HuggingFace and HookedTransformer models.

    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the HookedTransformer and HuggingFace model structures.
    """

    def __init__(
        self, model: Union[nn.Module, PreTrainedModel], adapter: ArchitectureAdapter, tokenizer: Any
    ):
        """Initialize the bridge.

        Args:
            model: The model to bridge (must be a PyTorch nn.Module or PreTrainedModel)
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
        """
        super().__init__()
        self.model = model
        self.bridge = adapter
        self.cfg = adapter.user_cfg
        self.tokenizer = tokenizer

        if not hasattr(adapter, "component_mapping") or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")

        # Create and replace components
        create_and_replace_components_from_mapping(
            self.bridge.get_component_mapping(), self.model, self.bridge, bridge=self
        )

    def __getattr__(self, name: str) -> Any:
        """Provide a clear error message for missing attributes."""
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Check the component mapping in your architecture adapter."
        )

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
            comp = self.bridge.get_component(self.model, path)
            if hasattr(comp, "original_component"):
                return f"{indent_str}{name}: {type(comp).__name__}({type(comp.original_component).__name__})"
            return f"{indent_str}{name}: {type(comp).__name__}"
        except Exception as e:
            return f"{indent_str}{name}: <error: {e}>"

    def _format_component_mapping(
        self, mapping: dict, indent: int = 0, prepend: str | None = None
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
            if isinstance(value, tuple):
                # Handle both 2-tuple (RemoteImport) and 3-tuple (BlockMapping) structures
                if len(value) == 3:
                    # This is a BlockMapping (path, bridge_type, sub_mapping)
                    _, _, sub_mapping = value
                    if isinstance(sub_mapping, dict):
                        # This is a BlockMapping (like blocks) - format recursively
                        path = f"{path}.0"
                        lines.append(self._format_single_component(name, path, indent))
                        # Recursively format subcomponents with updated prepend
                        sub_lines = self._format_component_mapping(sub_mapping, indent + 1, path)
                        lines.extend(sub_lines)
                    else:
                        # This should not happen with BlockMapping
                        lines.append(self._format_single_component(name, path, indent))
                elif len(value) == 2:
                    # This is a RemoteImport (path, bridge_type) - format as single component
                    lines.append(self._format_single_component(name, path, indent))
                else:
                    # Unknown tuple structure
                    lines.append(self._format_single_component(name, path, indent))
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
        mapping = self.bridge.get_component_mapping()
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

            # Fix padding token issue for GPT2 tokenizers
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
            output_ids = self.model.generate(input_ids, **gen_kwargs)  # type: ignore[operator,arg-type]
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
            decoded_texts = [
                self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_ids
            ]
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

        def make_cache_hook(name: str):
            def cache_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
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

        collect_hookpoints(self.model)

        # Register hooks
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))

        try:
            # Process input - if it's a string, tokenize it first
            processed_args = list(args)
            if len(args) > 0 and isinstance(args[0], str):
                assert self.tokenizer is not None, "Tokenizer must be set to pass string input."

                # Fix padding token issue for GPT2 tokenizers
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                input_ids = self.tokenizer(
                    args[0],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )["input_ids"]
                input_ids = input_ids.to(next(self.model.parameters()).device)
                processed_args[0] = input_ids

            # Run the underlying model's forward method
            output = self.model(*processed_args, **kwargs)

            # Extract logits if output is a HuggingFace model output object
            if hasattr(output, "logits"):
                output = output.logits

        finally:
            # Remove hooks
            for hp, _ in hooks:
                hp.remove_hooks()
        return output, cache

    def blocks(self):
        # Use the adapter to get the blocks component, for flexibility
        return self.bridge.get_component(self.model, "blocks")
