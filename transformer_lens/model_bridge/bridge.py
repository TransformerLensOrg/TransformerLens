"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
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
from jaxtyping import Float
from torch import nn

from transformer_lens import utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_setup import set_original_components
from transformer_lens.model_bridge.exceptions import StopAtLayerException
from transformer_lens.model_bridge.types import ComponentMapping
from transformer_lens.utilities.aliases import collect_aliases_recursive

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
    hook_aliases = {
        "hook_embed": "embed.hook_out",
        "hook_pos_embed": "pos_embed.hook_out",
    }

    def __init__(self, model: nn.Module, adapter: ArchitectureAdapter, tokenizer: Any):
        """Initialize the bridge.

        Args:
            model: The model to bridge (must be a PyTorch nn.Module or PreTrainedModel)
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
        """
        super().__init__()
        self.original_model: nn.Module = model
        self.adapter = adapter
        self.cfg = adapter.cfg
        self.tokenizer = tokenizer
        self.compatibility_mode = False

        # Add device information to config from the loaded model
        if not hasattr(self.cfg, "device"):
            try:
                self.cfg.device = next(self.original_model.parameters()).device
            except StopIteration:
                self.cfg.device = "cpu"

        if not hasattr(adapter, "component_mapping") or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")

        # Set original components on the pre-created bridge components
        set_original_components(self, self.adapter, self.original_model)

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        """Get all HookPoint objects in the model for compatibility with HookedTransformer."""
        hooks = {}
        visited = set()  # Move visited set outside the recursive function

        def collect_hookpoints(module: nn.Module, prefix: str = "") -> None:
            """Recursively collect all HookPoint objects."""
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
                    # Set the name on the HookPoint so it can be used in caching
                    attr.name = name
                    hooks[name] = attr
                elif isinstance(attr, nn.Module) and attr is not module:
                    collect_hookpoints(attr, name)
                elif isinstance(attr, (list, tuple)):
                    for i, item in enumerate(attr):
                        if isinstance(item, nn.Module):
                            collect_hookpoints(item, f"{name}[{i}]")

            # Also traverse named_children() to catch ModuleList and other containers
            for child_name, child_module in module.named_children():
                child_path = f"{prefix}.{child_name}" if prefix else child_name
                collect_hookpoints(child_module, child_path)

        collect_hookpoints(self, "")
        return hooks

    def __getattr__(self, name: str) -> Any:
        """Provide a clear error message for missing attributes."""
        if name in self.__dict__:
            return self.__dict__[name]

        return super().__getattr__(name)

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

    def enable_compatibility_mode(self, disable_warnings: bool = False) -> None:
        """Enable compatibility mode for the bridge.

        This sets up the bridge to work with legacy HookedTransformer components/hooks.
        It will also disable warnings about the usage of legacy components/hooks if specified.

        Args:
            disable_warnings: Whether to disable warnings about legacy components/hooks
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
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the key weights across all layers."""
        return torch.stack([block.attn.W_K.weight for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the query weights across all layers."""
        return torch.stack([block.attn.W_Q.weight for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stack the value weights across all layers."""
        return torch.stack([block.attn.W_V.weight for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stack the attn output weights across all layers."""
        return torch.stack([block.attn.W_O.weight for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stack the MLP input weights across all layers."""
        return torch.stack([block.mlp.W_in.weight for block in self.blocks], dim=0)

    @property
    def W_gate(self) -> Union[Float[torch.Tensor, "n_layers d_model d_mlp"], None]:
        """Stack the MLP gate weights across all layers.

        Only works for models with gated MLPs.
        """
        if self.cfg.gated_mlp:
            return torch.stack([block.mlp.W_gate.weight for block in self.blocks], dim=0)
        else:
            return None

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stack the MLP output weights across all layers."""
        return torch.stack([block.mlp.W_out.weight for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the key biases across all layers."""
        return torch.stack([block.attn.b_K.bias for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the query biases across all layers."""
        return torch.stack([block.attn.b_Q.bias for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stack the value biases across all layers."""
        return torch.stack([block.attn.b_V.bias for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stack the attn output biases across all layers."""
        return torch.stack([block.attn.b_O.bias for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stack the MLP input biases across all layers."""
        return torch.stack([block.mlp.b_in.bias for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stack the MLP output biases across all layers."""
        return torch.stack([block.mlp.b_out.bias for block in self.blocks], dim=0)

    @property
    def QK(self):
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self):
        return FactoredMatrix(self.W_V, self.W_O)

    # ==================== FORWARD PASS METHODS ====================

    def forward(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_type: str = "logits",
        loss_per_token: bool = False,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Forward pass through the model.

        Args:
            input: Input to the model
            return_type: Type of output to return ('logits', 'loss', 'both', None)
            loss_per_token: Whether to return loss per token
            prepend_bos: Whether to prepend BOS token
            padding_side: Which side to pad on
            **kwargs: Additional arguments passed to model

        Returns:
            Model output based on return_type
        """
        # Handle string input
        if isinstance(input, (str, list)):
            input_ids = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            input_ids = input

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

        # Handle different return types
        if return_type == "logits":
            if hasattr(output, "logits"):
                return output.logits
            return output
        elif return_type == "loss":
            if hasattr(output, "loss"):
                return output.loss
            # Calculate loss manually if needed
            logits = output.logits if hasattr(output, "logits") else output
            calculated_loss = self.loss_fn(logits, input_ids)
            return calculated_loss
        elif return_type == "both":
            logits = output.logits if hasattr(output, "logits") else output
            loss = output.loss if hasattr(output, "loss") else self.loss_fn(logits, input_ids)
            return logits, loss
        else:
            return output

    def loss_fn(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        per_token: bool = False,
    ) -> torch.Tensor:
        """Calculate cross-entropy loss.

        Args:
            logits: Model logits
            tokens: Target tokens
            attention_mask: Attention mask
            per_token: Whether to return per-token loss

        Returns:
            Loss tensor
        """
        # Simple cross-entropy loss implementation
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)

        # Shift tokens for next-token prediction
        target_tokens = tokens[:, 1:]
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
        ...

    @overload
    def run_with_cache(
        self,
        input: Union[str, List[str], torch.Tensor],
        return_cache_object: Literal[False],
        remove_batch_dim: bool = False,
        **kwargs,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        ...

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
        aliases = collect_aliases_recursive(self)

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

        # Recursively collect all HookPoint objects
        def collect_hookpoints(module: nn.Module, prefix: str = "") -> None:
            obj_id = id(module)
            if obj_id in visited:
                return
            visited.add(obj_id)

            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                # Skip the original_model to avoid collecting hooks from HuggingFace model
                if attr_name == "original_model":
                    continue
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue

                name = f"{prefix}.{attr_name}" if prefix else attr_name
                if isinstance(attr, HookPoint):
                    # Set the name on the HookPoint so it can be used in caching
                    attr.name = name
                    # Only add hook if it passes the names filter
                    if names_filter_fn(name):
                        hooks.append((attr, name))
                elif isinstance(attr, nn.Module):
                    collect_hookpoints(attr, name)
                elif isinstance(attr, (list, tuple)):
                    for i, item in enumerate(attr):
                        if isinstance(item, nn.Module):
                            collect_hookpoints(item, f"{name}[{i}]")

            # Also traverse named_children() to catch ModuleList and other containers
            for child_name, child_module in module.named_children():
                child_path = f"{prefix}.{child_name}" if prefix else child_name
                # Skip the original_model module
                if child_name == "original_model":
                    continue
                collect_hookpoints(child_module, child_path)

        # Collect hooks from bridge components (these have the clean TransformerLens paths)
        collect_hookpoints(self, "")

        # Register hooks
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))

        try:
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

                output = self.original_model(*processed_args, **filtered_kwargs)
                # Extract logits if output is a HuggingFace model output object
                if hasattr(output, "logits"):
                    output = output.logits
            except StopAtLayerException as e:
                # Return the intermediate output from the specified layer
                output = e.layer_output

        finally:
            for hp, _ in hooks:
                hp.remove_hooks()

        if self.compatibility_mode == True:
            # If compatibility mode is enabled, we need to handle aliases
            # Create duplicate cache entries for TransformerLens compatibility
            # Use the aliases collected from components (reverse mapping: new -> old)
            reverse_aliases = {new_name: old_name for old_name, new_name in aliases.items()}

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
                if target_name in cache and alias_name not in cache:
                    cache[alias_name] = cache[target_name]

        if return_cache_object:
            cache_obj = ActivationCache(cache, self, has_batch_dim=not remove_batch_dim)
            return output, cache_obj
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
            **kwargs: Additional arguments

        Returns:
            Model output
        """
        from transformer_lens.hook_points import HookPoint

        # Store hooks that we add so we can remove them later
        added_hooks: List[Tuple[HookPoint, str]] = []

        def add_hook_to_point(hook_point: HookPoint, hook_fn: Callable, name: str):
            hook_point.add_hook(hook_fn)
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
                    add_hook_to_point(hook_dict[block_hook_name], stop_hook, block_hook_name)

        # Helper function to apply hooks based on name or filter function
        def apply_hooks(hooks: List[Tuple[Union[str, Callable], Callable]], is_fwd: bool):
            # Collect aliases for resolving legacy hook names
            aliases = collect_aliases_recursive(self)

            for hook_name_or_filter, hook_fn in hooks:
                if isinstance(hook_name_or_filter, str):
                    # Direct hook name - check for aliases first
                    hook_dict = self.hook_dict
                    actual_hook_name = hook_name_or_filter

                    # If this is an alias, resolve it to the actual hook name
                    if hook_name_or_filter in aliases:
                        actual_hook_name = aliases[hook_name_or_filter]

                    if actual_hook_name in hook_dict:
                        add_hook_to_point(hook_dict[actual_hook_name], hook_fn, actual_hook_name)
                else:
                    # Filter function
                    hook_dict = self.hook_dict
                    for name, hook_point in hook_dict.items():
                        if hook_name_or_filter(name):
                            add_hook_to_point(hook_point, hook_fn, name)

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
        """Generate text from the model.

        Args:
            input: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            stop_at_eos: Whether to stop at EOS token
            eos_token_id: EOS token ID
            do_sample: Whether to sample from distribution
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Sampling temperature
            freq_penalty: Frequency penalty
            use_past_kv_cache: Whether to use KV cache
            prepend_bos: Whether to prepend BOS token
            padding_side: Which side to pad on
            return_type: Type of output to return
            verbose: Whether to show progress

        Returns:
            Generated text or tokens
        """
        # Use the underlying model's generate method if available
        if hasattr(self.original_model, "generate"):
            # Tokenize input if needed
            if isinstance(input, (str, list)):
                input_ids = self.to_tokens(
                    input, prepend_bos=prepend_bos, padding_side=padding_side
                )
            else:
                input_ids = input

            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

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

            if not hasattr(self.original_model, "generate"):
                raise RuntimeError("Underlying model does not support generate method.")
            output_ids = self.original_model.generate(input_ids, **gen_kwargs)  # type: ignore[operator]

            # Handle return type
            if return_type == "input":
                if isinstance(input, (str, list)):
                    return_type = "str"
                else:
                    return_type = "tokens"

            if return_type == "str":
                decoded_texts = [
                    self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_ids
                ]
                return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts
            elif return_type == "tokens":
                return output_ids
            else:
                return output_ids
        else:
            raise RuntimeError("Underlying model does not support generate method.")

    # ==================== UTILITY METHODS ====================

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

    def mps(self) -> "TransformerBridge":
        """Move model to MPS.

        Returns:
            Self for chaining
        """
        return self.to(torch.device("mps"))  # type: ignore
