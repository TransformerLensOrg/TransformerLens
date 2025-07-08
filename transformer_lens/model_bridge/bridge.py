"""Bridge module for connecting different model architectures.

This module provides the bridge components that wrap remote model components and provide
a consistent interface for accessing their weights and performing operations.
"""

from typing import (
    TYPE_CHECKING,
    Any,
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
import torch.nn as nn
import os
import logging

from transformer_lens import utils

from transformers import PreTrainedTokenizerBase, AutoTokenizer
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.component_creation import (
    create_and_replace_components_from_mapping,
)

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache


class TransformerBridge(nn.Module):
    """Bridge between HuggingFace and HookedTransformer models.

    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the HookedTransformer and HuggingFace model structures.
    """

    def __init__(self, model: nn.Module, adapter: ArchitectureAdapter, tokenizer: Any):
        """Initialize the bridge.

        Args:
            model: The model to bridge (must be a PyTorch nn.Module or PreTrainedModel)
            adapter: The architecture adapter to use
            tokenizer: The tokenizer to use (required)
        """
        super().__init__()
        self.original_model: nn.Module = model
        self.bridge = adapter
        self.cfg = adapter.user_cfg
        self.tokenizer = tokenizer

        tokenizer_name = self.cfg.model_type
        default_padding_side = None # TODO figure out

        if self.tokenizer is not None:
            self.set_tokenizer(self.tokenizer, default_padding_side=default_padding_side)
        elif tokenizer_name is not None:
            # If we have a tokenizer name, we can load it from HuggingFace
            if tokenizer_name in "":
                logging.warning(
                    "%s tokenizer not loaded. Please load manually.",
                    tokenizer_name,
                )
            else:
                # Hugging Face defaults to use_fast to True
                use_fast = True
                # Phi model's fast tokenizer does not support adding a BOS token, use_fast
                # should be False
                if "phi" in tokenizer_name.lower():
                    use_fast = False
                huggingface_token = os.environ.get("HF_TOKEN", "")
                self.set_tokenizer(
                    AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        add_bos_token=True,
                        trust_remote_code=True,  # TODO figure out,
                        use_fast=use_fast,
                        token=huggingface_token if len(huggingface_token) > 0 else None,
                    ),
                    default_padding_side=default_padding_side,
                )
        else:
            # If no tokenizer name is provided, we assume we're training on an algorithmic task and
            # will pass in tokens directly. In this case, we don't need a tokenizer.
            assert self.cfg.d_vocab != -1, "Must provide a tokenizer if d_vocab is not provided"
            self.tokenizer = None
            if default_padding_side != None:
                logging.warning(
                    "default_padding_side is explicitly given but ignored because tokenizer is not set."
                )

        if not hasattr(adapter, "component_mapping") or adapter.component_mapping is None:
            raise ValueError("Adapter must have a component_mapping attribute")

        # Create and replace components
        create_and_replace_components_from_mapping(
            self.bridge.get_component_mapping(), self.original_model, self.bridge, bridge=self
        )

    def set_tokenizer(
        self,
        tokenizer,
        default_padding_side=None,
    ):
        """Set the tokenizer to use for this model.

        Args:
            tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer.
            default_padding_side (str): "right" or "left", which side to pad on.

        """
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"

        assert default_padding_side in [
            "right",
            "left",
            None,
        ], f"padding_side must be 'right', 'left' or 'None', got {default_padding_side}"

        # Use a tokenizer that is initialized with add_bos_token=True as the default tokenizer.
        # Such a tokenizer should be set as the default tokenizer because the tokenization of some
        # tokenizers like LlamaTokenizer are different when bos token is automatically/manually
        # prepended, and add_bos_token cannot be dynamically controlled after initialization
        # (https://github.com/huggingface/transformers/issues/25886).
        tokenizer_with_bos = utils.get_tokenizer_with_bos(tokenizer)
        self.tokenizer = tokenizer_with_bos
        assert self.tokenizer is not None  # keep mypy happy

        # If user passes default_padding_side explicitly, use that value
        if default_padding_side is not None:
            self.tokenizer.padding_side = default_padding_side
        # If not, then use the tokenizer's default padding side
        # If the tokenizer doesn't have a default padding side, use the global default "right"
        if self.tokenizer.padding_side is None:
            self.tokenizer.padding_side = "right"

        # Some tokenizers doesn't automatically prepend the BOS token even when they are initialized
        # with add_bos_token=True. Therefore, we need this information to dynamically control prepend_bos.
        self.cfg.tokenizer_prepends_bos = len(self.tokenizer.encode("")) > 0

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

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
            comp = self.bridge.get_component(self.original_model, path)
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
        assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"

        # Handle prepend_bos logic
        if prepend_bos is None:
            prepend_bos = getattr(self.cfg, "default_prepend_bos", True)

        # Handle padding_side logic
        if padding_side is None:
            padding_side = getattr(self.tokenizer, "padding_side", "right")

        if prepend_bos and not self.cfg.tokenizer_prepends_bos:
                # We want to prepend bos but the tokenizer doesn't automatically do it, so we add it manually
                input = utils.get_input_with_manually_prepended_bos(self.tokenizer, input)

        # Tokenize
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=getattr(self.cfg, "n_ctx", None) if truncate else None,
        )["input_ids"]

        if move_to_device:
            device = next(self.original_model.parameters()).device
            tokens = tokens.to(device)

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
        assert self.tokenizer is not None, "Cannot use to_string without a tokenizer"

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
        assert self.tokenizer is not None, "Cannot use to_str_tokens without a tokenizer"

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
            output = self.original_model.forward(input_ids, **kwargs)
        else:
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
            return self.loss_fn(output.logits if hasattr(output, "logits") else output, input_ids)
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
        **kwargs,
    ) -> Tuple[Any, Union[ActivationCache, Dict[str, torch.Tensor]]]:
        """Run the model and cache all activations.

        Args:
            input: Input to the model
            return_cache_object: Whether to return ActivationCache object
            remove_batch_dim: Whether to remove batch dimension
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, cache)
        """
        from transformer_lens.hook_points import HookPoint

        cache: Dict[str, torch.Tensor] = {}
        hooks: List[Tuple[HookPoint, str]] = []
        visited: set[int] = set()

        def make_cache_hook(name: str):
            def cache_hook(tensor: torch.Tensor, *, hook: Any) -> torch.Tensor:
                cache[name] = tensor.detach().cpu()
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

        collect_hookpoints(self.original_model)

        # Register hooks
        for hp, name in hooks:
            hp.add_hook(make_cache_hook(name))

        try:
            processed_args = [input]
            # Handle string input whether passed positionally or as a kwarg
            if processed_args and isinstance(processed_args[0], str):
                assert self.tokenizer is not None, "Tokenizer must be set to pass string input."
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                input_ids = self.tokenizer(
                    processed_args[0],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )["input_ids"]
                input_ids = input_ids.to(next(self.original_model.parameters()).device)
                kwargs["input_ids"] = input_ids
                processed_args = processed_args[1:]
            elif "input" in kwargs and isinstance(kwargs["input"], str):
                assert self.tokenizer is not None, "Tokenizer must be set to pass string input."
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                input_ids = self.tokenizer(
                    kwargs["input"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )["input_ids"]
                input_ids = input_ids.to(next(self.original_model.parameters()).device)
                kwargs["input_ids"] = input_ids
                del kwargs["input"]

            # Run the underlying model's forward method
            output = self.original_model(*processed_args, **kwargs)

            # Extract logits if output is a HuggingFace model output object
            if hasattr(output, "logits"):
                output = output.logits

        finally:
            # Remove hooks
            for hp, _ in hooks:
                hp.remove_hooks()

        if return_cache_object:
            cache_obj = ActivationCache(cache, self, has_batch_dim=not remove_batch_dim)
            return output, cache_obj
        else:
            return output, cache

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

    @property
    def blocks(self):
        # Use the adapter to get the blocks component, for flexibility
        return self.bridge.get_component(self.original_model, "blocks")
