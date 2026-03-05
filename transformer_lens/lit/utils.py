"""Utility functions for the LIT integration module.

This module provides helper functions for converting between TransformerLens
data structures and LIT-compatible formats, as well as other utilities.

References:
    - LIT API: https://pair-code.github.io/lit/documentation/api
    - TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def check_lit_installed() -> bool:
    """Check if LIT (lit-nlp) is installed.

    Returns:
        bool: True if LIT is installed, False otherwise.
    """
    try:
        import lit_nlp  # noqa: F401

        return True
    except ImportError:
        return False


def tensor_to_numpy(
    tensor: Union[torch.Tensor, np.ndarray, None],
) -> Optional[np.ndarray]:
    """Convert a PyTorch tensor to a NumPy array.

    LIT expects all data to be in NumPy format, so this helper ensures
    proper conversion with detach and CPU transfer.

    Args:
        tensor: PyTorch tensor or None.

    Returns:
        NumPy array or None if input was None.
    """
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")


def numpy_to_tensor(
    array: Union[np.ndarray, torch.Tensor, None],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Convert a NumPy array to a PyTorch tensor.

    Args:
        array: NumPy array or None.
        device: Target device for the tensor.
        dtype: Target dtype for the tensor.

    Returns:
        PyTorch tensor or None if input was None.
    """
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.from_numpy(array)

    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_tokens_from_model(
    model: Any,
    text: str,
    prepend_bos: bool = True,
    truncate: bool = True,
    max_length: Optional[int] = None,
) -> Tuple[List[str], torch.Tensor]:
    """Get tokens and token IDs from a HookedTransformer model.

    Args:
        model: HookedTransformer model with tokenizer.
        text: Input text to tokenize.
        prepend_bos: Whether to prepend the BOS token.
        truncate: Whether to truncate to max_length.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (token strings, token ID tensor).

    Raises:
        ValueError: If model has no tokenizer.
    """
    if model.tokenizer is None:
        raise ValueError("Model must have a tokenizer to convert text to tokens")

    # Get token IDs
    token_ids = model.to_tokens(text, prepend_bos=prepend_bos, truncate=truncate)

    if max_length is not None and token_ids.shape[1] > max_length:
        token_ids = token_ids[:, :max_length]

    # Convert IDs to strings
    token_strings = model.tokenizer.convert_ids_to_tokens(token_ids.squeeze(0).tolist())

    return token_strings, token_ids.squeeze(0)


def clean_token_string(token: str) -> str:
    """Clean a token string for display.

    Handles common tokenizer artifacts like:
    - Ġ (GPT-2 style space prefix)
    - ▁ (SentencePiece space prefix)
    - ## (BERT style subword prefix)

    Args:
        token: Raw token string from tokenizer.

    Returns:
        Cleaned token string for display.
    """
    # Handle GPT-2/RoBERTa style space encoding
    if token.startswith("Ġ"):
        return "▁" + token[1:]  # Use Unicode space indicator
    # Handle SentencePiece
    if token.startswith("▁"):
        return token  # Already in preferred format
    # Handle BERT style
    if token.startswith("##"):
        return token[2:]  # Remove ## prefix
    return token


def clean_token_strings(tokens: List[str]) -> List[str]:
    """Clean a list of token strings for display.

    Args:
        tokens: List of raw token strings.

    Returns:
        List of cleaned token strings.
    """
    return [clean_token_string(t) for t in tokens]


def extract_attention_from_cache(
    cache: Any,
    layer: int,
    head: Optional[int] = None,
    batch_idx: int = 0,
) -> Optional[np.ndarray]:
    """Extract attention patterns from an activation cache.

    Args:
        cache: TransformerLens ActivationCache object.
        layer: Layer index to extract from.
        head: Optional head index. If None, returns all heads.
        batch_idx: Batch index to extract.

    Returns:
        Attention pattern as numpy array.
        Shape: [query_pos, key_pos] if head specified
        Shape: [num_heads, query_pos, key_pos] if head is None
    """
    # Get attention pattern from cache
    attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]

    # Remove batch dimension
    if attn_pattern.dim() == 4:
        attn_pattern = attn_pattern[batch_idx]

    # attn_pattern shape: [num_heads, query_pos, key_pos]
    if head is not None:
        attn_pattern = attn_pattern[head]

    return tensor_to_numpy(attn_pattern)


def extract_embeddings_from_cache(
    cache: Any,
    layer: int,
    position: str = "all",
    batch_idx: int = 0,
) -> Optional[np.ndarray]:
    """Extract embeddings from a specific layer in the activation cache.

    Args:
        cache: TransformerLens ActivationCache object.
        layer: Layer index to extract from.
        position: "all" for all positions, "first" for CLS-like, "last" for final token.
        batch_idx: Batch index to extract.

    Returns:
        Embeddings as numpy array.
    """
    # Get residual stream at layer
    resid = cache[f"blocks.{layer}.hook_resid_post"]

    # Remove batch dimension
    if resid.dim() == 3:
        resid = resid[batch_idx]

    # resid shape: [seq_len, d_model]
    if position == "first":
        embeddings = resid[0]
    elif position == "last":
        embeddings = resid[-1]
    elif position == "mean":
        embeddings = resid.mean(dim=0)
    else:  # "all"
        embeddings = resid

    return tensor_to_numpy(embeddings)


def compute_token_gradients(
    model: Any,
    text: str,
    target_idx: Optional[int] = None,
    prepend_bos: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Compute token-level gradients for salience.

    Uses gradient of the loss with respect to token embeddings to compute
    importance scores for each token.

    Args:
        model: HookedTransformer model.
        text: Input text.
        target_idx: Target token index for gradient computation.
                   If None, uses the last token.
        prepend_bos: Whether to prepend BOS token.

    Returns:
        Tuple of (grad_l2, grad_dot_input, tokens) where:
        - grad_l2: L2 norm of gradients per token [seq_len]
        - grad_dot_input: Gradient dot input embedding per token [seq_len]
        - tokens: List of token strings
    """
    # Tokenize
    tokens, token_ids = get_tokens_from_model(model, text, prepend_bos=prepend_bos)
    token_ids = token_ids.unsqueeze(0).to(model.cfg.device)

    # Get input embeddings
    input_embeds = model.embed(token_ids)
    input_embeds.requires_grad_(True)

    # Forward pass
    logits = model(input_embeds, start_at_layer=0)

    # Determine target
    if target_idx is None:
        target_idx = -1  # Last token

    # Get target logit and compute gradient
    target_logit = logits[0, target_idx, token_ids[0, target_idx + 1]]
    target_logit.backward()

    # Get gradients
    gradients = input_embeds.grad[0]  # [seq_len, d_model]

    # Compute gradient L2 norm per token
    grad_l2 = torch.norm(gradients, dim=-1)  # [seq_len]

    # Compute gradient dot input
    grad_dot_input = (gradients * input_embeds[0].detach()).sum(dim=-1)  # [seq_len]

    return (
        tensor_to_numpy(grad_l2),
        tensor_to_numpy(grad_dot_input),
        tokens,
    )


def get_top_k_predictions(
    logits: torch.Tensor,
    tokenizer: Any,
    k: int = 10,
    position: int = -1,
    batch_idx: int = 0,
) -> List[Tuple[str, float]]:
    """Get top-k token predictions with their probabilities.

    Args:
        logits: Model logits tensor.
        tokenizer: HuggingFace tokenizer.
        k: Number of top predictions to return.
        position: Position index to get predictions for.
        batch_idx: Batch index.

    Returns:
        List of (token_string, probability) tuples.
    """
    # Get logits at position
    pos_logits = logits[batch_idx, position]  # [d_vocab]

    # Convert to probabilities
    probs = torch.softmax(pos_logits, dim=-1)

    # Get top-k
    top_probs, top_indices = torch.topk(probs, k)

    # Convert to strings
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, prob))

    return results


def validate_input_example(
    example: Dict[str, Any],
    required_fields: List[str],
) -> bool:
    """Validate that an input example has all required fields.

    Args:
        example: Input example dictionary.
        required_fields: List of required field names.

    Returns:
        True if valid, False otherwise.
    """
    for field in required_fields:
        if field not in example:
            logger.warning(f"Missing required field '{field}' in input example")
            return False
    return True


def batch_examples(
    examples: List[Dict[str, Any]],
    batch_size: int,
) -> List[List[Dict[str, Any]]]:
    """Split examples into batches.

    Args:
        examples: List of example dictionaries.
        batch_size: Size of each batch.

    Returns:
        List of batches, where each batch is a list of examples.
    """
    return [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]


def unbatch_outputs(
    batched_outputs: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Split batched outputs into individual examples.

    Takes a dictionary with batched arrays and returns a list of
    dictionaries with individual arrays.

    Args:
        batched_outputs: Dictionary mapping field names to batched arrays.

    Returns:
        List of dictionaries, one per example.
    """
    if not batched_outputs:
        return []

    # Get batch size from first array
    first_key = next(iter(batched_outputs))
    batch_size = len(batched_outputs[first_key])

    # Split into individual examples
    results = []
    for i in range(batch_size):
        example_output = {}
        for key, value in batched_outputs.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                example_output[key] = value[i]
            elif isinstance(value, list):
                example_output[key] = value[i]
            else:
                example_output[key] = value
        results.append(example_output)

    return results


def get_hook_name_for_layer(template: str, layer: int, **kwargs) -> str:
    """Generate a hook point name from a template.

    Args:
        template: Hook name template with {layer} placeholder.
        layer: Layer index.
        **kwargs: Additional template parameters.

    Returns:
        Formatted hook point name.
    """
    return template.format(layer=layer, **kwargs)


def filter_cache_by_pattern(
    cache: Any,
    pattern: str,
) -> Dict[str, torch.Tensor]:
    """Filter activation cache entries by hook name pattern.

    Args:
        cache: TransformerLens ActivationCache.
        pattern: Pattern to match (e.g., "attn.hook_pattern" will match
                all attention pattern hooks).

    Returns:
        Dictionary of matching cache entries.
    """
    return {name: value for name, value in cache.items() if pattern in name}


def get_model_info(model: Any) -> Dict[str, Any]:
    """Extract relevant model information for LIT display.

    Args:
        model: HookedTransformer model.

    Returns:
        Dictionary with model metadata.
    """
    cfg = model.cfg
    return {
        "model_name": cfg.model_name,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_model": cfg.d_model,
        "d_head": cfg.d_head,
        "d_mlp": cfg.d_mlp,
        "d_vocab": cfg.d_vocab,
        "n_ctx": cfg.n_ctx,
        "act_fn": cfg.act_fn,
        "normalization_type": cfg.normalization_type,
        "positional_embedding_type": cfg.positional_embedding_type,
    }
