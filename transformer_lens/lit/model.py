"""LIT Model wrapper for TransformerLens HookedTransformer.

This module provides a LIT-compatible wrapper around TransformerLens's HookedTransformer,
enabling the use of Google's Learning Interpretability Tool (LIT) for model visualization
and analysis.

The wrapper exposes:
- Token predictions (logits, top-k tokens)
- Per-layer embeddings (residual stream)
- Attention patterns (all layers/heads)
- Token gradients for salience maps
- Loss computation

Example usage:
    >>> from transformer_lens import HookedTransformer  # doctest: +SKIP
    >>> from transformer_lens.lit import HookedTransformerLIT  # doctest: +SKIP
    >>>
    >>> # Load model
    >>> model = HookedTransformer.from_pretrained("gpt2-small")  # doctest: +SKIP
    >>>
    >>> # Create LIT wrapper
    >>> lit_model = HookedTransformerLIT(model)  # doctest: +SKIP
    >>>
    >>> # Run prediction
    >>> inputs = [{"text": "Hello, world!"}]  # doctest: +SKIP
    >>> outputs = list(lit_model.predict(inputs))  # doctest: +SKIP

References:
    - LIT Model API: https://pair-code.github.io/lit/documentation/api#models
    - TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Union

import torch

from .constants import DEFAULTS, ERRORS, INPUT_FIELDS, OUTPUT_FIELDS
from .utils import (
    check_lit_installed,
    clean_token_strings,
    extract_attention_from_cache,
    get_model_info,
    get_tokens_from_model,
    tensor_to_numpy,
)

if TYPE_CHECKING:
    from lit_nlp.api import model as lit_model_types  # noqa: F401
    from lit_nlp.api import types as lit_types_module  # noqa: F401

# Check for LIT installation and import conditionally
if check_lit_installed():
    from lit_nlp.api import (  # type: ignore[import-not-found]  # noqa: F401
        model as lit_model,
    )
    from lit_nlp.api import (  # type: ignore[import-not-found]  # noqa: F401
        types as lit_types,
    )
    from lit_nlp.lib import utils as lit_utils  # type: ignore[import-not-found]

    _LIT_AVAILABLE = True
else:
    _LIT_AVAILABLE = False
    # Create placeholder when LIT not installed
    lit_model = None  # type: ignore[assignment]
    lit_types = None  # type: ignore[assignment]
    lit_utils = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class HookedTransformerLITConfig:
    """Configuration for the HookedTransformerLIT wrapper."""

    max_seq_length: int = DEFAULTS.MAX_SEQ_LENGTH
    batch_size: int = DEFAULTS.BATCH_SIZE
    top_k: int = DEFAULTS.TOP_K
    compute_gradients: bool = DEFAULTS.COMPUTE_GRADIENTS
    output_attention: bool = DEFAULTS.OUTPUT_ATTENTION
    output_embeddings: bool = DEFAULTS.OUTPUT_EMBEDDINGS
    output_all_layers: bool = DEFAULTS.OUTPUT_ALL_LAYERS
    embedding_layers: Optional[List[int]] = None
    prepend_bos: bool = DEFAULTS.PREPEND_BOS
    device: Optional[str] = None


def _ensure_lit_available():
    """Raise ImportError if LIT is not available."""
    if not _LIT_AVAILABLE:
        raise ImportError(ERRORS.LIT_NOT_INSTALLED)


# Create base class dynamically based on LIT availability
if _LIT_AVAILABLE:
    _LITModelBase = lit_model.Model
else:
    _LITModelBase = object  # type: ignore[misc,assignment]


class HookedTransformerLIT(_LITModelBase):  # type: ignore[valid-type,misc]
    """LIT Model wrapper for TransformerLens HookedTransformer.

    This wrapper implements the LIT Model API, enabling the use of LIT's
    visualization and analysis tools with TransformerLens models.

    The wrapper provides:
    - Token predictions with top-k probabilities
    - Per-layer embeddings for embedding projector
    - Attention patterns for attention visualization
    - Token gradients for salience maps

    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")  # doctest: +SKIP
        >>> lit_model = HookedTransformerLIT(model)  # doctest: +SKIP
        >>> lit_model.input_spec()  # doctest: +SKIP
        {'text': TextSegment(), ...}
    """

    def __init__(
        self,
        model: Any,
        config: Optional[HookedTransformerLITConfig] = None,
    ):
        """Initialize the LIT wrapper.

        Args:
            model: TransformerLens HookedTransformer model.
            config: Optional configuration. Uses defaults if not provided.

        Raises:
            ImportError: If lit-nlp is not installed.
            TypeError: If model is not a HookedTransformer.
        """
        _ensure_lit_available()

        # Validate model type
        from transformer_lens import HookedTransformer

        if not isinstance(model, HookedTransformer):
            raise TypeError(ERRORS.INVALID_MODEL.format(model_type=type(model)))

        self.model = model
        self.config = config or HookedTransformerLITConfig()

        # Gradients require embeddings to be output (for alignment)
        if self.config.compute_gradients and not self.config.output_embeddings:
            logger.info("Enabling output_embeddings (required for compute_gradients)")
            self.config.output_embeddings = True

        # Set device
        if self.config.device is None:
            self.config.device = str(model.cfg.device)

        # Cache model info
        self._model_info = get_model_info(model)

        logger.info(f"Created HookedTransformerLIT wrapper for {self._model_info['model_name']}")

    @property
    def supports_concurrent_predictions(self) -> bool:
        """Whether this model supports concurrent predictions.

        Returns False as PyTorch models typically aren't thread-safe.
        """
        return False

    def description(self) -> str:
        """Return a human-readable description of the model.

        Returns:
            Model description string.
        """
        info = self._model_info
        return (
            f"TransformerLens: {info['model_name']} "
            f"({info['n_layers']}L, {info['n_heads']}H, d={info['d_model']})"
        )

    @classmethod
    def init_spec(cls) -> Dict[str, Any]:
        """Return spec for model initialization in LIT UI.

        This allows loading new models through the LIT interface.

        Returns:
            Specification for initialization parameters.
        """
        _ensure_lit_available()
        return {
            "model_name": lit_types.String(  # type: ignore[union-attr]
                default="gpt2-small",
                required=True,
            ),
            "max_seq_length": lit_types.Integer(  # type: ignore[union-attr]
                default=DEFAULTS.MAX_SEQ_LENGTH,
                min_val=1,
                max_val=2048,
                required=False,
            ),
            "compute_gradients": lit_types.Boolean(  # type: ignore[union-attr]
                default=DEFAULTS.COMPUTE_GRADIENTS,
                required=False,
            ),
            "output_attention": lit_types.Boolean(  # type: ignore[union-attr]
                default=DEFAULTS.OUTPUT_ATTENTION,
                required=False,
            ),
            "output_embeddings": lit_types.Boolean(  # type: ignore[union-attr]
                default=DEFAULTS.OUTPUT_EMBEDDINGS,
                required=False,
            ),
        }

    def input_spec(self) -> Dict[str, Any]:
        """Return spec describing the model inputs.

        Defines the expected input format for the model. LIT uses this
        to validate inputs and generate appropriate UI controls.

        Returns:
            Dictionary mapping field names to LIT type specs.
        """
        _ensure_lit_available()

        spec = {
            # Primary text input
            INPUT_FIELDS.TEXT: lit_types.TextSegment(),  # type: ignore[union-attr]
            # Optional pre-tokenized input (for Integrated Gradients)
            INPUT_FIELDS.TOKENS: lit_types.Tokens(  # type: ignore[union-attr]
                parent=INPUT_FIELDS.TEXT,
                required=False,
            ),
        }

        # Add optional embeddings input for Integrated Gradients
        if self.config.output_embeddings:
            spec[INPUT_FIELDS.TOKEN_EMBEDDINGS] = lit_types.TokenEmbeddings(  # type: ignore[union-attr]
                align=INPUT_FIELDS.TOKENS,
                required=False,
            )

        # Add target mask for sequence salience
        if self.config.compute_gradients:
            spec[INPUT_FIELDS.TARGET_MASK] = lit_types.Tokens(  # type: ignore[union-attr]
                parent=INPUT_FIELDS.TEXT,
                required=False,
            )

        return spec

    def output_spec(self) -> Dict[str, Any]:
        """Return spec describing the model outputs.

        Defines all the outputs that the model produces. LIT uses this
        to determine which visualizations to show.

        Returns:
            Dictionary mapping field names to LIT type specs.
        """
        _ensure_lit_available()

        spec = {}

        # Tokens (always output)
        spec[OUTPUT_FIELDS.TOKENS] = lit_types.Tokens(  # type: ignore[union-attr]
            parent=INPUT_FIELDS.TEXT,
        )

        # Top-K predictions for next token
        spec[OUTPUT_FIELDS.TOP_K_TOKENS] = lit_types.TokenTopKPreds(  # type: ignore[union-attr]
            align=OUTPUT_FIELDS.TOKENS,
        )

        # Embeddings
        if self.config.output_embeddings:
            # Input embeddings (for Integrated Gradients)
            spec[OUTPUT_FIELDS.INPUT_EMBEDDINGS] = lit_types.TokenEmbeddings(  # type: ignore[union-attr]
                align=OUTPUT_FIELDS.TOKENS,
            )

            # Final layer embedding (CLS-style)
            spec[OUTPUT_FIELDS.CLS_EMBEDDING] = lit_types.Embeddings()  # type: ignore[union-attr]

            # Mean pooled embedding
            spec[OUTPUT_FIELDS.MEAN_EMBEDDING] = lit_types.Embeddings()  # type: ignore[union-attr]

            # Per-layer embeddings
            layers_to_output = self._get_embedding_layers()
            for layer in layers_to_output:
                field_name = OUTPUT_FIELDS.LAYER_EMB_TEMPLATE.format(layer=layer)
                spec[field_name] = lit_types.Embeddings()  # type: ignore[union-attr]

        # Attention patterns
        if self.config.output_attention:
            for layer in range(self._model_info["n_layers"]):
                field_name = OUTPUT_FIELDS.LAYER_ATTENTION_TEMPLATE.format(layer=layer)
                spec[field_name] = lit_types.AttentionHeads(  # type: ignore[union-attr]
                    align_in=OUTPUT_FIELDS.TOKENS,
                    align_out=OUTPUT_FIELDS.TOKENS,
                )

        # Gradients for salience
        if self.config.compute_gradients:
            # TokenGradients spec requirements (per LIT API):
            # - align: must point to a Tokens field (for token alignment)
            # - grad_for: must point to a TokenEmbeddings field (for grad-dot-input)
            # LIT's GradientNorm component computes L2 norm internally
            # LIT's GradientDotInput component computes dot product with embeddings
            spec[OUTPUT_FIELDS.GRAD_L2] = lit_types.TokenGradients(  # type: ignore[union-attr]
                align=OUTPUT_FIELDS.TOKENS,
                grad_for=OUTPUT_FIELDS.INPUT_EMBEDDINGS,
            )
            # Gradient dot input uses same format
            spec[OUTPUT_FIELDS.GRAD_DOT_INPUT] = lit_types.TokenGradients(  # type: ignore[union-attr]
                align=OUTPUT_FIELDS.TOKENS,
                grad_for=OUTPUT_FIELDS.INPUT_EMBEDDINGS,
            )

        return spec

    def _get_embedding_layers(self) -> List[int]:
        """Get the layers to output embeddings for.

        Returns:
            List of layer indices.
        """
        if self.config.embedding_layers is not None:
            return self.config.embedding_layers

        n_layers = self._model_info["n_layers"]

        if self.config.output_all_layers:
            return list(range(n_layers))
        else:
            # Output first, middle, and last layers by default
            if n_layers <= 3:
                return list(range(n_layers))
            return [0, n_layers // 2, n_layers - 1]

    def predict(
        self,
        inputs: Iterable[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        """Run prediction on a sequence of inputs.

        This is the main entry point for LIT to get model outputs.

        Args:
            inputs: Iterable of input dictionaries, each with fields
                   matching input_spec().

        Yields:
            Output dictionaries for each input, with fields matching
            output_spec().
        """
        for example in inputs:
            yield self._predict_single(example)

    def _predict_single(
        self,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run prediction on a single example.

        Args:
            example: Input dictionary with text field.

        Returns:
            Output dictionary with predictions.
        """
        text = example[INPUT_FIELDS.TEXT]

        # Check for pre-tokenized input (reserved for future use)
        _ = example.get(INPUT_FIELDS.TOKENS)
        _ = example.get(INPUT_FIELDS.TOKEN_EMBEDDINGS)

        # Initialize output
        output: Dict[str, Any] = {}

        # Tokenize
        if self.model.tokenizer is None:
            raise ValueError(ERRORS.NO_TOKENIZER)

        tokens, token_ids = get_tokens_from_model(
            self.model,
            text,
            prepend_bos=self.config.prepend_bos,
            max_length=self.config.max_seq_length,
        )
        output[OUTPUT_FIELDS.TOKENS] = clean_token_strings(tokens)

        # Prepare input
        input_tokens = token_ids.unsqueeze(0).to(self.config.device)

        # Run with cache to get all activations
        with torch.no_grad():
            result, cache = self.model.run_with_cache(
                input_tokens,
                return_type="logits",
            )
            # Ensure logits is a tensor (run_with_cache returns Output type)
            logits: torch.Tensor = (
                result if isinstance(result, torch.Tensor) else torch.tensor(result)
            )

        # Top-K predictions
        output[OUTPUT_FIELDS.TOP_K_TOKENS] = self._get_top_k_per_position(logits, len(tokens))

        # Embeddings
        if self.config.output_embeddings:
            output.update(self._extract_embeddings(cache, len(tokens)))

        # Attention
        if self.config.output_attention:
            output.update(self._extract_attention(cache))

        # Gradients (requires separate forward pass with gradients enabled)
        if self.config.compute_gradients:
            output.update(self._compute_gradients(text, example))

        return output

    def _get_top_k_per_position(
        self,
        logits: torch.Tensor,
        seq_len: int,
    ) -> List[List[tuple]]:
        """Get top-k predictions for each position.

        Args:
            logits: Model logits [batch, pos, vocab].
            seq_len: Sequence length.

        Returns:
            List of lists of (token, probability) tuples.
        """
        results = []
        # Ensure logits is a tensor (handle Output type from run_with_cache)
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits)
        probs = torch.softmax(logits[0], dim=-1)

        for pos in range(seq_len):
            top_probs, top_indices = torch.topk(probs[pos], self.config.top_k)
            pos_results = []
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
                if self.model.tokenizer is not None:
                    token_str = self.model.tokenizer.decode([idx])
                else:
                    token_str = f"<{idx}>"
                pos_results.append((token_str, prob))
            results.append(pos_results)

        return results

    def _extract_embeddings(
        self,
        cache: Any,
        seq_len: int,
    ) -> Dict[str, Any]:
        """Extract embeddings from the activation cache.

        Args:
            cache: Activation cache from forward pass.
            seq_len: Sequence length.

        Returns:
            Dictionary of embedding arrays.
        """
        output = {}

        # Input embeddings (from hook_embed)
        input_emb = cache["hook_embed"][0]  # [seq_len, d_model]
        output[OUTPUT_FIELDS.INPUT_EMBEDDINGS] = tensor_to_numpy(input_emb)

        # Final layer embeddings
        final_layer = self._model_info["n_layers"] - 1
        final_resid = cache[f"blocks.{final_layer}.hook_resid_post"][0]

        # CLS-style (first token)
        output[OUTPUT_FIELDS.CLS_EMBEDDING] = tensor_to_numpy(final_resid[0])

        # Mean pooled
        output[OUTPUT_FIELDS.MEAN_EMBEDDING] = tensor_to_numpy(final_resid.mean(dim=0))

        # Per-layer embeddings
        for layer in self._get_embedding_layers():
            resid = cache[f"blocks.{layer}.hook_resid_post"][0]
            # Use mean pooled embedding for the layer
            field_name = OUTPUT_FIELDS.LAYER_EMB_TEMPLATE.format(layer=layer)
            output[field_name] = tensor_to_numpy(resid.mean(dim=0))

        return output

    def _extract_attention(
        self,
        cache: Any,
    ) -> Dict[str, Any]:
        """Extract attention patterns from the activation cache.

        Args:
            cache: Activation cache from forward pass.

        Returns:
            Dictionary of attention pattern arrays.
        """
        output = {}

        for layer in range(self._model_info["n_layers"]):
            # Get attention pattern for this layer
            attn = extract_attention_from_cache(cache, layer, head=None, batch_idx=0)
            # attn shape: [num_heads, query_pos, key_pos]
            field_name = OUTPUT_FIELDS.LAYER_ATTENTION_TEMPLATE.format(layer=layer)
            output[field_name] = attn

        return output

    def _compute_gradients(
        self,
        text: str,
        example: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute token gradients for salience.

        Args:
            text: Input text.
            example: Full input example (may contain target_mask).

        Returns:
            Dictionary with gradient arrays.
        """
        output = {}

        # Tokenize
        tokens, token_ids = get_tokens_from_model(
            self.model,
            text,
            prepend_bos=self.config.prepend_bos,
            max_length=self.config.max_seq_length,
        )
        input_tokens = token_ids.unsqueeze(0).to(self.config.device)

        # Get target mask if provided
        target_mask = example.get(INPUT_FIELDS.TARGET_MASK)

        # Get embeddings with gradient tracking
        with torch.enable_grad():
            # Get input embeddings and make them a leaf tensor for gradients
            embed = self.model.embed(input_tokens).detach().clone()
            embed.requires_grad_(True)

            # Add positional embeddings if applicable
            if self.model.cfg.positional_embedding_type == "standard":
                pos_embed = self.model.pos_embed(input_tokens)
                residual = embed + pos_embed
            else:
                residual = embed

            # Forward through the rest of the model
            logits = self.model(residual, start_at_layer=0)

            # Compute loss or target logit
            if target_mask is not None:
                # Use masked tokens as targets
                # For now, use simple next-token prediction loss
                pass

            # Use last token prediction as target
            target_idx = token_ids[-1].item()  # Predict last token
            target_logit = logits[0, -2, target_idx]  # Logit at second-to-last position

            # Backward pass
            target_logit.backward()

            # Get gradients - now embed is a leaf tensor so grad should be populated
            if embed.grad is None:
                # Fallback: return zeros if gradients couldn't be computed
                gradients = torch.zeros_like(embed[0])
            else:
                gradients = embed.grad[0]  # [seq_len, d_model]

        # Return the full gradient tensor - LIT computes norms internally
        # TokenGradients expects shape [num_tokens, emb_dim]
        output[OUTPUT_FIELDS.GRAD_L2] = tensor_to_numpy(gradients)
        output[OUTPUT_FIELDS.GRAD_DOT_INPUT] = tensor_to_numpy(gradients)

        return output

    def max_minibatch_size(self) -> int:
        """Return the maximum batch size for prediction.

        Returns:
            Maximum batch size.
        """
        return self.config.batch_size

    def get_embedding_table(self) -> tuple:
        """Return the token embedding table.

        Required by LIT for certain generators like HotFlip.

        Returns:
            Tuple of (vocab_list, embedding_matrix) where vocab_list is
            a list of token strings and embedding_matrix is [vocab, d_model].
        """
        # Get the embedding matrix from the model
        embed_weight = self.model.embed.W_E.detach().cpu().numpy()

        # Get vocabulary list - use tokenizer's vocab size to avoid index errors
        if self.model.tokenizer is not None:
            # Use the tokenizer's actual vocabulary size
            tokenizer_vocab_size = len(self.model.tokenizer)
            # Use the smaller of embedding size and tokenizer vocab size
            vocab_size = min(embed_weight.shape[0], tokenizer_vocab_size)
            vocab_list = []
            for i in range(vocab_size):
                try:
                    token = self.model.tokenizer.decode([i])
                    vocab_list.append(token)
                except Exception:
                    vocab_list.append(f"<{i}>")
            # Truncate embedding matrix to match vocab_list
            embed_weight = embed_weight[:vocab_size]
        else:
            vocab_list = [f"<{i}>" for i in range(embed_weight.shape[0])]

        return vocab_list, embed_weight

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        config: Optional[HookedTransformerLITConfig] = None,
        **model_kwargs,
    ) -> "HookedTransformerLIT":
        """Create a LIT wrapper from a pretrained model name.

        Convenience method that loads the HookedTransformer model
        and wraps it for LIT.

        Args:
            model_name: Name of the pretrained model (e.g., "gpt2-small").
            config: Optional wrapper configuration.
            **model_kwargs: Additional arguments for HookedTransformer.from_pretrained.

        Returns:
            HookedTransformerLIT wrapper instance.

        Example:
            >>> lit_model = HookedTransformerLIT.from_pretrained("gpt2-small")  # doctest: +SKIP
        """
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(model_name, **model_kwargs)
        return cls(model, config=config)


# If LIT is available, register as a proper LIT BatchedModel subclass
if _LIT_AVAILABLE:

    class HookedTransformerLITBatched(lit_model.BatchedModel):  # type: ignore[union-attr]
        """Batched version of HookedTransformerLIT for better performance.

        This class implements the BatchedModel interface for efficient
        batch processing. Use this for production deployments.
        """

        def __init__(
            self,
            model: Any,
            config: Optional[HookedTransformerLITConfig] = None,
        ):
            """Initialize the batched LIT wrapper.

            Args:
                model: TransformerLens HookedTransformer model.
                config: Optional configuration.
            """
            # Use the non-batched wrapper internally
            self._wrapper = HookedTransformerLIT(model, config)
            self.model = model
            self.config = self._wrapper.config

        def description(self) -> str:
            return self._wrapper.description()

        @classmethod
        def init_spec(cls) -> Dict[str, Any]:
            return HookedTransformerLIT.init_spec()

        def input_spec(self) -> Dict[str, Any]:
            return self._wrapper.input_spec()

        def output_spec(self) -> Dict[str, Any]:
            return self._wrapper.output_spec()

        def max_minibatch_size(self) -> int:
            return self._wrapper.max_minibatch_size()

        def predict_minibatch(  # type: ignore[union-attr]
            self,
            inputs,  # type: ignore[override]
        ):
            """Run prediction on a minibatch of inputs.

            Args:
                inputs: List of input dictionaries.

            Returns:
                List of output dictionaries.
            """
            # For now, just iterate (can be optimized for true batching)
            return [self._wrapper._predict_single(ex) for ex in inputs]  # type: ignore[union-attr]

        @classmethod
        def from_pretrained(
            cls,
            model_name: str,
            config: Optional[HookedTransformerLITConfig] = None,
            **model_kwargs,
        ) -> "HookedTransformerLITBatched":
            """Create a batched LIT wrapper from a pretrained model.

            Args:
                model_name: Name of the pretrained model.
                config: Optional wrapper configuration.
                **model_kwargs: Additional arguments for model loading.

            Returns:
                HookedTransformerLITBatched instance.
            """
            from transformer_lens import HookedTransformer

            model = HookedTransformer.from_pretrained(model_name, **model_kwargs)
            return cls(model, config=config)
