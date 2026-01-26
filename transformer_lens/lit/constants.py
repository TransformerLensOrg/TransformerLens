"""Constants for the LIT integration module.

This module defines constants used throughout the LIT integration with TransformerLens.
These include default configuration values, field names, and other settings that
ensure consistency across the integration.

Note: LIT (Learning Interpretability Tool) is Google's framework-agnostic tool for
ML model interpretability. See: https://pair-code.github.io/lit/

References:
    - LIT Documentation: https://pair-code.github.io/lit/documentation/
    - LIT API: https://pair-code.github.io/lit/documentation/api
    - TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
"""

from dataclasses import dataclass
from typing import Optional

# =============================================================================
# Field Names - Used in input_spec and output_spec
# =============================================================================


@dataclass(frozen=True)
class InputFieldNames:
    """Field names for model inputs in LIT."""

    # Primary text input
    TEXT: str = "text"
    # Optional pre-tokenized input
    TOKENS: str = "tokens"
    # Optional token embeddings for integrated gradients
    TOKEN_EMBEDDINGS: str = "token_embeddings"
    # Target for gradient computation
    TARGET: str = "target"
    # Gradient target mask (for sequence salience)
    TARGET_MASK: str = "target_mask"


@dataclass(frozen=True)
class OutputFieldNames:
    """Field names for model outputs in LIT."""

    # Tokens (tokenized input)
    TOKENS: str = "tokens"
    # Token IDs
    TOKEN_IDS: str = "token_ids"
    # Logits over vocabulary
    LOGITS: str = "logits"
    # Top-k predicted tokens
    TOP_K_TOKENS: str = "top_k_tokens"
    # Generated text (for autoregressive generation)
    GENERATED_TEXT: str = "generated_text"
    # Probabilities for next token prediction
    PROBAS: str = "probas"
    # Loss per token
    LOSS: str = "loss"
    # Embeddings at specific layer (template)
    LAYER_EMB_TEMPLATE: str = "layer_{layer}/embeddings"
    # CLS-style embedding (first token of final layer)
    CLS_EMBEDDING: str = "cls_embedding"
    # Mean pooled embedding
    MEAN_EMBEDDING: str = "mean_embedding"
    # Attention pattern for layer/head (template)
    ATTENTION_TEMPLATE: str = "layer_{layer}/head_{head}/attention"
    # Full attention tensor per layer
    LAYER_ATTENTION_TEMPLATE: str = "layer_{layer}/attention"
    # Token gradients for salience
    TOKEN_GRADIENTS: str = "token_gradients"
    # Gradient L2 norm (scalar per token)
    GRAD_L2: str = "grad_l2"
    # Gradient dot input (scalar per token)
    GRAD_DOT_INPUT: str = "grad_dot_input"
    # Input token embeddings (for integrated gradients)
    INPUT_EMBEDDINGS: str = "input_embeddings"


# Instantiate as singletons for easy access
INPUT_FIELDS = InputFieldNames()
OUTPUT_FIELDS = OutputFieldNames()

# =============================================================================
# Default Configuration Values
# =============================================================================


@dataclass(frozen=True)
class DefaultConfig:
    """Default configuration values for the LIT wrapper."""

    # Maximum sequence length for tokenization
    MAX_SEQ_LENGTH: int = 512
    # Batch size for inference
    BATCH_SIZE: int = 8
    # Number of top-k tokens to return for predictions
    TOP_K: int = 10
    # Whether to compute and return gradients
    COMPUTE_GRADIENTS: bool = True
    # Whether to return attention patterns
    OUTPUT_ATTENTION: bool = True
    # Whether to return embeddings per layer
    OUTPUT_EMBEDDINGS: bool = True
    # Whether to output all layer embeddings or just final
    OUTPUT_ALL_LAYERS: bool = False
    # Layers to include for embeddings (None = all)
    EMBEDDING_LAYERS: Optional[tuple] = None
    # Whether to prepend BOS token
    PREPEND_BOS: bool = True
    # Device for computation (None = auto-detect)
    DEVICE: Optional[str] = None
    # Whether to use FP16 for memory efficiency
    USE_FP16: bool = False


DEFAULTS = DefaultConfig()

# =============================================================================
# Hook Point Names - TransformerLens specific
# =============================================================================


@dataclass(frozen=True)
class HookPointNames:
    """Common hook point names used in TransformerLens.

    These correspond to the hook points defined in HookedTransformer where
    we can intercept and extract intermediate activations.
    """

    # Embedding hooks
    HOOK_EMBED: str = "hook_embed"
    HOOK_POS_EMBED: str = "hook_pos_embed"
    HOOK_TOKENS: str = "hook_tokens"

    # Residual stream hooks (template - requires layer number)
    RESID_PRE_TEMPLATE: str = "blocks.{layer}.hook_resid_pre"
    RESID_POST_TEMPLATE: str = "blocks.{layer}.hook_resid_post"
    RESID_MID_TEMPLATE: str = "blocks.{layer}.hook_resid_mid"

    # Attention hooks (template)
    ATTN_OUT_TEMPLATE: str = "blocks.{layer}.hook_attn_out"
    ATTN_PATTERN_TEMPLATE: str = "blocks.{layer}.attn.hook_pattern"
    ATTN_SCORES_TEMPLATE: str = "blocks.{layer}.attn.hook_attn_scores"

    # QKV hooks
    Q_TEMPLATE: str = "blocks.{layer}.attn.hook_q"
    K_TEMPLATE: str = "blocks.{layer}.attn.hook_k"
    V_TEMPLATE: str = "blocks.{layer}.attn.hook_v"

    # MLP hooks
    MLP_OUT_TEMPLATE: str = "blocks.{layer}.hook_mlp_out"
    MLP_PRE_TEMPLATE: str = "blocks.{layer}.mlp.hook_pre"
    MLP_POST_TEMPLATE: str = "blocks.{layer}.mlp.hook_post"

    # Final layer norm
    LN_FINAL: str = "ln_final.hook_normalized"


HOOK_POINTS = HookPointNames()

# =============================================================================
# LIT Type Mappings
# =============================================================================

# Mapping from TransformerLens output types to LIT types
# This helps with automatic spec generation
LIT_TYPE_MAPPING = {
    "text": "TextSegment",
    "tokens": "Tokens",
    "embeddings": "Embeddings",
    "token_embeddings": "TokenEmbeddings",
    "attention": "AttentionHeads",
    "gradients": "TokenGradients",
    "multiclass": "MulticlassPreds",
    "regression": "RegressionScore",
    "generated_text": "GeneratedText",
    "top_k_tokens": "TokenTopKPreds",
}

# =============================================================================
# Error Messages
# =============================================================================


@dataclass(frozen=True)
class ErrorMessages:
    """Standard error messages for the LIT integration."""

    NO_TOKENIZER: str = (
        "HookedTransformer has no tokenizer. "
        "Please load a model with a tokenizer or set one manually."
    )
    INVALID_MODEL: str = (
        "Model must be an instance of HookedTransformer. " "Got: {model_type}"
    )
    LIT_NOT_INSTALLED: str = (
        "LIT (lit-nlp) is not installed. " "Please install it with: pip install lit-nlp"
    )
    INCOMPATIBLE_INPUT: str = (
        "Input does not match the expected input_spec. "
        "Expected fields: {expected}, got: {actual}"
    )
    BATCH_SIZE_MISMATCH: str = "Batch size mismatch. Expected {expected}, got {actual}"


ERRORS = ErrorMessages()

# =============================================================================
# LIT Server Defaults
# =============================================================================


@dataclass(frozen=True)
class ServerConfig:
    """Default configuration for the LIT server."""

    # Default port for LIT server
    DEFAULT_PORT: int = 5432
    # Default host
    DEFAULT_HOST: str = "localhost"
    # Page title
    DEFAULT_TITLE: str = "TransformerLens + LIT"
    # Development mode (hot reload)
    DEV_MODE: bool = False
    # Warm start (load examples on startup)
    WARM_START: bool = True
    # Maximum examples to load
    MAX_EXAMPLES: int = 1000


SERVER_CONFIG = ServerConfig()
