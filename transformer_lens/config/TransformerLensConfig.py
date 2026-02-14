"""TransformerLens Configuration.

Module with a dataclass for storing the configuration of a
:class:`transformer_lens.model_bridge.TransformerBridge` model.
"""

from __future__ import annotations

import inspect
import pprint
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch


@dataclass
class TransformerLensConfig:
    """
    Configuration class for TransformerLens bridge components.

    This class contains only the configuration parameters that are actually used
    by the system. It serves as a minimal base configuration.

    Args:
        # Core model architecture parameters
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_layers (int): The number of transformer blocks.
        n_ctx (int): The maximum sequence length.
        n_heads (int): The number of attention heads. If not specified, will be set to d_model // d_head.
        d_mlp (int, optional): The dimensionality of the feedforward mlp network.
        d_vocab (int): The size of the vocabulary. Defaults to -1, which means not set.

        # Device configuration
        device (str, optional): The device to use for the model. Defaults to 'cuda' if available, else 'cpu'.

        # Attention configuration
        use_attn_result (bool): Whether to explicitly calculate the amount each head adds to the residual stream.
        use_split_qkv_input (bool): Whether to explicitly calculate the input of each head separately.

        # Tokenizer configuration
        default_prepend_bos (bool): Default behavior of whether to prepend the BOS token.

        # Positional embedding configuration
        positional_embedding_type (str): The positional embedding used.

        # GQA configuration
        n_key_value_heads (int, optional): The number of groups of heads that use the same key and value matrix.
    """

    # Core model architecture parameters
    d_model: int
    d_head: int
    n_layers: int
    n_ctx: int
    n_heads: int = -1
    d_mlp: Optional[int] = None
    d_vocab: int = -1

    # Device configuration
    device: Optional[str] = None

    # Attention configuration
    use_attn_result: bool = False
    use_split_qkv_input: bool = False

    # Tokenizer configuration
    default_prepend_bos: bool = True

    # Positional embedding configuration
    positional_embedding_type: str = "standard"

    # GQA configuration
    n_key_value_heads: Optional[int] = None

    # Attention only model
    attn_only: bool = False

    # Gated MLP
    gated_mlp: bool = False

    # Normalization configuration
    uses_rms_norm: bool = False

    # Epsilon for normalization
    eps: float = 1e-5

    # Layer norm folding activated
    layer_norm_folding: bool = False

    # Activation function
    act_fn: str = "relu"

    # Normalization type
    normalization_type: Optional[str] = "LN"

    # Number of experts
    num_experts: Optional[int] = None

    # Number of experts per token
    experts_per_token: Optional[int] = None

    # Final RMS norm
    final_rms: bool = False

    # Model dtype for LayerNormPre compatibility
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        """Post-initialization processing and validation."""
        # Set n_heads if not specified
        if self.n_heads == -1:
            self.n_heads = self.d_model // self.d_head
            if not self.d_model % self.d_head == 0:
                raise ValueError(
                    f"d_model ({self.d_model}) must be divisible by d_head ({self.d_head})"
                )

        # Set device if not specified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set d_mlp if not specified
        if self.d_mlp is None:
            self.d_mlp = self.d_model * 4

    @classmethod
    def unwrap(cls, config: Union[Dict, "TransformerLensConfig"]) -> "TransformerLensConfig":
        """
        Convenience function to avoid duplicate code from a common way config is passed to various components.
        """
        return TransformerLensConfig.from_dict(config) if isinstance(config, Dict) else config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `TransformerLensConfig` from a Python dictionary of parameters.
        Only includes fields that are defined in the TransformerLensConfig dataclass.
        """
        # Get the field names from the dataclass
        valid_fields = set(inspect.signature(cls).parameters.keys())

        # Filter the config dict to only include valid fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return self.__dict__.copy()

    def __repr__(self) -> str:
        """String representation of the config."""
        return "TransformerLensConfig:\n" + pprint.pformat(self.to_dict())
