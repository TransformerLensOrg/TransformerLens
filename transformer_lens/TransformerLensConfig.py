"""TransformerLens Config.

Base configuration class for TransformerLens models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class TransformerLensConfig:
    """Base configuration class for TransformerLens models.
    
    This class provides the core configuration parameters needed for any transformer model
    in the TransformerLens framework. It serves as the base class for more specific
    configuration classes like HookedTransformerConfig.
    
    Args:
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_layers (int): The number of transformer blocks.
        n_ctx (int): The maximum sequence length.
        n_heads (int): The number of attention heads. If not specified, will be set to d_model // d_head.
        d_mlp (int, optional): The dimensionality of the feedforward mlp network. Defaults to 4 * d_model.
        d_vocab (int): The size of the vocabulary. Defaults to -1, which means not set.
        act_fn (str, optional): The activation function to use. Always lowercase.
        eps (float): The epsilon value to use for layer normalization. Defaults to 1e-5.
        device (str): The device to use for the model. Defaults to 'cuda' if available, else 'cpu'.
        dtype (torch.dtype): The model's dtype. Defaults to torch.float32.
    """
    
    d_model: int
    d_head: int
    n_layers: int
    n_ctx: int
    n_heads: int = -1
    d_mlp: Optional[int] = None
    d_vocab: int = -1
    act_fn: Optional[str] = None
    eps: float = 1e-5
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> TransformerLensConfig:
        """Create a config from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            A new TransformerLensConfig instance
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return self.__dict__

    def __repr__(self) -> str:
        """String representation of the config."""
        return "TransformerLensConfig:\n" + "\n".join(f"  {k}: {v}" for k, v in self.to_dict().items()) 