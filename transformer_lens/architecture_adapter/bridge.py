"""Bridge between HuggingFace and HookedTransformer models."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from transformer_lens.architecture_adapter.conversion_utils.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.architecture_adapter.generalized_components import (
    AttentionBridge,
    MLPBridge,
)
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


@dataclass
class Block:
    """A transformer block in the bridge."""

    ln1: nn.Module
    attn: AttentionBridge
    ln2: nn.Module
    mlp: MLPBridge


class TransformerBridge:
    """Bridge between HuggingFace and HookedTransformer models.
    
    This class provides a standardized interface to access components of a transformer
    model, regardless of the underlying architecture. It uses an architecture adapter
    to map between the HookedTransformer and HuggingFace model structures.
    """

    def __init__(self, model: PreTrainedModel, adapter: ArchitectureAdapter):
        """Initialize the bridge.
        
        Args:
            model: The HuggingFace model to bridge
            adapter: The architecture adapter to use
        """
        self.model = model
        self.adapter = adapter
        self.cfg = adapter.cfg
        
        # Get components from the adapter
        self.embed = adapter.get_component(model, "embed")
        self.blocks = []
        
        # Build blocks
        for i in range(self.cfg.n_layers):
            # Get block components
            ln1 = adapter.get_component(model, f"blocks.{i}.ln1")
            ln2 = adapter.get_component(model, f"blocks.{i}.ln2")
            
            # Get attention and wrap with bridge
            attn = adapter.get_component(model, f"blocks.{i}.attn")
            if not isinstance(attn, AttentionBridge):
                attn = AttentionBridge(attn, f"blocks.{i}.attn")
            
            # Get MLP and wrap with bridge
            mlp = adapter.get_component(model, f"blocks.{i}.mlp")
            if not isinstance(mlp, MLPBridge):
                mlp = MLPBridge(mlp, f"blocks.{i}.mlp")
            
            # Create block
            block = Block(ln1=ln1, attn=attn, ln2=ln2, mlp=mlp)
            self.blocks.append(block)
            
        # Get final components
        self.ln_final = adapter.get_component(model, "ln_final")
        self.unembed = adapter.get_component(model, "unembed")
        
    def __str__(self) -> str:
        """Get a string representation of the bridge.
        
        Returns:
            A string describing the bridge's components
        """
        lines = []
        lines.append("TransformerBridge:")
        lines.append(f"  embed: {type(self.embed).__name__}")
        lines.append(f"  ln_final: {type(self.ln_final).__name__}")
        lines.append(f"  unembed: {type(self.unembed).__name__}")
        lines.append("  blocks:")
        for i, block in enumerate(self.blocks):
            lines.append(f"    {i}:")
            lines.append(f"      ln1: {type(block.ln1).__name__}")
            lines.append(f"      attn: {type(block.attn).__name__}")
            lines.append(f"      ln2: {type(block.ln2).__name__}")
            lines.append(f"      mlp: {type(block.mlp).__name__}")
        return "\n".join(lines)
        
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Generate text using the underlying model.
        
        Args:
            *args: Positional arguments to pass to the model's generate method
            **kwargs: Keyword arguments to pass to the model's generate method
            
        Returns:
            The generated output from the model
        """
        return self.model.generate(*args, **kwargs) 