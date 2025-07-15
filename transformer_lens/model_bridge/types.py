"""Type definitions for architecture adapters."""

from typing import Any, Callable, Dict, TypeAlias

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

# Type aliases for paths
TransformerLensPath: TypeAlias = str  # Path in TransformerLens format (e.g. "blocks.0.attn")
RemotePath: TypeAlias = str  # Path in the remote model format (e.g. "transformer.h.0.attn")

# Component types
RemoteModel: TypeAlias = nn.Module
RemoteComponent: TypeAlias = nn.Module

# Modern component mapping (instance-based)
# Each bridge component instance contains its remote path as the 'name' attribute
ComponentMapping: TypeAlias = Dict[TransformerLensPath, GeneralizedComponent]

# Hook-related types
HookFunction: TypeAlias = Callable[[torch.Tensor, Any], torch.Tensor]
HookRegistry: TypeAlias = Dict[str, list[HookFunction]]
