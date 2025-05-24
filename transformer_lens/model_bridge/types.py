"""Type definitions for architecture adapters."""

from collections.abc import Callable
from typing import Any, TypeAlias

import torch.nn as nn

# Type aliases for paths
TransformerLensPath: TypeAlias = str  # Path in TransformerLens format (e.g. "blocks.0.attn")
RemotePath: TypeAlias = str  # Path in the remote model format (e.g. "transformer.h.0.attn")

# Component mapping types
RemoteImport: TypeAlias = tuple[RemotePath, Callable[..., Any]]  # Path and component factory
ComponentLayer: TypeAlias = dict[
    TransformerLensPath, RemoteImport
]  # Maps TransformerLens components to remote components
BlockMapping: TypeAlias = tuple[RemotePath, ComponentLayer]  # Maps a block and its components
ComponentMapping: TypeAlias = dict[
    TransformerLensPath, RemoteImport | BlockMapping
]  # Complete component mapping

RemoteModel: TypeAlias = nn.Module
RemoteComponent: TypeAlias = nn.Module
