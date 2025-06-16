"""Type definitions for architecture adapters."""

from typing import TypeAlias

import torch.nn as nn

# Type aliases for paths
TransformerLensPath: TypeAlias = str  # Path in TransformerLens format (e.g. "blocks.0.attn")
RemotePath: TypeAlias = str  # Path in the remote model format (e.g. "transformer.h.0.attn")

# Component mapping types
RemoteImport: TypeAlias = tuple[RemotePath, type]  # Path and component class
BlockMapping: TypeAlias = tuple[
    RemotePath, type, dict
]  # Maps a block path, bridge type, and its components
ComponentMapping: TypeAlias = dict[
    TransformerLensPath, RemoteImport | BlockMapping
]  # Complete component mapping

RemoteModel: TypeAlias = nn.Module
RemoteComponent: TypeAlias = nn.Module
