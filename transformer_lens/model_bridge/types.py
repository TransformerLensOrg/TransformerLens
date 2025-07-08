"""Type definitions for architecture adapters."""

from typing import Any, Callable, Dict, Optional, TypeAlias, Union

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
BridgeComponent: TypeAlias = GeneralizedComponent

# Configuration types
BridgeConfig: TypeAlias = Any  # Configuration object for bridge components
ComponentConfig: TypeAlias = Union[BridgeConfig, Dict[str, Any], None]

# Component mapping types
RemoteImport: TypeAlias = tuple[RemotePath, type[BridgeComponent]]  # Path and bridge component class
RemoteImportWithConfig: TypeAlias = tuple[RemotePath, tuple[type[BridgeComponent], ComponentConfig]]  # Path, bridge class, and config

# Block mapping types - for handling transformer blocks with sub-components
BlockMapping: TypeAlias = tuple[
    RemotePath,  # Path to the block container (e.g. "transformer.h")
    type[BridgeComponent],  # Bridge component class for the block
    Dict[str, Union[RemoteImport, RemoteImportWithConfig]]  # Sub-component mapping
]
BlockMappingWithConfig: TypeAlias = tuple[
    RemotePath,  # Path to the block container
    tuple[type[BridgeComponent], ComponentConfig],  # Bridge class and config
    Dict[str, Union[RemoteImport, RemoteImportWithConfig]]  # Sub-component mapping
]

# Complete component mapping
ComponentMapping: TypeAlias = Dict[
    TransformerLensPath,
    Union[
        RemoteImport,
        RemoteImportWithConfig,
        BlockMapping,
        BlockMappingWithConfig
    ]
]

# Function signatures for component creation
ComponentCreationFunction: TypeAlias = Callable[
    [RemoteImport, RemoteModel, Any, str, Optional[ComponentConfig]],
    BridgeComponent
]

# Hook-related types
HookFunction: TypeAlias = Callable[[torch.Tensor, Any], torch.Tensor]
HookRegistry: TypeAlias = Dict[str, list[HookFunction]]
