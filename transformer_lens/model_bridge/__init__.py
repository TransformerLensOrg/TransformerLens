"""Model bridge module.

This module provides functionality to bridge between different model architectures.
"""

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.architecture_adapter import (
    ArchitectureAdapter,
)
from transformer_lens.model_bridge.bridge import (
    TransformerBridge,
)
from transformer_lens.model_bridge.component_creation import (
    create_bridged_component,
    replace_remote_component,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    AttentionConfig,
    BlockBridge,
    EmbeddingBridge,
    LayerNormBridge,
    LinearBridge,
    MLPBridge,
    MoEBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.types import (
    BridgeComponent,
    BridgeConfig,
    ComponentConfig,
    ComponentMapping,
    HookFunction,
    HookRegistry,
    RemoteComponent,
    RemoteImport,
    RemoteImportWithConfig,
    RemoteModel,
    RemotePath,
    TransformerLensPath,
)

__all__ = [
    "ArchitectureAdapter",
    "ArchitectureAdapterFactory",
    "TransformerBridge",
    "AttentionBridge",
    "AttentionConfig",
    "BlockBridge",
    "EmbeddingBridge",
    "LayerNormBridge",
    "LinearBridge",
    "MLPBridge",
    "MoEBridge",
    "UnembeddingBridge",
    "create_bridged_component",
    "replace_remote_component",
    # Type definitions
    "BridgeComponent",
    "BridgeConfig",
    "ComponentConfig",
    "ComponentMapping",
    "HookFunction",
    "HookRegistry",
    "RemoteComponent",
    "RemoteImport",
    "RemoteImportWithConfig",
    "RemoteModel",
    "RemotePath",
    "TransformerLensPath",
]
