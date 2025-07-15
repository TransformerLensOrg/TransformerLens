"""Model bridge module.

This module provides functionality to bridge between different model architectures.
"""

from transformer_lens.model_bridge.architecture_adapter import (
    ArchitectureAdapter,
)

from transformer_lens.model_bridge.bridge import (
    TransformerBridge,
)
from transformer_lens.model_bridge.component_setup import (
    replace_remote_component,
    set_original_components,
    setup_blocks_bridge,
    setup_components,
    setup_submodules,
)
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LayerNormBridge,
    LinearBridge,
    MLPBridge,
    MoEBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.types import (
    ComponentMapping,
    HookFunction,
    HookRegistry,
    RemoteComponent,
    RemoteModel,
    RemotePath,
    TransformerLensPath,
)

from transformer_lens.model_bridge.sources import transformers


def __getattr__(name):
    """Lazy import for ArchitectureAdapterFactory to avoid circular imports."""
    if name == "ArchitectureAdapterFactory":
        from transformer_lens.factories.architecture_adapter_factory import (
            ArchitectureAdapterFactory,
        )

        return ArchitectureAdapterFactory
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "ArchitectureAdapter",
    "ArchitectureAdapterFactory",
    "TransformerBridge",
    "AttentionBridge",
    "BlockBridge",
    "EmbeddingBridge",
    "LayerNormBridge",
    "LinearBridge",
    "MLPBridge",
    "MoEBridge",
    "UnembeddingBridge",
    "replace_remote_component",
    "set_original_components",
    "setup_blocks_bridge",
    "setup_components",
    "setup_submodules",
    # Type definitions
    "ComponentMapping",
    "HookFunction",
    "HookRegistry",
    "RemoteComponent",
    "RemoteModel",
    "RemotePath",
    "TransformerLensPath",
]
