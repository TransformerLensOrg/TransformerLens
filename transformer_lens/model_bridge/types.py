"""Type definitions for architecture adapters."""
from typing import Any, Callable, Dict, TypeAlias

import torch
import torch.nn as nn

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

TransformerLensPath: TypeAlias = str
RemotePath: TypeAlias = str
RemoteModel: TypeAlias = nn.Module
RemoteComponent: TypeAlias = nn.Module
ComponentMapping: TypeAlias = Dict[TransformerLensPath, GeneralizedComponent]
HookFunction: TypeAlias = Callable[[torch.Tensor, Any], torch.Tensor]
HookRegistry: TypeAlias = Dict[str, list[HookFunction]]
