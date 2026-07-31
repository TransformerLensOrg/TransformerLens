"""Compatibility re-export shim.

Historically, :class:`TransformerBridge` lived in this module. The class was
split into three (``BridgeCore`` framework-agnostic parent, ``TransformerBridge``
torch-backed, ``RemoteBridge`` non-torch) across their own files; this shim
preserves the original import path:

    from transformer_lens.model_bridge.bridge import TransformerBridge
"""
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.model_bridge.remote_bridge import RemoteBridge
from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

__all__ = ["BridgeCore", "RemoteBridge", "TransformerBridge"]
