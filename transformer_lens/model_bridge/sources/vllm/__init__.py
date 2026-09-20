"""vLLM source for TransformerBridge.

Provides :func:`boot_vllm`, which constructs a vLLM ``LLM`` and wraps its
inner ``nn.Module`` in a :class:`TransformerBridge`. vLLM drives the forward
pass (PagedAttention, ``torch.compile``, CUDA graphs); the bridge surface is
populated from GPU buffers written by hooks the plugin installs pre-compile.
"""
from __future__ import annotations

from .source import boot_vllm

__all__ = ["boot_vllm"]
