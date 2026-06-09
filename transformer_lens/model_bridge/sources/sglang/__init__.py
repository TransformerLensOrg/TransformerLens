"""SGLang source for TransformerBridge. :func:`boot_sglang` wraps an ``Engine``
in a :class:`RemoteBridge`; patterns mirror :mod:`sources.vllm` so version-drift
fixes flow between them."""
from __future__ import annotations

from .source import boot_sglang

__all__ = ["boot_sglang"]
