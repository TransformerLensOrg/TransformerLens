"""Registry of per-architecture vLLM overlays."""
from __future__ import annotations

from typing import Dict

from .base import AdapterOverlay
from .llama import LlamaVLLMOverlay

VLLM_OVERLAYS: Dict[str, AdapterOverlay] = {
    "LlamaForCausalLM": LlamaVLLMOverlay(),
}

__all__ = ["VLLM_OVERLAYS", "AdapterOverlay"]
