"""vLLM overlay registry.

One :class:`DecoderOnlyOverlay` handles every vLLM decoder-only model
(Llama / Qwen / Mistral / Gemma / Phi3 / Qwen3 / Kimi / GLM / …). It's the
default for any architecture; per-architecture overlays would only land if a
model breaks vLLM's conventional decoder-only shape.
"""
from __future__ import annotations

from .base import AdapterOverlay
from .decoder_only import DecoderOnlyOverlay

DEFAULT_VLLM_OVERLAY: AdapterOverlay = DecoderOnlyOverlay()


def get_overlay(architecture: str) -> AdapterOverlay:
    """Return the overlay for an architecture; falls back to the decoder-only default."""
    return DEFAULT_VLLM_OVERLAY


__all__ = ["AdapterOverlay", "DEFAULT_VLLM_OVERLAY", "DecoderOnlyOverlay", "get_overlay"]
