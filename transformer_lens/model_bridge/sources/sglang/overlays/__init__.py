"""SGLang overlay registry. ``DecoderOnlyOverlay`` covers Llama/Qwen3/Mistral/
Gemma/Phi3/Kimi/GLM. DeepSeek-V2/V3/V4 share the dotted-path layout but expose
extra non-fireable hooks for MLA + per-expert MoE routing."""
from __future__ import annotations

from .base import AdapterOverlay
from .decoder_only import DecoderOnlyOverlay
from .deepseek import DeepseekOverlay

DEFAULT_SGLANG_OVERLAY: AdapterOverlay = DecoderOnlyOverlay()

# Architecture-string → overlay mapping. Defaults to DecoderOnly when unset.
_OVERLAYS_BY_ARCH = {
    "DeepseekV2ForCausalLM": DeepseekOverlay(),
    "DeepseekV3ForCausalLM": DeepseekOverlay(),
    "DeepseekV4ForCausalLM": DeepseekOverlay(),
}


def get_overlay(architecture: str) -> AdapterOverlay:
    """Return the overlay for an architecture; falls back to the decoder-only default."""
    return _OVERLAYS_BY_ARCH.get(architecture, DEFAULT_SGLANG_OVERLAY)


__all__ = [
    "AdapterOverlay",
    "DEFAULT_SGLANG_OVERLAY",
    "DecoderOnlyOverlay",
    "DeepseekOverlay",
    "get_overlay",
]
