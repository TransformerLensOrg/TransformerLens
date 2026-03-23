"""LLava-NeXT architecture adapter.

Same module hierarchy as base LLava; high-res tiling differences are
handled internally by HuggingFace's forward().
"""

from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)


class LlavaNextArchitectureAdapter(LlavaArchitectureAdapter):
    """Architecture adapter for LLaVA-NeXT (1.6) models."""

    pass
