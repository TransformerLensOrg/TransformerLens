"""LLava-OneVision architecture adapter.

Same module hierarchy as base LLava; SigLIP encoder and Qwen2 backbone
are handled dynamically by the base adapter and HuggingFace's forward().
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)


class LlavaOnevisionArchitectureAdapter(LlavaArchitectureAdapter):
    """Architecture adapter for LLaVA-OneVision models."""

    def prepare_model(self, hf_model: Any) -> None:
        """Fix weight tying when text_config and top-level config disagree.

        Some checkpoints have tie_word_embeddings=True in text_config but False
        at the top level, leaving lm_head randomly initialized.
        """
        if not hasattr(hf_model, "lm_head") or not hasattr(hf_model, "model"):
            return
        language_model = getattr(hf_model.model, "language_model", None)
        if language_model is None:
            return
        embed = getattr(language_model, "embed_tokens", None)
        if embed is None:
            return

        # Check if text config expects tied weights but top-level config doesn't
        text_config = getattr(hf_model.config, "text_config", None)
        if text_config is not None and getattr(text_config, "tie_word_embeddings", False):
            if not getattr(hf_model.config, "tie_word_embeddings", True):
                hf_model.lm_head.weight = embed.weight
