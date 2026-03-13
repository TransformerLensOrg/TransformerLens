"""LLava-OneVision architecture adapter.

LlavaOnevisionForConditionalGeneration shares the same module hierarchy
as LlavaForConditionalGeneration (vision_tower, multi_modal_projector,
language_model, lm_head).  The differences — SigLIP vision encoder,
Qwen2 language backbone, anyres tiling, and video support — are either
handled dynamically by the base adapter (vision encoder selection) or
internally by HuggingFace's forward().
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.llava import (
    LlavaArchitectureAdapter,
)


class LlavaOnevisionArchitectureAdapter(LlavaArchitectureAdapter):
    """Architecture adapter for LLaVA-OneVision models."""

    def prepare_model(self, hf_model: Any) -> None:
        """Fix weight tying for LlavaOnevision models.

        Some LlavaOnevision checkpoints (e.g. llava-onevision-qwen2-0.5b-ov-hf)
        have tie_word_embeddings=True in the text config but False at the top level.
        This causes lm_head.weight to be randomly initialized instead of tied to
        embed_tokens. We detect and fix this by copying embed weights to lm_head.
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
