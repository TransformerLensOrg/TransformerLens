"""GLM-ASR architecture adapter.

Z.ai's GLM-ASR (``GlmAsrForConditionalGeneration``, GLM-ASR-Nano): a
conv-downsampled audio encoder at ``audio_tower``, a linear projector at
``multi_modal_projector``, and a full LlamaForCausalLM at
``language_model`` — the exact Qwen2-Audio layout with a Llama text stack,
whose module names match Qwen2's, so the Qwen2-Audio mapping applies
unchanged.
"""

from transformer_lens.model_bridge.supported_architectures.qwen2_audio import (
    Qwen2AudioArchitectureAdapter,
)


class GlmAsrArchitectureAdapter(Qwen2AudioArchitectureAdapter):
    """Architecture adapter for GlmAsrForConditionalGeneration models."""
