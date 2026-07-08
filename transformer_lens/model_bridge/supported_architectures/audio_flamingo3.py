"""Audio Flamingo 3 architecture adapter.

NVIDIA's Audio Flamingo 3 (``AudioFlamingo3ForConditionalGeneration``):
a fine-tuned Whisper encoder at ``audio_tower``, a two-layer projector at
``multi_modal_projector``, and a full Qwen2ForCausalLM at
``language_model`` — the identical layout the Qwen2-Audio adapter maps,
under the same module names, so this is a pure subclass.
"""

from transformer_lens.model_bridge.supported_architectures.qwen2_audio import (
    Qwen2AudioArchitectureAdapter,
)


class AudioFlamingo3ArchitectureAdapter(Qwen2AudioArchitectureAdapter):
    """Architecture adapter for AudioFlamingo3ForConditionalGeneration models."""
