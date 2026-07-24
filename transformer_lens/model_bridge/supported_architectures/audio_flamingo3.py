"""Audio Flamingo 3 (``AudioFlamingo3ForConditionalGeneration``) adapter: identical
module layout to Qwen2-Audio, so a pure subclass."""

from transformer_lens.model_bridge.supported_architectures.qwen2_audio import (
    Qwen2AudioArchitectureAdapter,
)


class AudioFlamingo3ArchitectureAdapter(Qwen2AudioArchitectureAdapter):
    """Architecture adapter for AudioFlamingo3ForConditionalGeneration models."""
