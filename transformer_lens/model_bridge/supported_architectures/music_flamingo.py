"""Music Flamingo (``MusicFlamingoForConditionalGeneration``) adapter: Audio Flamingo 3
layout plus a delegated rotary temporal embedding in the conditioning path."""

from transformer_lens.model_bridge.supported_architectures.audio_flamingo3 import (
    AudioFlamingo3ArchitectureAdapter,
)


class MusicFlamingoArchitectureAdapter(AudioFlamingo3ArchitectureAdapter):
    """Architecture adapter for MusicFlamingoForConditionalGeneration models."""
