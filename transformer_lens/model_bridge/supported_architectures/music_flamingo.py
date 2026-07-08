"""Music Flamingo architecture adapter.

NVIDIA's Music Flamingo / Audio Flamingo Next
(``MusicFlamingoForConditionalGeneration``): Audio Flamingo 3's layout
plus a rotary temporal embedding inside the conditioning path
(``pos_emb``), which rides along opaquely with the delegated audio tower
and projector.
"""

from transformer_lens.model_bridge.supported_architectures.audio_flamingo3 import (
    AudioFlamingo3ArchitectureAdapter,
)


class MusicFlamingoArchitectureAdapter(AudioFlamingo3ArchitectureAdapter):
    """Architecture adapter for MusicFlamingoForConditionalGeneration models."""
