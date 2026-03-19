"""Audio feature extractor bridge component.

This module contains the bridge component for HuBERT's CNN feature extractor,
which converts raw audio waveforms into feature representations.
"""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AudioFeatureExtractorBridge(GeneralizedComponent):
    """Bridge for audio CNN feature extractors (HuBERT, wav2vec2).

    Wraps the multi-layer 1D convolutional feature extractor that converts
    raw audio waveforms into feature representations. Provides hook_in
    (raw waveform) and hook_out (extracted features) for interpretability.
    """

    hook_aliases = {
        "hook_audio_features": "hook_out",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ):
        """Initialize the audio feature extractor bridge.

        Args:
            name: The name of this component (e.g., "hubert.feature_extractor")
            config: Optional configuration object
            submodules: Dictionary of submodules to register
        """
        super().__init__(name, config, submodules=submodules or {})

    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the CNN feature extractor.

        Args:
            input_values: Raw audio waveform [batch, num_samples]
            **kwargs: Additional arguments

        Returns:
            Extracted features [batch, conv_dim, num_frames]
        """
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        input_values = self.hook_in(input_values)
        output = self.original_component(input_values, **kwargs)

        if isinstance(output, tuple):
            output = (self.hook_out(output[0]),) + output[1:]
        else:
            output = self.hook_out(output)

        return output
