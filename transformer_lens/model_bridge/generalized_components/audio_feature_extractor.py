"""Bridge component for audio CNN feature extractors (HuBERT, wav2vec2)."""

from typing import Any, Dict, Optional

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class AudioFeatureExtractorBridge(GeneralizedComponent):
    """Wraps the multi-layer 1D CNN that converts raw waveforms into features.

    hook_in captures the raw waveform, hook_out captures extracted features.
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
        super().__init__(name, config, submodules=submodules or {})

    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """input_values: [batch, num_samples] -> [batch, conv_dim, num_frames]"""
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
