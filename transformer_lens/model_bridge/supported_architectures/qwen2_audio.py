"""Qwen2-Audio architecture adapter.

``Qwen2AudioForConditionalGeneration``: a Whisper-style audio encoder at
``model.audio_tower``, a linear projector at ``model.multi_modal_projector``,
and a Qwen2 text decoder at ``model.language_model`` with a top-level
``lm_head`` (the transformers >= 5.13 layout). Text-only forwards work without
audio; audio features enter via ``input_features``.
"""

from typing import Any

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.qwen2 import (
    Qwen2ArchitectureAdapter,
)


class Qwen2AudioArchitectureAdapter(Qwen2ArchitectureAdapter):
    """Architecture adapter for Qwen2AudioForConditionalGeneration models."""

    _testing_lm_attr = "model.language_model"
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Qwen2-Audio architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        self._reprefix_components("model.", "model.language_model.")

        # The Whisper-style encoder and projector are wrapped opaquely
        # (hook_in/hook_out only) — audio features are injected by the HF
        # forward when input_features is passed.
        self.components["audio_encoder"] = GeneralizedComponent(name="model.audio_tower")
        self.components["audio_projector"] = GeneralizedComponent(
            name="model.multi_modal_projector"
        )
