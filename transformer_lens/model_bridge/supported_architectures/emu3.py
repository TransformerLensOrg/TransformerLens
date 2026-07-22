"""Emu3 architecture adapter.

BAAI's Emu3 (``Emu3ForConditionalGeneration``, native in transformers):
unified next-token prediction over text AND image tokens — images are
VQ-VAE-quantized into the shared vocabulary, so one Llama-shaped decoder
(at ``model.text_model``) generates both modalities. The VQ tokenizer
(``model.vqmodel``) is not mapped: it has no forward (encode/decode only)
and images become ordinary tokens before the decoder runs — reach it via
bridge.original_model.
"""

from typing import Any

from transformer_lens.model_bridge.supported_architectures.llama import (
    LlamaArchitectureAdapter,
)


class Emu3ArchitectureAdapter(LlamaArchitectureAdapter):
    """Architecture adapter for Emu3ForConditionalGeneration models."""

    _testing_lm_attr = "model.text_model"
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Emu3 architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        self._reprefix_components("model.", "model.text_model.")
