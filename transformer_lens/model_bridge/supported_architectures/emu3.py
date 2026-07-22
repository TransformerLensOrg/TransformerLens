"""Emu3 architecture adapter.

BAAI's Emu3 (``Emu3ForConditionalGeneration``, native in transformers):
unified next-token prediction over text AND image tokens — images are
VQ-VAE-quantized into the shared vocabulary, so one Llama-shaped decoder
(at ``model.text_model``) generates both modalities. The VQ tokenizer
(``model.vqmodel``) and the image-vocabulary mapping ride along opaquely;
pixel_values entering forward are quantized to tokens before the decoder.

The text stack is standard Llama (GQA q/k/v/o, gated SiLU MLP, RMS pre-norms,
shared rotary), so attention uses the full bridge reimplementation with
uniform Q/K/V conversions.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)


class Emu3ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Emu3ForConditionalGeneration models."""

    _testing_lm_attr = "model.text_model"
    _testing_eager = "config"

    def __init__(self, cfg: Any) -> None:
        """Initialize the Emu3 architecture adapter."""
        super().__init__(cfg)

        self.cfg.is_multimodal = True
        self._set_rms_rotary_defaults()

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            # The VQ-VAE image tokenizer (model.vqmodel) is not mapped: it has
            # no forward (encode/decode only) and images become ordinary tokens
            # before the decoder runs. Reach it via bridge.original_model.
            "embed": EmbeddingBridge(name="model.text_model.embed_tokens"),
            "rotary_emb": RotaryEmbeddingBridge(name="model.text_model.rotary_emb"),
            "blocks": BlockBridge(
                name="model.text_model.layers",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": PositionEmbeddingsAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="o_proj"),
                        },
                        requires_attention_mask=True,
                        requires_position_embeddings=True,
                    ),
                    "mlp": self._gated_mlp(),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.text_model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }
