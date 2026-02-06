"""T5 architecture adapter."""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    PosEmbedBridge,
    RMSNormalizationBridge,
    T5BlockBridge,
    UnembeddingBridge,
)


class T5ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for T5 models.

    T5 is an encoder-decoder model with:
    - Shared embeddings
    - Encoder stack (self-attention + FFN)
    - Decoder stack (self-attention + cross-attention + FFN)
    - Language modeling head
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize the T5 architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "relative_positional_bias"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        self.weight_processing_conversions = {}

        self.component_mapping = {
            # Shared embeddings
            "embed": EmbeddingBridge(name="shared"),
            # Encoder positional embeddings (relative attention bias)
            "pos_embed": PosEmbedBridge(
                name="encoder.block.0.layer.0.SelfAttention.relative_attention_bias"
            ),
            # Encoder blocks (2 layers: self-attn, FFN)
            "encoder_blocks": T5BlockBridge(
                name="encoder.block",
                config=self.cfg,
                is_decoder=False,
                submodules={
                    "ln1": RMSNormalizationBridge(name="layer.0.layer_norm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="layer.0.SelfAttention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q"),
                            "k": LinearBridge(name="k"),
                            "v": LinearBridge(name="v"),
                            "o": LinearBridge(name="o"),
                        },
                    ),
                    "ln2": RMSNormalizationBridge(name="layer.1.layer_norm", config=self.cfg),
                    "mlp": MLPBridge(
                        name="layer.1.DenseReluDense",
                        submodules={
                            "in": LinearBridge(name="wi"),
                            "out": LinearBridge(name="wo"),
                        },
                    ),
                },
            ),
            # Encoder final layer norm
            "encoder_ln_final": RMSNormalizationBridge(
                name="encoder.final_layer_norm", config=self.cfg
            ),
            # Decoder positional embeddings (relative attention bias)
            "decoder_pos_embed": PosEmbedBridge(
                name="decoder.block.0.layer.0.SelfAttention.relative_attention_bias"
            ),
            # Decoder blocks (3 layers: self-attn, cross-attn, FFN)
            "decoder_blocks": T5BlockBridge(
                name="decoder.block",
                config=self.cfg,
                is_decoder=True,
                submodules={
                    "ln1": RMSNormalizationBridge(name="layer.0.layer_norm", config=self.cfg),
                    "self_attn": AttentionBridge(
                        name="layer.0.SelfAttention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q"),
                            "k": LinearBridge(name="k"),
                            "v": LinearBridge(name="v"),
                            "o": LinearBridge(name="o"),
                        },
                    ),
                    "ln2": RMSNormalizationBridge(name="layer.1.layer_norm", config=self.cfg),
                    "cross_attn": AttentionBridge(
                        name="layer.1.EncDecAttention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="q"),
                            "k": LinearBridge(name="k"),
                            "v": LinearBridge(name="v"),
                            "o": LinearBridge(name="o"),
                        },
                    ),
                    "ln3": RMSNormalizationBridge(name="layer.2.layer_norm", config=self.cfg),
                    "mlp": MLPBridge(
                        name="layer.2.DenseReluDense",
                        submodules={
                            "in": LinearBridge(name="wi"),
                            "out": LinearBridge(name="wo"),
                        },
                    ),
                },
            ),
            # Decoder final layer norm
            "decoder_ln_final": RMSNormalizationBridge(
                name="decoder.final_layer_norm", config=self.cfg
            ),
            # Language modeling head
            "unembed": UnembeddingBridge(name="lm_head"),
        }
