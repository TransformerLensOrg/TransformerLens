"""BERT architecture adapter.

This module provides the architecture adapter for BERT models.
"""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    UnembeddingBridge,
)


class BertArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for BERT models."""

    def __init__(self, cfg: Any) -> None:
        """Initialize the BERT architecture adapter.

        Args:
            cfg: The configuration object.
        """
        super().__init__(cfg)

        # Set config variables for weight processing
        self.cfg.normalization_type = "LN"
        self.cfg.positional_embedding_type = "standard"
        self.cfg.final_rms = False
        self.cfg.gated_mlp = False
        self.cfg.attn_only = False

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_head d_model"
                ),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_head d_model"
                ),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "(h d_head) d_model -> h d_head d_model"
                ),
            ),
            "blocks.{i}.attn.q.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head"),
            ),
            "blocks.{i}.attn.k.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head"),
            ),
            "blocks.{i}.attn.v.bias": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(h d_head) -> h d_head"),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion(
                    "d_model (h d_head) -> h d_head d_model"
                ),
            ),
        }

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="bert.embeddings"),
            "pos_embed": PosEmbedBridge(name="bert.embeddings.position_embeddings"),
            "blocks": BlockBridge(
                name="bert.encoder.layer",
                submodules={
                    "ln1": NormalizationBridge(name="attention.output.LayerNorm", config=self.cfg),
                    "ln2": NormalizationBridge(name="output.LayerNorm", config=self.cfg),
                    "attn": AttentionBridge(
                        name="attention",
                        config=self.cfg,
                        submodules={
                            "q": LinearBridge(name="self.query"),
                            "k": LinearBridge(name="self.key"),
                            "v": LinearBridge(name="self.value"),
                            "o": LinearBridge(name="output.dense"),
                        },
                    ),
                    "mlp": MLPBridge(
                        name="intermediate",
                        config=self.cfg,
                        submodules={
                            "in": LinearBridge(name="dense"),
                            "out": LinearBridge(name="../output.dense"),
                        },
                    ),
                },
            ),
            "unembed": UnembeddingBridge(name="cls.predictions"),
            "ln_final": NormalizationBridge(name="bert.pooler.dense", config=self.cfg),
        }
