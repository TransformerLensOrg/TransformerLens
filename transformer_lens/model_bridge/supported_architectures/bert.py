"""BERT architecture adapter.

This module provides the architecture adapter for BERT models.
"""

from typing import Any

from transformer_lens.conversion_utils.conversion_steps import (
    HookConversionSet,
    RearrangeHookConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
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

        self.conversion_rules = HookConversionSet(
            {
                "embed.e": "bert.embeddings.word_embeddings.weight",
                "pos_embed.pos": "bert.embeddings.position_embeddings.weight",
                "embed.token_type_embeddings": "bert.embeddings.token_type_embeddings.weight",
                "embed.LayerNorm.weight": "bert.embeddings.LayerNorm.weight",
                "embed.LayerNorm.bias": "bert.embeddings.LayerNorm.bias",
                "blocks.{i}.ln1.w": "bert.encoder.layer.{i}.attention.output.LayerNorm.weight",
                "blocks.{i}.ln1.b": "bert.encoder.layer.{i}.attention.output.LayerNorm.bias",
                "blocks.{i}.ln2.w": "bert.encoder.layer.{i}.output.LayerNorm.weight",
                "blocks.{i}.ln2.b": "bert.encoder.layer.{i}.output.LayerNorm.bias",
                "blocks.{i}.attn.q": (
                    "bert.encoder.layer.{i}.attention.self.query.weight",
                    RearrangeHookConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.k": (
                    "bert.encoder.layer.{i}.attention.self.key.weight",
                    RearrangeHookConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.v": (
                    "bert.encoder.layer.{i}.attention.self.value.weight",
                    RearrangeHookConversion("(h d_head) d_model -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_Q": (
                    "bert.encoder.layer.{i}.attention.self.query.bias",
                    RearrangeHookConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_K": (
                    "bert.encoder.layer.{i}.attention.self.key.bias",
                    RearrangeHookConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.b_V": (
                    "bert.encoder.layer.{i}.attention.self.value.bias",
                    RearrangeHookConversion("(h d_head) -> h d_head"),
                ),
                "blocks.{i}.attn.o": (
                    "bert.encoder.layer.{i}.attention.output.dense.weight",
                    RearrangeHookConversion("d_model (h d_head) -> h d_head d_model"),
                ),
                "blocks.{i}.attn.b_O": "bert.encoder.layer.{i}.attention.output.dense.bias",
                "blocks.{i}.mlp.in": "bert.encoder.layer.{i}.intermediate.dense.weight",
                "blocks.{i}.mlp.b_in": "bert.encoder.layer.{i}.intermediate.dense.bias",
                "blocks.{i}.mlp.out": "bert.encoder.layer.{i}.output.dense.weight",
                "blocks.{i}.mlp.b_out": "bert.encoder.layer.{i}.output.dense.bias",
                "ln_final.w": "bert.pooler.dense.weight",
                "ln_final.b": "bert.pooler.dense.bias",
                "unembed.u": "cls.predictions.transform.dense.weight",
                "unembed.b_U": "cls.predictions.transform.dense.bias",
                "unembed.LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
                "unembed.LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
                "unembed.decoder.weight": "cls.predictions.decoder.weight",
                "unembed.decoder.bias": "cls.predictions.bias",
            }
        )

        # Set up component mapping
        self.component_mapping = {
            "embed": EmbeddingBridge(name="bert.embeddings"),
            "pos_embed": PosEmbedBridge(name="bert.embeddings.position_embeddings"),
            "blocks": BlockBridge(
                name="bert.encoder.layer",
                submodules={
                    "ln1": NormalizationBridge(name="attention.output.LayerNorm", config=self.cfg),
                    "ln2": NormalizationBridge(name="output.LayerNorm", config=self.cfg),
                    "attn": AttentionBridge(name="attention", config=self.cfg),
                    "mlp": MLPBridge(name="intermediate"),
                },
            ),
            "unembed": UnembeddingBridge(name="cls.predictions"),
            "ln_final": NormalizationBridge(name="bert.pooler.dense", config=self.cfg),
        }
